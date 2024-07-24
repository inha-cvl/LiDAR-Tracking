#include <iostream>
#include "cloud_processor/cloud_processor.hpp"

#include <mutex>
#include <thread>
#include <csignal>
#include <condition_variable>

boost::shared_ptr<PatchWorkpp<PointType>> PatchworkppGroundSeg; // PatchWorkpp

pcl::PointCloud<PointType>::Ptr fullCloud(new pcl::PointCloud<PointType>);
pcl::PointCloud<PointType>::Ptr undistortionCloud(new pcl::PointCloud<PointType>);
pcl::PointCloud<PointType>::Ptr projectionCloud(new pcl::PointCloud<PointType>);
pcl::PointCloud<PointType>::Ptr cropCloud(new pcl::PointCloud<PointType>);
pcl::PointCloud<PointType>::Ptr groundCloud(new pcl::PointCloud<PointType>);
pcl::PointCloud<PointType>::Ptr nonGroundCloud(new pcl::PointCloud<PointType>);
pcl::PointCloud<pcl::PointXYZ>::Ptr downsamplingCloud(new pcl::PointCloud<pcl::PointXYZ>);

pcl::PointCloud<PointType>::Ptr testCloud(new pcl::PointCloud<PointType>);

ros::Publisher pub_undistortion_cloud;
ros::Publisher pub_projection_cloud;
ros::Publisher pub_crop_cloud;
ros::Publisher pub_downsampling_cloud;
ros::Publisher pub_ground;
ros::Publisher pub_non_ground;
ros::Publisher pub_cluster_box;

ros::Publisher pub_test;

vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cluster_array;
jsk_recognition_msgs::BoundingBoxArray cluster_bbox_array;

ros::Time input_stamp;
double t1,t2,t3,t4,t5,t6,t7;

/// To notify new data
std::mutex mtx_buffer;
std::condition_variable sig_buffer;
bool b_exit = false;
bool b_reset = false;

/// Buffers for measurements
double last_timestamp_lidar = -1;
std::deque<sensor_msgs::PointCloud2::ConstPtr> lidar_buffer;
double last_timestamp_imu = -1;
std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

std::vector<double> times_t1, times_t2, times_t3, times_t4, times_t5, times_t6, times_t7;

void SigHandle(int sig) 
{
    b_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

void callbackCloud(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{   
    input_stamp = msg->header.stamp;

    const double timestamp = msg->header.stamp.toSec();

    // ROS_DEBUG("get point cloud at time: %.6f", timestamp);
    mtx_buffer.lock();
    if (timestamp < last_timestamp_lidar) 
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = timestamp;

    lidar_buffer.push_back(msg);
    //std::cout << "received point size: " << float(msg->data.size())/float(msg->point_step) << "\n";
    mtx_buffer.unlock();

    sig_buffer.notify_all();

    // pcl::fromROSMsg(*msg, *fullCloud);
    // projectPointCloud(fullCloud, testCloud, t1);
    // pub_test.publish(cloud2msg(*testCloud, ros::Time::now(), frameID));

}

void callbackIMU(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    double timestamp = msg->header.stamp.toSec();
    // ROS_DEBUG("get imu at time: %.6f", timestamp);

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu) {
        ROS_ERROR("imu loop back, clear buffer");
        imu_buffer.clear();
        b_reset = true;
    }
    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);

    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

bool SyncMeasure(MeasureGroup &measgroup) 
{
    if (lidar_buffer.empty() || imu_buffer.empty()) 
    {
        /// Note: this will happen
        return false;
    }

    if (imu_buffer.front()->header.stamp.toSec() > lidar_buffer.back()->header.stamp.toSec()) 
    {
        lidar_buffer.clear();
        ROS_ERROR("clear lidar buffer, only happen at the beginning");
        return false;
    }

    if (imu_buffer.back()->header.stamp.toSec() < lidar_buffer.front()->header.stamp.toSec()) 
    {
        return false;
    }

    /// Add lidar data, and pop from buffer
    measgroup.lidar = lidar_buffer.front();
    lidar_buffer.pop_front();
    double lidar_time = measgroup.lidar->header.stamp.toSec();

    /// Add imu data, and pop from buffer
    measgroup.imu.clear();
    int imu_cnt = 0;
    for (const auto &imu : imu_buffer) 
    {
        double imu_time = imu->header.stamp.toSec();
        if (imu_time <= lidar_time) 
        {
            measgroup.imu.push_back(imu);
            imu_cnt++;
        }
    }
    for (int i = 0; i < imu_cnt; ++i) 
    {
        imu_buffer.pop_front();
    }
    // ROS_DEBUG("add %d imu msg", imu_cnt);

    return true;
}

void ProcessLoop(std::shared_ptr<ImuProcess> p_imu)
{
    ROS_INFO("Start Cloud Segmentation");

    ros::Rate r(1000);
    while (ros::ok()) 
    {
        MeasureGroup meas;
        std::unique_lock<std::mutex> lk(mtx_buffer);
        sig_buffer.wait(lk, [&meas]() -> bool { return SyncMeasure(meas) || b_exit; });
        lk.unlock();

        if (b_exit) {
            ROS_INFO("b_exit=true, exit");
            break;
        }

        if (b_reset) {
            ROS_WARN("reset when rosbag play back");
            p_imu->Reset();
            b_reset = false;
            continue;
        }
        
        p_imu->Process(meas, undistortionCloud, t1); // undistortion
        pub_undistortion_cloud.publish(cloud2msg(*undistortionCloud, input_stamp, frameID));

        // projectPointCloud(undistortionCloud, projectionCloud, t2); // projection
        //pub_projection_cloud.publish(cloud2msg(*projectionCloud, ros::Time::now(), frameID));

        cropPointCloud(undistortionCloud, cropCloud, t3); // crop
        //pub_crop_cloud.publish(cloud2msg(*cropCloud, ros::Time::now(), frameID));

        PatchworkppGroundSeg->estimate_ground(*cropCloud, *groundCloud, *nonGroundCloud, t4); // ground removal
        pub_ground.publish(cloud2msg(*groundCloud, input_stamp, frameID));
        pub_non_ground.publish(cloud2msg(*nonGroundCloud, input_stamp, frameID)); // detection 전달 때문에 input_stamp 사용

        // depthClustering(nonGroundCloud, cluster_array, t6);
        downsamplingPointCloud(nonGroundCloud, downsamplingCloud, t5); // downsampling
        pub_downsampling_cloud.publish(cloud2msg(*downsamplingCloud, ros::Time::now(), frameID));
        adaptiveClustering(downsamplingCloud, cluster_array, t6);
        //EuclideanClustering(downsamplingCloud, cluster_array, t6); // clustering

        fittingLShape(cluster_array, input_stamp, cluster_bbox_array, t7); // L shape fitting
        //fittingPCA(cluster_array, input_stamp, cluster_bbox_array, t7);
        pub_cluster_box.publish(bba2msg(cluster_bbox_array, input_stamp, frameID)); // input_stamp

        std::cout << "\033[2J" << "\033[" << 10 << ";" << 30 << "H" << std::endl;
        std::cout << "undistortion : " << t1 << " sec" << std::endl;
        std::cout << "projection : " << t2 << " sec" << std::endl;
        std::cout << "crop : " << t3 << " sec" << std::endl;
        std::cout << "ground removal : " << t4 << " sec" << std::endl;
        std::cout << "downsampling : " << t5 << " sec" << std::endl;
        std::cout << "clustering : " << t6 << " sec" << std::endl;
        std::cout << "Lshape fitting : " << t7 << " sec" << std::endl;

        // time check
        if (t1 != 0.0) {
            times_t1.push_back(t1);
            times_t2.push_back(t2);
            times_t3.push_back(t3);
            times_t4.push_back(t4);
            times_t5.push_back(t5);
            times_t6.push_back(t6);
            times_t7.push_back(t7);
        }
        
        r.sleep();
    }

    // time check
    auto average = [](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };
    std::cout << "Average undistortion time: " << average(times_t1) << " sec" << std::endl;
    std::cout << "Average projection time: " << average(times_t2) << " sec" << std::endl;
    std::cout << "Average crop time: " << average(times_t3) << " sec" << std::endl;
    std::cout << "Average ground removal time: " << average(times_t4) << " sec" << std::endl;
    std::cout << "Average downsampling time: " << average(times_t5) << " sec" << std::endl;
    std::cout << "Average clustering time: " << average(times_t6) << " sec" << std::endl;
    std::cout << "Average Lshape fitting time: " << average(times_t7) << " sec" << std::endl;

}

int main(int argc, char**argv) {

    ros::init(argc, argv, "cloud_segmentation");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    
    signal(SIGINT, SigHandle);

    ros::Subscriber sub_cloud = nh.subscribe(cloud_topic, 1, callbackCloud);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 1, callbackIMU);

    pub_undistortion_cloud = pnh.advertise<sensor_msgs::PointCloud2>("undistortioncloud", 1, true);
    pub_projection_cloud  = pnh.advertise<sensor_msgs::PointCloud2>("projectioncloud", 1, true);
    pub_crop_cloud  = pnh.advertise<sensor_msgs::PointCloud2>("cropcloud", 1, true);
    pub_downsampling_cloud  = pnh.advertise<sensor_msgs::PointCloud2>("downsamplingcloud", 1, true);
    pub_ground      = pnh.advertise<sensor_msgs::PointCloud2>("ground", 1, true);
    pub_non_ground  = pnh.advertise<sensor_msgs::PointCloud2>("nonground", 1, true);
    pub_cluster_box = pnh.advertise<jsk_recognition_msgs::BoundingBoxArray>("cluster_box", 1, true);

    pub_test = pnh.advertise<sensor_msgs::PointCloud2>("testcloud", 1, true);

    std::shared_ptr<ImuProcess> p_imu(new ImuProcess());
    PatchworkppGroundSeg.reset(new PatchWorkpp<PointType>(&pnh));

    std::thread th_proc(ProcessLoop, p_imu);

    ros::Rate r(1000);
    while (ros::ok()) {
        if (b_exit) break;
        ros::spinOnce();
        r.sleep();
    }

    ROS_INFO("Wait for process loop exit");
    if (th_proc.joinable()) th_proc.join();

    return 0;

}


/*
void callbackCloud2(const sensor_msgs::PointCloud2::Ptr &cloud_msg)
{
    input_stamp = cloud_msg->header.stamp; // ouster input_stamp
    
    pcl::fromROSMsg(*cloud_msg, *fullCloud); // topic to pcl
    
    projectPointCloud(fullCloud, projectionCloud, t2); // projection
    //pub_projection_cloud.publish(cloud2msg(*projectionCloud, ros::Time::now(), frameID));

    cropPointCloud(projectionCloud, cropCloud, t3); // crop
    //pub_crop_cloud.publish(cloud2msg(*cropCloud, ros::Time::now(), frameID));

    PatchworkppGroundSeg->estimate_ground(*cropCloud, *groundCloud, *nonGroundCloud, t4); // ground removal
    pub_ground.publish(cloud2msg(*groundCloud, input_stamp, frameID));
    pub_non_ground.publish(cloud2msg(*nonGroundCloud, input_stamp, frameID)); // detection 전달 때문에 input_stamp 사용

    // depthClustering(nonGroundCloud, cluster_array, t6);
    downsamplingPointCloud(nonGroundCloud, downsamplingCloud, t5); // downsampling
    pub_downsampling_cloud.publish(cloud2msg(*downsamplingCloud, ros::Time::now(), frameID));
    adaptiveClustering(downsamplingCloud, cluster_array, t6);
    //EuclideanClustering(downsamplingCloud, cluster_array, t6); // clustering

    fittingLShape(cluster_array, input_stamp, cluster_bbox_array, t7); // L shape fitting
    pub_cluster_box.publish(bba2msg(cluster_bbox_array, input_stamp, frameID)); // input_stamp

    std::cout << "\033[2J" << "\033[" << 10 << ";" << 30 << "H" << std::endl;
    std::cout << "undistortion : " << t1 << " sec" << std::endl;
    std::cout << "projection : " << t2 << " sec" << std::endl;
    std::cout << "crop : " << t3 << " sec" << std::endl;
    std::cout << "ground removal : " << t4 << " sec" << std::endl;
    std::cout << "downsampling : " << t5 << " sec" << std::endl;
    std::cout << "clustering : " << t6 << " sec" << std::endl;
    std::cout << "Lshape fitting : " << t7 << " sec" << std::endl;
}
int main(int argc, char**argv) {

    ros::init(argc, argv, "cloud_segmentation");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    cout << "Operating cloud segementation..." << endl;
    PatchworkppGroundSeg.reset(new PatchWorkpp<PointType>(&pnh));

    pub_undistortion_cloud = pnh.advertise<sensor_msgs::PointCloud2>("undistortioncloud", 1, true); 
    pub_projection_cloud  = pnh.advertise<sensor_msgs::PointCloud2>("fullcloud", 1, true);
    pub_crop_cloud  = pnh.advertise<sensor_msgs::PointCloud2>("cropcloud", 1, true);
    pub_downsampling_cloud  = pnh.advertise<sensor_msgs::PointCloud2>("clusteringcloud", 1, true);
    pub_ground      = pnh.advertise<sensor_msgs::PointCloud2>("ground", 1, true);
    pub_non_ground  = pnh.advertise<sensor_msgs::PointCloud2>("nonground", 1, true);
    pub_cluster_box = pnh.advertise<jsk_recognition_msgs::BoundingBoxArray>("cluster_box", 1, true);
    
    ros::Subscriber sub_cloud = nh.subscribe(cloud_topic, 1, callbackCloud2);
    
    ros::spin();
    return 0;
}
*/