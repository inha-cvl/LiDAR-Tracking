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

cv::Mat projectionImage;

pcl::PointCloud<PointType>::Ptr testCloud(new pcl::PointCloud<PointType>);

ros::Publisher pub_undistortion_cloud;
ros::Publisher pub_projection_image;
ros::Publisher pub_projection_cloud;
ros::Publisher pub_crop_cloud;
ros::Publisher pub_downsampling_cloud;
ros::Publisher pub_ground;
ros::Publisher pub_non_ground;
ros::Publisher pub_cluster_box;

ros::Publisher pub_test;

vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cluster_array;
jsk_recognition_msgs::BoundingBoxArray cluster_bbox_array;

double last_timestamp_imu = -1;
std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
const size_t MAX_IMU_BUFFER_SIZE = 1000;
Eigen::Quaterniond rotation = Eigen::Quaterniond(1, 0, 0, 0);

ros::Time input_stamp, rotation_stamp;
double t1,t2,t3,t4,t5,t6,t7;

std::vector<double> times_t1, times_t2, times_t3, times_t4, times_t5, times_t6, times_t7;

void callbackIMU(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    double timestamp = msg->header.stamp.toSec();
    // ROS_DEBUG("get imu at time: %.6f", timestamp);

    if (timestamp < last_timestamp_imu) {
        ROS_ERROR("imu loop back, clear buffer");
        imu_buffer.clear();
    }
    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);

    if (input_stamp != rotation_stamp) {
        rotation = calculateRotationBetweenStamps(imu_buffer, rotation_stamp, input_stamp);
    }

    if (imu_buffer.size() > MAX_IMU_BUFFER_SIZE) { imu_buffer.pop_front(); }
    
    rotation_stamp = input_stamp;
}

void callbackCloud(const sensor_msgs::PointCloud2::Ptr &cloud_msg)
{
    input_stamp = cloud_msg->header.stamp; // input_stamp
    
    pcl::fromROSMsg(*cloud_msg, *fullCloud); // topic to pcl
    
    // projectPointCloud(fullCloud, projectionCloud, t2); // projection
    // pub_projection_cloud.publish(cloud2msg(*projectionCloud, ros::Time::now(), frameID));
    //convertPointCloudToImage(projectionCloud, projectionImage);
    // pub_projection_image.publish(image2msg(projectionImage, ros::Time::now(), frameID));

    cropPointCloud(fullCloud, cropCloud, t3); // crop
    //pub_crop_cloud.publish(cloud2msg(*cropCloud, ros::Time::now(), frameID));

    PatchworkppGroundSeg->estimate_ground(*cropCloud, *groundCloud, *nonGroundCloud, t4); // ground removal
    // pub_ground.publish(cloud2msg(*groundCloud, input_stamp, frameID));
    pub_non_ground.publish(cloud2msg(*nonGroundCloud, input_stamp, frameID)); // detection 전달 때문에 input_stamp 사용

    undistortPointCloud(nonGroundCloud, rotation, undistortionCloud, t1);
    // pub_undistortion_cloud.publish(cloud2msg(*undistortionCloud, input_stamp, frameID));

    // depthClustering(nonGroundCloud, cluster_array, t6);
    downsamplingPointCloud(undistortionCloud, downsamplingCloud, t5); // downsampling
    // pub_downsampling_cloud.publish(cloud2msg(*downsamplingCloud, ros::Time::now(), frameID));
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
    pub_projection_image = nh.advertise<sensor_msgs::Image>("projected_image", 1, true);
    pub_projection_cloud  = pnh.advertise<sensor_msgs::PointCloud2>("fullcloud", 1, true);
    pub_crop_cloud  = pnh.advertise<sensor_msgs::PointCloud2>("cropcloud", 1, true);
    pub_downsampling_cloud  = pnh.advertise<sensor_msgs::PointCloud2>("clusteringcloud", 1, true);
    pub_ground      = pnh.advertise<sensor_msgs::PointCloud2>("ground", 1, true);
    pub_non_ground  = pnh.advertise<sensor_msgs::PointCloud2>("nonground", 1, true);
    pub_cluster_box = pnh.advertise<jsk_recognition_msgs::BoundingBoxArray>("cluster_box", 1, true);
    
    ros::Subscriber sub_cloud = nh.subscribe(cloud_topic, 1, callbackCloud);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 1, callbackIMU);
    
    ros::spin();
    return 0;
}
