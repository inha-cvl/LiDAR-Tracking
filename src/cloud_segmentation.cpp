#include <iostream>
#include <csignal>
#include "cloud_segmentation/cloud_segmentation.hpp"

using PointType = PointXYZIT; // hesai

boost::shared_ptr<PatchWorkpp<PointType>> PatchworkppGroundSeg; // PatchWorkpp
boost::shared_ptr<CloudSegmentation<PointType>> CloudSegmentation_;

pcl::PointCloud<PointType> fullCloud, projectionCloud, cropCloud, groundCloud, nonGroundCloud, undistortionCloud;
pcl::PointCloud<ClusterPointT> downsamplingCloud;
cv::Mat projectionImage;
std::vector<float> pointArray;
vector<pcl::PointCloud<ClusterPointT>> cluster_array;
jsk_recognition_msgs::BoundingBoxArray cluster_bbox_array;

ros::Publisher pub_projection_cloud;
ros::Publisher pub_projection_image;
ros::Publisher pub_crop_cloud;
ros::Publisher pub_ground;
ros::Publisher pub_non_ground;
ros::Publisher pub_undistortion_cloud;
ros::Publisher pub_point_array;
ros::Publisher pub_downsampling_cloud;
ros::Publisher pub_cluster_array;
ros::Publisher pub_cluster_box;

tf2_ros::Buffer tf_buffer;

ros::Time input_stamp;
double t1,t2,t3,t4,t5,t6,t7,t8;

std::string lidar_topic, imu_topic, lidar_frame, target_frame, world_frame;

void callbackIMU(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    CloudSegmentation_->updateImu(msg_in);
}

void callbackCloud(const sensor_msgs::PointCloud2::Ptr &cloud_msg)
{
    input_stamp = cloud_msg->header.stamp; // input_stamp
    
    CloudSegmentation_->msgToPointCloud(cloud_msg, fullCloud);

    // CloudSegmentation_->projectPointCloud(fullCloud, projectionCloud, t1);
    // pub_projection_cloud.publish(cloud2msg(projectionCloud, input_stamp, lidar_frame));

    // CloudSegmentation_->convertPointCloudToImage(projectionCloud, projectionImage, t2);
    // pub_projection_image.publish(image2msg(projectionImage, input_stamp, lidar_frame));

    CloudSegmentation_->cropPointCloud(fullCloud, cropCloud, t3);
    // pub_crop_cloud.publish(cloud2msg(cropCloud, input_stamp, lidar_frame));

    CloudSegmentation_->removalGroundPointCloud(cropCloud, nonGroundCloud, groundCloud, t4);
    // pub_non_ground.publish(cloud2msg(nonGroundCloud, input_stamp, lidar_frame));
    // pub_ground.publish(cloud2msg(groundCloud, input_stamp, lidar_frame));

    CloudSegmentation_->undistortPointCloud(nonGroundCloud, undistortionCloud, t5);
    pub_undistortion_cloud.publish(cloud2msg(undistortionCloud, input_stamp, lidar_frame));

    CloudSegmentation_->adaptiveVoxelClustering(undistortionCloud, cluster_array, t7);
    // pub_cluster_array.publish(cluster2msg(cluster_array, input_stamp, lidar_frame));
    CloudSegmentation_->fittingLShape(cluster_array, cluster_bbox_array, t8);
    pub_cluster_box.publish(bba2msg(cluster_bbox_array, input_stamp, lidar_frame));

    std::cout << "\033[2J" << "\033[" << 10 << ";" << 30 << "H" << std::endl;
    std::cout << "projection : " << t1 << " sec" << std::endl;
    std::cout << "converstion : " << t2 << " sec" << std::endl;
    std::cout << "crop : " << t3 << " sec" << std::endl;
    std::cout << "ground removal : " << t4 << " sec" << std::endl;
    std::cout << "undistortion : " << t5 << " sec" << std::endl;
    std::cout << "downsampling : " << t6 << " sec" << std::endl;
    std::cout << "clustering : " << t7 << " sec" << std::endl;
    std::cout << "lshape fitting : " << t8 << " sec" << std::endl;
}

int main(int argc, char**argv) {

    ros::init(argc, argv, "cloud_segmentation");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    tf2_ros::TransformListener tf_listener(tf_buffer);

    pnh.param<string>("lidar_topic", lidar_topic, "/lidar_points");
    pnh.param<string>("imu_topic", imu_topic, "/ublox/imu_meas");
    pnh.param<string>("lidar_frame", lidar_frame, "hesai_lidar");

    cout << "Operating cloud segementation..." << endl;
    PatchworkppGroundSeg.reset(new PatchWorkpp<PointType>(&pnh));
    CloudSegmentation_.reset(new CloudSegmentation<PointType>(pnh));

    pub_projection_cloud  = pnh.advertise<sensor_msgs::PointCloud2>("projectioncloud", 1, true);
    pub_projection_image = nh.advertise<sensor_msgs::Image>("projectedimage", 1, true);
    pub_crop_cloud  = pnh.advertise<sensor_msgs::PointCloud2>("cropcloud", 1, true);
    pub_ground      = pnh.advertise<sensor_msgs::PointCloud2>("ground", 1, true);
    pub_non_ground  = pnh.advertise<sensor_msgs::PointCloud2>("nonground", 1, true);
    pub_undistortion_cloud = pnh.advertise<sensor_msgs::PointCloud2>("undistortioncloud", 1, true);
    // pub_point_array = pnh.advertise<std_msgs::Float32MultiArray>("pointarray", 1, true);
    pub_downsampling_cloud  = pnh.advertise<sensor_msgs::PointCloud2>("downsampledcloud", 1, true);
    pub_cluster_array  = pnh.advertise<sensor_msgs::PointCloud2>("cluster_array", 1, true);
    pub_cluster_box = pnh.advertise<jsk_recognition_msgs::BoundingBoxArray>("cluster_box", 1, true);
    
    ros::Subscriber sub_cloud = nh.subscribe(lidar_topic, 1, callbackCloud);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 1000, callbackIMU);
    
    ros::spin();

    return 0;
}