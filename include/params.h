#pragma once
#include "point_type/os_point.h"
#include "point_type/hesai_point.h"

// Hesai
int V_SCAN = 128; // Pandar 64 or OT 128
int H_SCAN = 1800;
float ang_res_x = 360.0 / float(H_SCAN);
float ang_res_y = 40.0 / V_SCAN;
float ang_bottom = 25;
using PointType = PointXYZIT;
std::string cloud_topic = "/lidar_points";
std::string imu_topic = "ublox/imu_meas"; // novatel : imu/data_raw | mgi : ublox/imu_meas
std::string frameID = "hesai_lidar"; //
std::string target_frame = "ego_car";
std::string world_frame = "world";
// ring crop
bool crop_ring = false;
uint16_t ring = 2; // if (point.ring % ring == 0)
// intensity crop
bool crop_intensity = false;
float intensity = 10;
// downsampling
float leaf_size_x = 0.2f;
float leaf_size_y = 0.2f;
float leaf_size_z = 0.2f;
// Filtering Cluster
float minClusterSizeX = 0.1; // 0.2
float maxClusterSizeX = 1.5; // 13
float minClusterSizeY = 0.2; // 0.3
float maxClusterSizeY = 1.5; // 13
float minClusterSizeZ = 0.1; // 0.3
float maxClusterSizeZ = 1.5;
// Euclidean Clustering
float clusterTolerance = 0.3; // 더 올리면 전방 차량 클러스터링 못 함
int minSize = 10;
int maxSize = 2000; // hesai 기준으로 이 정도는 해야 버스 인식
// Depth Clustering
float vertical_resolution = 5.0f; // 가상의 수직 해상도
float horizontal_resolution = 1.5f; // 가상의 수평 해상도
int lidar_lines = 64; // LiDAR 라인 수
int cluster_size = 3; // 최소 클러스터 크기
// Adaptive Clustering
float start_tolerance = 0.5;
float delta_tolerance = 0.05;
const int max_region = 120;
const int number_region = 5;
float threshIOU = 0.1; // clustering and integration
// L shape fitting
float projection_range = 0.2; // *2

// OS1-128
// int V_SCAN = 128;
// int H_SCAN = 1024;
// float ang_res_x = 360.0 / float(H_SCAN);
// float ang_res_y = 45.0 / V_SCAN;
// float ang_bottom = 22.5+0.1;
// using PointType = ouster_ros::Point;
// std::string cloud_topic = "/ouster/points";
// std::string imu_topic = "ouster/imu";
// std::string frameID = "os_sensor";
// std::string target_frame = "ego_car";
// std::string world_frame = "world";
// // downsampling
// float leaf_size_x = 0.1f;
// float leaf_size_y = 0.1f;
// float leaf_size_z = 0.1f;
// // Filtering Cluster
// float minClusterSizeX = 0.1;
// float maxClusterSizeX = 10;
// float minClusterSizeY = 0.1;
// float maxClusterSizeY = 10;
// float minClusterSizeZ = 0.05;
// float maxClusterSizeZ = 3;
// // Euclidean Clustering
// float clusterTolerance = 1.0;
// int minSize = 3;
// int maxSize = 2000;
// // Depth Clustering
// float vertical_resolution = 4.5f; // 가상의 수직 해상도
// float horizontal_resolution = 1.2f; // 가상의 수평 해상도
// int lidar_lines = 128; // LiDAR 라인 수
// int cluster_size = 3; // 최소 클러스터 크기
// // Adaptive Clustering
// float start_tolerance = 0.6;
// float delta_tolerance = 0.04;
// const int max_region = 50;
// const int number_region = 10;
// float threshIOU = 0.1; // clustering and integration
// // L shape fitting
// float projection_range = 0.2; // *2

// OS1-16
// int V_SCAN = 16;
// int H_SCAN = 1024;
// float ang_res_x = 360.0 / float(H_SCAN);
// float ang_res_y = 33.2 / V_SCAN;
// float ang_bottom = 11.6+0.1;
// using PointType = ouster_ros::Point;
// std::string cloud_topic = "/os1_cloud_node1/points";
// std::string imu_topic = "os1_cloud_node1/imu";
// std::string frameID = "sensor1/os_sensor";

// ROI
float MAX_X = 120; // 120
float MIN_X = -5; // 0
float MAX_Y = 40.0; // 2.5
float MIN_Y = -40.0; // 2.5
float MAX_Z =  -0.3; // -1.0
float MIN_Z = -1.9; // -1.5
float max_x = 2;
float min_x = -2;
float max_y = 0.8;
float min_y = -0.8;
float max_z = -1.5;
float min_z = -1.7;
float R = 50;

// integration
double distance_threshold = 1.0; // Distance threshold for calculating IOU