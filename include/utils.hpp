#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <std_msgs/Float32MultiArray.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf/tf.h>

#include "point_type/os_point.h"
#include "point_type/hesai_point.h"

#include <pcl/common/pca.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>

#include <cstdint>
#include "sophus/so3.hpp"
#include "sophus/se3.hpp"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <jsoncpp/json/json.h>

#include "undistortion/gyr_int.h"
#include "lshape/lshaped_fitting.h"
#include "track/track.h"

#include <message_filters/cache.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

template<typename PointT>
sensor_msgs::PointCloud2 cloud2msg(const pcl::PointCloud<PointT> &cloud, 
                                    const ros::Time &stamp, const std::string &frame_id)
{
    sensor_msgs::PointCloud2 cloud_ROS;
    pcl::toROSMsg(cloud, cloud_ROS);
    cloud_ROS.header.stamp = stamp;
    cloud_ROS.header.frame_id = frame_id;
    return cloud_ROS;
}

sensor_msgs::ImagePtr image2msg(const cv::Mat& image, const ros::Time &stamp, const std::string &frame_id)
{
    cv_bridge::CvImage cv_image;
    cv_image.header.stamp = stamp; 
    cv_image.header.frame_id = frame_id;
    cv_image.encoding = "mono8"; 
    cv_image.image = image;
    return cv_image.toImageMsg();
}

std_msgs::Float32MultiArray array2msg(const std::vector<float> &array, const ros::Time &stamp, const std::string &frame_id)
{
    std_msgs::Float32MultiArray array_msg;
    
    array_msg.data = array;

    // 헤더 설정 (timestamp와 frame_id 저장)
    array_msg.layout.dim.resize(2);
    array_msg.layout.dim[0].label = "timestamp";
    array_msg.layout.dim[0].size = stamp.sec;
    array_msg.layout.dim[0].stride = stamp.nsec;
    
    array_msg.layout.dim[1].label = frame_id;
    array_msg.layout.dim[1].size = array.size();
    array_msg.layout.dim[1].stride = 1;
    array_msg.layout.data_offset = 0;

    return array_msg;
}

template<typename PointT>
sensor_msgs::PointCloud2 cluster2msg(const std::vector<pcl::PointCloud<PointT>>& cluster_array, const ros::Time& stamp, const std::string& frame_id) 
{
    pcl::PointCloud<PointT> combined_cloud;

    for (const auto& cluster : cluster_array) {
        combined_cloud += cluster;
    }

    sensor_msgs::PointCloud2 output_msg;
    pcl::toROSMsg(combined_cloud, output_msg);
    output_msg.header.stamp = stamp;
    output_msg.header.frame_id = frame_id;

    return output_msg;
}

jsk_recognition_msgs::BoundingBoxArray bba2msg(const jsk_recognition_msgs::BoundingBoxArray bba, 
                                                const ros::Time &stamp, const std::string &frame_id)
{
    jsk_recognition_msgs::BoundingBoxArray bba_ROS;
    bba_ROS.header.stamp = stamp;
    bba_ROS.header.frame_id = frame_id;
    bba_ROS.boxes = bba.boxes;
    return bba_ROS;
}

visualization_msgs::MarkerArray ta2msg(const visualization_msgs::MarkerArray& ta, 
                                        const ros::Time &stamp, const std::string &frame_id)
{
    visualization_msgs::MarkerArray ta_ROS;
    for (const auto& marker : ta.markers)
    {
        visualization_msgs::Marker temp_marker = marker;
        temp_marker.header.stamp = stamp;     
        temp_marker.header.frame_id = frame_id;
        temp_marker.lifetime = ros::Duration(0.05);
        ta_ROS.markers.push_back(temp_marker);
    }

    return ta_ROS;
}

// label에 따라 모델 다른 거 적용 예정, car 말고는 dae 파일이 없음
visualization_msgs::MarkerArray bba2ma(const jsk_recognition_msgs::BoundingBoxArray &bba, 
                                        const ros::Time &stamp, const std::string &frame_id) 
{
    visualization_msgs::MarkerArray marker_array;

    for (size_t i = 0; i < bba.boxes.size(); ++i) {
        const auto& bbox = bba.boxes[i];
        
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.lifetime = ros::Duration(0.05);
        marker.header.stamp = stamp;
        marker.ns = "model";
        marker.id = i;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position = bbox.pose.position;

        // 1 : Car, 2 : Truck, 3 : Motorcycle
        if (bbox.label == 1)
        {
            marker.type = visualization_msgs::Marker::MESH_RESOURCE;
            marker.mesh_resource = "package://lidar_tracking/urdf/car.dae";
            marker.scale.x = 1.7;
            marker.scale.y = 2.1;
            marker.scale.z = 1.6;
            marker.color.r = 0.5;
            marker.color.g = 0.5;
            marker.color.b = 0.5;
            marker.color.a = 2.0;
        }
        else if (bbox.label == 2)
        {
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.scale = bbox.dimensions;
            marker.color.r = 0.0;
            marker.color.g = 0.0;
            marker.color.b = 1.0;
            marker.color.a = 0.0;
        }
        else if (bbox.label == 3)
        {
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.scale.x = 1.5;
            marker.scale.y = 1.5;
            marker.scale.z = 1.5;
            marker.color.r = 1.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
            marker.color.a = 1.0;
        }
        else
        {
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.scale.x = 1.5;
            marker.scale.y = 1.5;
            marker.scale.z = 1.5;
            marker.color.r = 1.0;
            marker.color.g = 1.0;
            marker.color.b = 1.0;
            marker.color.a = 1.0;
        }
        
        tf::Quaternion q_rot(0, 0, M_SQRT1_2, M_SQRT1_2);
        marker.pose.orientation = tf::createQuaternionMsgFromYaw(tf::getYaw(bbox.pose.orientation) + M_PI_2); // dae를 Pandar64 좌표계로 회전

        marker_array.markers.push_back(marker);
    }

    return marker_array;
}

double getBBoxOverlap(jsk_recognition_msgs::BoundingBox bbox1, jsk_recognition_msgs::BoundingBox bbox2)
{
    double boxA[4] = {
        bbox1.pose.position.x - bbox1.dimensions.x / 2.0, 
        bbox1.pose.position.y - bbox1.dimensions.y / 2.0, 
        bbox1.pose.position.x + bbox1.dimensions.x / 2.0, 
        bbox1.pose.position.y + bbox1.dimensions.y / 2.0
    };
    
    double boxB[4] = {
        bbox2.pose.position.x - bbox2.dimensions.x / 2.0, 
        bbox2.pose.position.y - bbox2.dimensions.y / 2.0, 
        bbox2.pose.position.x + bbox2.dimensions.x / 2.0, 
        bbox2.pose.position.y + bbox2.dimensions.y / 2.0
    };
    
    // Calculate the intersection points
    double xA = std::max(boxA[0], boxB[0]);
    double yA = std::max(boxA[1], boxB[1]);
    double xB = std::min(boxA[2], boxB[2]);
    double yB = std::min(boxA[3], boxB[3]);

    // Calculate the intersection area (without the +1 offset)
    double interArea = std::max(0.0, xB - xA) * std::max(0.0, yB - yA);
    if (interArea == 0.0) return 0.0; // No overlap

    // Calculate the area of box A (without the +1 offset)
    double boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);

    // Calculate the overlap ratio
    double overlap = interArea / boxAArea;

    return overlap;
}

template<typename PointT>
std::vector<cv::Point2f> pcl2Point2f(const pcl::PointCloud<PointT>& cloud, double projection_range)
{
    std::vector<cv::Point2f> points;
    points.reserve(cloud.size());

    PointT minPoint, maxPoint;
    pcl::getMinMax3D(cloud, minPoint, maxPoint);
    double center_z = (minPoint.z + maxPoint.z) / 2;
    
    for (const auto& point : cloud)
    {
        if ( (point.z < center_z + projection_range && point.z > center_z - projection_range) ||
              point.z > maxPoint.z - projection_range || point.z < minPoint.z + projection_range )
        {
            // Extract x and y coordinates from PointXYZ and create a cv::Point2f
            cv::Point2f point2f(point.x, point.y);
            points.push_back(point2f);
        }
        // cv::Point2f point2f(point.x, point.y);
        // points.push_back(point2f);
    }
    
    return points;
}

Eigen::Quaterniond calculateRotationBetweenStamps(const std::deque<sensor_msgs::Imu::ConstPtr>& imu_buffer,
                                                  const ros::Time& last_stamp, const ros::Time& input_stamp) 
{
    Sophus::SO3d rotation_increment = Sophus::SO3d(); // Identity rotation

    for (const auto& imu : imu_buffer) {
        ros::Time imu_time = imu->header.stamp;

        if (imu_time > last_stamp && imu_time <= input_stamp) {
            auto it = std::find(imu_buffer.begin(), imu_buffer.end(), imu);
            if (it != imu_buffer.begin()) {
                auto prev_it = std::prev(it);

                // 시간 간격 계산
                double dt1 = input_stamp.toSec() - (*prev_it)->header.stamp.toSec();
                double dt2 = imu_time.toSec() - input_stamp.toSec();
                double w1 = dt2 / (dt1 + dt2 + 1e-9);
                double w2 = dt1 / (dt1 + dt2 + 1e-9);

                // 각속도를 보간하여 평균 각속도를 구함
                Eigen::Vector3d gyr1((*prev_it)->angular_velocity.x, (*prev_it)->angular_velocity.y, (*prev_it)->angular_velocity.z);
                Eigen::Vector3d gyr2(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z);

                Eigen::Vector3d angular_velocity = w1 * gyr1 + w2 * gyr2;

                // 회전 변환을 SO3로 변환하고 누적
                double dt = (imu_time - (*prev_it)->header.stamp).toSec();
                Eigen::Vector3d delta_angle = dt * angular_velocity;
                Sophus::SO3d delta_rotation = Sophus::SO3d::exp(delta_angle);

                // 변환 순서 유지
                rotation_increment = delta_rotation * rotation_increment;
            }
        }
    }

    return rotation_increment.unit_quaternion();
}

geometry_msgs::PoseStamped getENU(const message_filters::Cache<geometry_msgs::PoseStamped>& enu_cache, const ros::Time& input_stamp)
{
    // 가장 가까운 PoseStamped를 찾기 위한 변수 초기화
    geometry_msgs::PoseStamped closest_pose;

    // input_stamp 이전의 가장 가까운 PoseStamped
    boost::shared_ptr<geometry_msgs::PoseStamped const> pose_before = enu_cache.getElemBeforeTime(input_stamp);
    // input_stamp 이후의 가장 가까운 PoseStamped
    boost::shared_ptr<geometry_msgs::PoseStamped const> pose_after = enu_cache.getElemAfterTime(input_stamp);

    // 두 PoseStamped 중 input_stamp와 더 가까운 것을 선택
    if (pose_before && pose_after)
    {
        // time difference 계산
        ros::Duration diff_before = input_stamp - pose_before->header.stamp;
        ros::Duration diff_after = pose_after->header.stamp - input_stamp;

        // 더 가까운 PoseStamped를 선택
        if (diff_before < diff_after)
        {
            closest_pose = *pose_before;
        }
        else
        {
            closest_pose = *pose_after;
        }
    }
    // pose_before만 존재하는 경우
    else if (pose_before)
    {
        closest_pose = *pose_before;
    }
    // pose_after만 존재하는 경우
    else if (pose_after)
    {
        closest_pose = *pose_after;
    }
    else
    {
        // enu_cache에 적절한 값이 없는 경우
        // 캐시 내에서 가장 최근 값을 가져오기 위해 마지막 요소를 찾음
        boost::shared_ptr<geometry_msgs::PoseStamped const> last_pose = enu_cache.getElemAfterTime(ros::Time(0));
        if (last_pose)
        {
            closest_pose = *last_pose;  // 가장 최근 데이터를 선택
            ROS_WARN("No matching PoseStamped found. Returning the most recent data.");
        }
        else
        {
            ROS_WARN("No matching PoseStamped found in the cache, and no recent data available.");
        }
    }

    return closest_pose;
}

std::vector<std::pair<float, float>> map_reader(std::string map_path)
{
    Json::Value root;      
    Json::Reader reader;
    std::vector<std::pair<float, float>> global_path;
    std::string index;

    // ROS 패키지 경로 얻기
    std::string package_path = ros::package::getPath("lidar_tracking");
    std::string json_file_path = package_path + map_path;  // 상대 경로 사용

    std::ifstream t(json_file_path);
    if (!t.is_open()) {
        std::cout << "Failed to open file: " << json_file_path << std::endl;
        return global_path;
    }

    if (!reader.parse(t, root)) {
        std::cout << "Parsing Failed" << std::endl;
        return global_path;
    }

    for (int k = 0; k < root.size(); ++k) {
        index = std::to_string(k);
        double x = root[index][0].asDouble();
        double y = root[index][1].asDouble();
        global_path.emplace_back(x, y);
    }
    
    std::cout << "Global Path Created: " << global_path.size() << std::endl;
    
    return global_path;
}

void transformMsgToEigen(const geometry_msgs::Transform &transform_msg, Eigen::Affine3f &transform) 
{  
    transform =
        Eigen::Translation3f(transform_msg.translation.x,
                             transform_msg.translation.y,
                             transform_msg.translation.z) *
        Eigen::Quaternionf(transform_msg.rotation.w, transform_msg.rotation.x,
                           transform_msg.rotation.y, transform_msg.rotation.z);
}

bool checkTransform(tf2_ros::Buffer &tf_buffer, const std::string &lidar_frame, const std::string &target_frame)
{
    try {
        geometry_msgs::TransformStamped TransformStamped = tf_buffer.lookupTransform(target_frame, lidar_frame, ros::Time(0));
        return true;
    }
    catch (tf2::TransformException &ex) {
        return false;
    }
}

void compareTransforms(const geometry_msgs::TransformStamped &transform1, 
                        const geometry_msgs::TransformStamped &transform2) 
{
    ROS_INFO("Comparing transforms:");

    // Print translation of transform1 and transform2
    ROS_INFO("Transform 1 - Translation: x = %f, y = %f, z = %f",
             transform1.transform.translation.x,
             transform1.transform.translation.y,
             transform1.transform.translation.z);

    ROS_INFO("Transform 2 - Translation: x = %f, y = %f, z = %f",
             transform2.transform.translation.x,
             transform2.transform.translation.y,
             transform2.transform.translation.z);

    // Print rotation of transform1 and transform2
    ROS_INFO("Transform 1 - Rotation: x = %f, y = %f, z = %f, w = %f",
             transform1.transform.rotation.x,
             transform1.transform.rotation.y,
             transform1.transform.rotation.z,
             transform1.transform.rotation.w);

    ROS_INFO("Transform 2 - Rotation: x = %f, y = %f, z = %f, w = %f",
             transform2.transform.rotation.x,
             transform2.transform.rotation.y,
             transform2.transform.rotation.z,
             transform2.transform.rotation.w);

    // Compare translations
    if (std::abs(transform1.transform.translation.x - transform2.transform.translation.x) < 1e-3 &&
        std::abs(transform1.transform.translation.y - transform2.transform.translation.y) < 1e-3 &&
        std::abs(transform1.transform.translation.z - transform2.transform.translation.z) < 1e-3) {
        ROS_INFO("Translations are identical.");
    } else {
        ROS_INFO("Translations are different.");
    }

    // Compare rotations
    if (std::abs(transform1.transform.rotation.x - transform2.transform.rotation.x) < 1e-3 &&
        std::abs(transform1.transform.rotation.y - transform2.transform.rotation.y) < 1e-3 &&
        std::abs(transform1.transform.rotation.z - transform2.transform.rotation.z) < 1e-3 &&
        std::abs(transform1.transform.rotation.w - transform2.transform.rotation.w) < 1e-3) {
        ROS_INFO("Rotations are identical.");
    } else {
        ROS_INFO("Rotations are different.");
    }
}

void clearLogFile(const std::string& file_path) 
{
    std::ofstream file(file_path, std::ios::trunc);
    file.close();
}

// for experiment
void saveTimeToFile(const std::string& timing_file, double time_taken) 
{   
    // std::ofstream file(timing_file, std::ios::app);

    // if (!file.is_open()) {
    //     std::cerr << "Error opening file: " << timing_file << std::endl;
    //     return;
    // }

    // file << time_taken << "\n";
    // file.close();
}

double calculateAverageTime(const std::string& timing_file) 
{
    std::ifstream file(timing_file);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << timing_file << std::endl;
        return -1;
    }

    std::vector<double> times;
    double time;

    // 파일에서 시간을 읽어와 벡터에 저장
    while (file >> time) {
        times.push_back(time);
    }

    file.close();

    if (times.empty()) {
        std::cerr << "No timing data found in file: " << timing_file << std::endl;
        return 0;  // 시간이 없을 경우 0 반환
    }

    // 벡터의 모든 값의 합을 구한 후, 벡터 크기로 나누어 평균 계산
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double average = sum / times.size();

    return average;
}

/*
// no use
void EuclideanClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloudIn, 
                            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &outputClusters, double &time_taken)
{
    auto start = std::chrono::steady_clock::now();

    if (cloudIn->points.empty()) {
        std::cerr << "Input cloud is empty! <- EuclideanClustering" << std::endl;
        return;
    }

    outputClusters.clear();

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

    // mingu
    tree->setInputCloud(cloudIn);

    std::vector<pcl::PointIndices> clusterIndices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(clusterTolerance);
    ec.setMinClusterSize(minSize);
    ec.setMaxClusterSize(maxSize);
    ec.setSearchMethod(tree);
    // mingu
    ec.setInputCloud(cloudIn);

    ec.extract(clusterIndices);

    for(pcl::PointIndices getIndices: clusterIndices)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudCluster (new pcl::PointCloud<pcl::PointXYZ>);
        for (int index : getIndices.indices)
            // mingu
            cloudCluster->points.push_back (cloudIn->points[index]);

        cloudCluster->width = cloudCluster->points.size();
        cloudCluster->height = 1;
        cloudCluster->is_dense = true;

        // size filtering
        pcl::PointXYZ minPt, maxPt;
        pcl::getMinMax3D(*cloudCluster, minPt, maxPt);
        double clusterSizeX = maxPt.x - minPt.x;
        double clusterSizeY = maxPt.y - minPt.y;
        double clusterSizeZ = maxPt.z - minPt.z;

        if (clusterSizeX > minClusterSizeX && clusterSizeX < maxClusterSizeX
        && clusterSizeY > minClusterSizeY && clusterSizeY < maxClusterSizeY
        && clusterSizeZ > minClusterSizeZ && clusterSizeZ < maxClusterSizeZ) 
        {           
            outputClusters.push_back(cloudCluster);
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
}

#include "depth_clustering/depth_cluster.h"
void depthClustering(const pcl::PointCloud<PointType>::Ptr &cloudIn, 
                    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &outputClusters, double &time_taken)
{
    auto start = std::chrono::steady_clock::now();

    if (cloudIn->points.empty()) {
        std::cerr << "Input cloud is empty!" << std::endl;
        return;
    }

    DepthCluster depthCluster(vertical_resolution, horizontal_resolution, lidar_lines, cluster_size);

    // 입력 포인트 클라우드 설정
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloudInXYZI(new pcl::PointCloud<pcl::PointXYZI>);
    copyPointCloud(*cloudIn, *cloudInXYZI); // PointType에서 XYZI로 변환
    depthCluster.setInputCloud(cloudInXYZI);

    // 클러스터 추출 실행
    auto clusters_indices = depthCluster.getClustersIndex();

    // 결과 클러스터 추출
    outputClusters.clear();
    for (const auto& indices : clusters_indices) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudCluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (int index : indices) {
            // cloudCluster->points.push_back(cloudIn->points[index]);
            pcl::PointXYZ point;
            point.x = cloudIn->points[index].x;
            point.y = cloudIn->points[index].y;
            point.z = cloudIn->points[index].z;
            cloudCluster->points.push_back(point);
        }
        cloudCluster->width = cloudCluster->points.size();
        cloudCluster->height = 1;
        cloudCluster->is_dense = true;
        outputClusters.push_back(cloudCluster);
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
}

// L-shape Fitting 과 비교용
void fittingPCA(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &inputClusters, const ros::Time &input_stamp, 
                jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double &time_taken)
{
    auto start = std::chrono::steady_clock::now();

    output_bbox_array.boxes.clear();

    for (auto cluster : inputClusters)
    {
        pcl::PCA<pcl::PointXYZ> pca;
        pcl::PointXYZ minPoint, maxPoint;
        pcl::getMinMax3D(*cluster, minPoint, maxPoint);

        // Find the oriented bounding box
        // pca.setInputCloud(cluster);
        // Eigen::Vector3f eigen_values = pca.getEigenValues();
        // Eigen::Matrix3f eigen_vectors = pca.getEigenVectors();

        // Create jsk_recognition_msgs::BoundingBox
        jsk_recognition_msgs::BoundingBox bbox;
        bbox.header.stamp = input_stamp;
        bbox.header.frame_id = frameID;
        bbox.pose.position.x = (minPoint.x + maxPoint.x) / 2.0;
        bbox.pose.position.y = (minPoint.y + maxPoint.y) / 2.0;
        bbox.pose.position.z = (minPoint.z + maxPoint.z) / 2.0;
        bbox.dimensions.x = maxPoint.x - minPoint.x;
        bbox.dimensions.y = maxPoint.y - minPoint.y;
        bbox.dimensions.z = maxPoint.z - minPoint.z;
        // Eigen::Quaternionf quat(eigen_vectors);
        // bbox.pose.orientation.x = quat.x();
        // bbox.pose.orientation.y = quat.y();
        // bbox.pose.orientation.z = quat.z();
        // bbox.pose.orientation.w = quat.w();
        bbox.pose.orientation.x = 0;
        bbox.pose.orientation.y = 0;
        bbox.pose.orientation.z = 0;
        bbox.pose.orientation.w = 1;

        output_bbox_array.boxes.push_back(bbox);
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
}
*/