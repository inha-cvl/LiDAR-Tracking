#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf/tf.h>

#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>

#include "params.h"
#include "undistortion/data_process/data_process.h"
#include "patchworkpp/patchworkpp.hpp"
#include "lshape/lshaped_fitting.h"
#include "track/track.h"
#include "depth_clustering/depth_cluster.h"

template<typename PointT>
sensor_msgs::PointCloud2 cloud2msg(const pcl::PointCloud<PointT>& cloud, 
                                    const ros::Time& stamp, const std::string& frame_id)
{
    sensor_msgs::PointCloud2 cloud_ROS;
    pcl::toROSMsg(cloud, cloud_ROS);
    cloud_ROS.header.stamp = stamp;
    cloud_ROS.header.frame_id = frame_id;
    return cloud_ROS;
}

jsk_recognition_msgs::BoundingBoxArray bba2msg(const jsk_recognition_msgs::BoundingBoxArray bba, 
                                                const ros::Time& stamp, const std::string& frame_id)
{
    jsk_recognition_msgs::BoundingBoxArray bba_ROS;
    bba_ROS.header.stamp = stamp;
    bba_ROS.header.frame_id = frame_id;
    bba_ROS.boxes = bba.boxes;
    return bba_ROS;
}

visualization_msgs::MarkerArray ta2msg(const visualization_msgs::MarkerArray& ta, 
                                        const ros::Time& stamp, const std::string& frame_id)
{
    visualization_msgs::MarkerArray ta_ROS;
    for (const auto& marker : ta.markers)
    {
        visualization_msgs::Marker temp_marker = marker;
        temp_marker.header.stamp = stamp;     
        temp_marker.header.frame_id = frame_id;
        temp_marker.lifetime = ros::Duration(0.2);
        ta_ROS.markers.push_back(temp_marker);
    }

    return ta_ROS;
}

// label에 따라 모델 다른 거 적용 예정, car 말고는 dae 파일이 없음
visualization_msgs::MarkerArray bba2ma(const jsk_recognition_msgs::BoundingBoxArray &bba, 
                                        const ros::Time& stamp, const std::string& frame_id) 
{
    visualization_msgs::MarkerArray marker_array;

    for (size_t i = 0; i < bba.boxes.size(); ++i) {
        const auto& bbox = bba.boxes[i];
        
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.lifetime = ros::Duration(0.2);
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
            marker.scale.x = 1.6;
            marker.scale.y = 2.2;
            marker.scale.z = 1.4;
            marker.color.r = 0.5;
            marker.color.g = 0.5;
            marker.color.b = 0.5;
            marker.color.a = 1.0;
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
	double boxA[4] = {bbox1.pose.position.x - bbox1.dimensions.x/2.0, 
					 bbox1.pose.position.y - bbox1.dimensions.y/2.0, 
					 bbox1.pose.position.x + bbox1.dimensions.x/2.0, 
					 bbox1.pose.position.y + bbox1.dimensions.y/2.0};
 	double boxB[4] = {bbox2.pose.position.x - bbox2.dimensions.x/2.0, 
					 bbox2.pose.position.y - bbox2.dimensions.y/2.0, 
					 bbox2.pose.position.x + bbox2.dimensions.x/2.0, 
					 bbox2.pose.position.y + bbox2.dimensions.y/2.0};
	double xA = std::max(boxA[0], boxB[0]);
	double yA = std::max(boxA[1], boxB[1]);
	double xB = std::min(boxA[2], boxB[2]);
	double yB = std::min(boxA[3], boxB[3]);

	double interArea = max(0.0, xB - xA + 1) * max(0.0, yB - yA + 1);
 	double boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1);

	double overlap = interArea / boxAArea;

	return overlap;
}

std::vector<cv::Point2f> pcl2Point2f(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
    std::vector<cv::Point2f> points;
    points.reserve(cloud->size());

    pcl::PointXYZ minPoint, maxPoint;
    pcl::getMinMax3D(*cloud, minPoint, maxPoint);
    double center_z = (minPoint.z + maxPoint.z) / 2;
    
    for (const auto& point : *cloud)
    {
        if ( (point.z < center_z + projection_range && point.z > center_z - projection_range) ||
              point.z > maxPoint.z - projection_range || point.z < minPoint.z + projection_range )
        {
            // Extract x and y coordinates from PointXYZI and create a cv::Point2f
            cv::Point2f point2f(point.x, point.y);
            points.push_back(point2f);
        }
        // cv::Point2f point2f(point.x, point.y);
        // points.push_back(point2f);
    }
    
    
    return points;
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

bool checkTransform(tf2_ros::Buffer &tf_buffer, const std::string lidar_frame, const std::string target_frame)
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

// hesai
void projectPointCloud(const pcl::PointCloud<PointXYZIT>::Ptr &cloudIn, 
                        pcl::PointCloud<PointXYZIT>::Ptr &cloudOut, double &time_taken)
{
    auto start = std::chrono::steady_clock::now();

    cloudOut -> clear();
    cloudOut->points.resize(V_SCAN * H_SCAN);

    float verticalAngle, horizonAngle;
    size_t rowIdn, columnIdn, index;
    PointXYZIT outPoint;
    
    for (const auto& inPoint : cloudIn->points)
    {   
        outPoint = inPoint;
        outPoint.x = -inPoint.y;
        outPoint.y = inPoint.x;

        verticalAngle = atan2(outPoint.z, sqrt(outPoint.x * outPoint.x + outPoint.y * outPoint.y)) * 180 / M_PI;

        if (verticalAngle < -19){
            rowIdn = (verticalAngle + ang_bottom) / 6;
        }
        else if (verticalAngle < -14){
            rowIdn = (verticalAngle + ang_bottom - 6) / 5 + 2;
        }
        else if (verticalAngle < -6){
            rowIdn = (verticalAngle + ang_bottom - 11) / 1 + 3;
        }
        else if (verticalAngle < 2){
            rowIdn = (verticalAngle + ang_bottom - 19) / 0.167 + 11;
        }
        else if (verticalAngle < 3){
            rowIdn = (verticalAngle + ang_bottom - 27) / 1 + 59;
        }
        else if (verticalAngle < 5){
            rowIdn = (verticalAngle + ang_bottom - 28) / 2 + 60;
        }
        else if (verticalAngle < 11){
            rowIdn = (verticalAngle + ang_bottom - 30) / 3 + 61;
        }
        else
        {
            rowIdn = (verticalAngle + ang_bottom - 36) / 4 + 63;
        }

        if (rowIdn < 0 || rowIdn >= V_SCAN)
            continue;

        horizonAngle = atan2(outPoint.x, outPoint.y) * 180 / M_PI;
        columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + H_SCAN / 2;

        if (columnIdn >= H_SCAN)
            columnIdn -= H_SCAN;

        if (columnIdn < 0 || columnIdn >= H_SCAN)
            continue;

        index = columnIdn + rowIdn * H_SCAN;

        cloudOut->points[index] = outPoint;
    }
    
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
}

// ouster
void projectPointCloud(const pcl::PointCloud<ouster_ros::Point>::Ptr &cloudIn, 
                        pcl::PointCloud<ouster_ros::Point>::Ptr &cloudOut, double &time_taken)
{
    auto start = std::chrono::steady_clock::now();

    cloudOut -> clear();
    cloudOut->points.resize(V_SCAN * H_SCAN);

    float verticalAngle, horizonAngle;
    size_t rowIdn, columnIdn, index;
    ouster_ros::Point outPoint;

    for (const auto& inPoint : cloudIn->points)
    {   
        outPoint = inPoint;

        verticalAngle = atan2(outPoint.z, sqrt(outPoint.x * outPoint.x + outPoint.y * outPoint.y)) * 180 / M_PI;

        rowIdn = (verticalAngle + ang_bottom) / ang_res_y;

        if (rowIdn < 0 || rowIdn >= V_SCAN)
            continue;

        horizonAngle = atan2(outPoint.x, outPoint.y) * 180 / M_PI;
        columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + H_SCAN / 2;

        if (columnIdn >= H_SCAN)
            columnIdn -= H_SCAN;

        if (columnIdn < 0 || columnIdn >= H_SCAN)
            continue;

        index = columnIdn + rowIdn * H_SCAN;
        cloudOut->points[index] = outPoint;

    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
}

void cropPointCloud(const pcl::PointCloud<PointType>::Ptr &cloudIn, 
                    pcl::PointCloud<PointType>::Ptr &cloudOut, double &time_taken)
{   
    auto start = std::chrono::steady_clock::now();

    cloudOut->clear();
    cloudOut->reserve(cloudIn->size());

    for (auto it = cloudIn->begin(); it != cloudIn->end(); ++it)
    {
        const PointType& point = *it;

        // Car exclusion
        if (point.x >= min_x && point.x <= max_x &&
            point.y >= min_y && point.y <= max_y)
        {
            continue;
        }

        // Rectangle
        if (point.x >= MIN_X && point.x <= MAX_X &&
            point.y >= MIN_Y && point.y <= MAX_Y) //&&
            // point.z >= MIN_Z && point.z <= MAX_Z)
        {
            cloudOut->push_back(point);
        }

        // Degree
        // double min_angle_rad = -40 * M_PI / 180.0;
        // double max_angle_rad = 40 * M_PI / 180.0;
        // double angle = std::atan2(point.y, point.x);
        // if (angle >= min_angle_rad && angle <= max_angle_rad)
        // {
        //     cloudOut->push_back(point);
        // }

        // Circle
        /*
        float distance = std::sqrt(point.x * point.x + point.y * point.y);
        if (distance < R)
        {
            cloudOut->push_back(point);
        }
        */
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
}

void downsamplingPointCloud(const pcl::PointCloud<PointType>::Ptr &cloudIn, 
                            pcl::PointCloud<pcl::PointXYZ>::Ptr &cloudOut, double &time_taken)
{
    auto start = std::chrono::steady_clock::now();
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloudIn, *tempCloud);
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_filter;
    voxel_grid_filter.setInputCloud(tempCloud);
    voxel_grid_filter.setLeafSize(leaf_size_x, leaf_size_y, leaf_size_z);
    voxel_grid_filter.filter(*cloudOut);
    
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
}

void EuclideanClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloudIn, 
                            vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &outputClusters, double &time_taken)
{
    auto start = std::chrono::steady_clock::now();

    if (cloudIn->points.empty()) {
        std::cerr << "Input cloud is empty!" << std::endl;
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

// 개발 중,,
void adaptiveClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloudIn, 
                        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &outputClusters, double &time_taken)
{
    auto start = std::chrono::high_resolution_clock::now();

    if (cloudIn->points.empty()) {
        std::cerr << "Input cloud is empty!" << std::endl;
        return;
    }

    outputClusters.clear();

    // Divide the point cloud into nested circular regions
    std::vector<float> regions(max_region, max_region / number_region); // Example: Fill regions with a distance increment of 15m each
    boost::array<std::vector<int>, max_region> indices_array;

    for (int i = 0; i < cloudIn->size(); i++) {
        float distance = cloudIn->points[i].x * cloudIn->points[i].x + cloudIn->points[i].y * cloudIn->points[i].y;
        float range = 0.0;
        for (int j = 0; j < max_region; j++) {
            if (distance > range * range && distance <= (range + regions[j]) * (range + regions[j])) {
                indices_array[j].push_back(i);
                break;
            }
            range += regions[j];
        }
    }

    // Euclidean clustering for each region
    float tolerance = 0.3; // Start tolerance
    for (int i = 0; i < max_region; i++) {
        if (indices_array[i].empty()) continue;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudSegment(new pcl::PointCloud<pcl::PointXYZ>);
        for (int index : indices_array[i]) {
            cloudSegment->points.push_back(cloudIn->points[index]);
        }

        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(cloudSegment);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(tolerance + delta_tolerance * i); // Increment tolerance for farther regions
        ec.setMinClusterSize(minSize);
        ec.setMaxClusterSize(maxSize);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloudSegment);
        ec.extract(cluster_indices);

        for (auto& indices : cluster_indices) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (int idx : indices.indices) {
                cluster->points.push_back(cloudSegment->points[idx]);
            }
            cluster->width = cluster->size();
            cluster->height = 1;
            cluster->is_dense = true;
            
            // // Size filterin
            pcl::PointXYZ minPt, maxPt;
            pcl::getMinMax3D(*cluster, minPt, maxPt);
            double clusterSizeX = maxPt.x - minPt.x;
            double clusterSizeY = maxPt.y - minPt.y;
            double clusterSizeZ = maxPt.z - minPt.z;

            if (clusterSizeX > minClusterSizeX && clusterSizeX < maxClusterSizeX &&
                clusterSizeY > minClusterSizeY && clusterSizeY < maxClusterSizeY &&
                clusterSizeZ > minClusterSizeZ && clusterSizeZ < maxClusterSizeZ) {
                outputClusters.push_back(cluster);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    time_taken = elapsed.count();
}

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

void fittingLShape(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &inputClusters, const ros::Time &input_stamp, 
                jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double &time_taken)
{
    auto start = std::chrono::steady_clock::now();

    output_bbox_array.boxes.clear();

    for (pcl::PointCloud<pcl::PointXYZ>::Ptr cluster : inputClusters)
    {   
        pcl::PointXYZ minPoint, maxPoint;
        pcl::getMinMax3D(*cluster, minPoint, maxPoint);

        // rectangle
        LShapedFIT lshaped;
        std::vector<cv::Point2f> points = pcl2Point2f(cluster); // 3d cluster를 BEV로 투영
        cv::RotatedRect rr = lshaped.FitBox(&points);
        std::vector<cv::Point2f> vertices = lshaped.getRectVertex();
        double yaw = (rr.angle+90)*M_PI/180.0; // degree2radian

        if (rr.center.x == 0.0) { continue; } // 왜 원점에 이상한 클러스터가 생길까?

        // Create jsk_recognition_msgs::BoundingBox
        jsk_recognition_msgs::BoundingBox bbox;
        //bbox.header.stamp = ros::Time::now();
        bbox.header.stamp = input_stamp; // mingu
        bbox.header.frame_id = frameID;
        bbox.pose.position.x = rr.center.x;
        bbox.pose.position.y = rr.center.y;
        bbox.pose.position.z = (minPoint.z + maxPoint.z) / 2.0; // Set z-axis height to be halfway between 0 and 1
        bbox.dimensions.x = rr.size.height;
        bbox.dimensions.y = rr.size.width;
        bbox.dimensions.z = maxPoint.z - minPoint.z;
        bbox.pose.orientation.z = std::sin(yaw / 2.0);
        bbox.pose.orientation.w = std::cos(yaw / 2.0);
        
        output_bbox_array.boxes.push_back(bbox);
    }

    for (int i = 0; i < output_bbox_array.boxes.size(); i++)
    {
        for (int j = 0; j < output_bbox_array.boxes.size(); j++)
        {
            double overlap = getBBoxOverlap(output_bbox_array.boxes[j], output_bbox_array.boxes[i]);
            if(i != j && overlap > threshIOU){
                output_bbox_array.boxes.erase(output_bbox_array.boxes.begin() + j);
            }
        }
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
}
// L-shape Fitting 과 비교용
#include <pcl/common/pca.h>
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

void integrationBbox(const jsk_recognition_msgs::BoundingBoxArray &cluster_bbox_array, 
                    const jsk_recognition_msgs::BoundingBoxArray &deep_bbox_array, 
                    jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double &time_taken)
{
    auto start = std::chrono::steady_clock::now();

    output_bbox_array.boxes.clear();
    // Iterate through each bbox in cluster_bbox_array
    for (const auto &cluster_bbox : cluster_bbox_array.boxes)
    {
        bool keep_cluster_bbox = true;
        // Check IOU with bbox in deep_bbox_array within distance_threshold
        for (const auto &deep_bbox : deep_bbox_array.boxes)
        {
            double overlap = getBBoxOverlap(cluster_bbox, deep_bbox);
            // If IOU exceeds threshold, prioritize deep_bbox
            if (overlap > threshIOU)
            {
                keep_cluster_bbox = false;
                break;
            }
        }
        if (keep_cluster_bbox)
        {
            output_bbox_array.boxes.push_back(cluster_bbox);
        }
    }
    // Add remaining deep_bbox_array to output_bbox_array
    output_bbox_array.boxes.insert(output_bbox_array.boxes.end(), deep_bbox_array.boxes.begin(), deep_bbox_array.boxes.end());

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
}

void tracking(Track &tracker, const jsk_recognition_msgs::BoundingBoxArray &bbox_array, 
                jsk_recognition_msgs::BoundingBoxArray &track_bbox_array, visualization_msgs::MarkerArray &track_text_array,
                const ros::Time &stamp, double &time_taken)
{
    auto start = std::chrono::steady_clock::now();

    track_bbox_array.boxes.clear();
    track_text_array.markers.clear();
    // jsk_recognition_msgs::BoundingBoxArray filtered_bbox_array = tracker.filtering(cluster_bbox_array);
    tracker.predictNewLocationOfTracks();
    tracker.assignDetectionsTracks(bbox_array);
    tracker.assignedTracksUpdate(bbox_array, stamp);
    tracker.unassignedTracksUpdate();
    tracker.deleteLostTracks();
    tracker.createNewTracks(bbox_array);
    pair<jsk_recognition_msgs::BoundingBoxArray, visualization_msgs::MarkerArray> bbox = tracker.displayTrack();
    track_bbox_array = bbox.first;
    track_text_array = bbox.second;

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
}

void transformBbox(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, const std::string &frame_id, 
                    const std::string &target_frame, tf2_ros::Buffer &tf_buffer,
                   jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double &time_taken)
{
    auto start = std::chrono::high_resolution_clock::now();

    output_bbox_array.boxes.clear();

    // 변환 정보 조회
    geometry_msgs::TransformStamped transformStamped;
    try {
        transformStamped = tf_buffer.lookupTransform(target_frame, frame_id, ros::Time(0)); // static tf
    } catch (tf2::TransformException &ex) {
        //ROS_WARN("TF2 error: %s", ex.what());
        output_bbox_array = input_bbox_array;
        return;
    }

    // 각 bounding box 변환 적용
    for (const auto &box : input_bbox_array.boxes) {
        geometry_msgs::PoseStamped input_pose, output_pose;

        input_pose.pose = box.pose;
        tf2::doTransform(input_pose, output_pose, transformStamped);

        jsk_recognition_msgs::BoundingBox transformed_box;
        transformed_box.header = box.header;
        transformed_box.header.frame_id = target_frame;
        transformed_box.pose = output_pose.pose;
        transformed_box.dimensions = box.dimensions;
        transformed_box.value = box.value;
        transformed_box.label = box.label;
        output_bbox_array.boxes.push_back(transformed_box);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
}

// 개발중,,
void correctionBbox(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, const ros::Time &input_stamp, 
                    const std::string target_frame, const std::string world_frame, tf2_ros::Buffer &tf_buffer,
                    jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double &time_taken)
{
    auto start = std::chrono::steady_clock::now();

    output_bbox_array.boxes.clear();

    geometry_msgs::TransformStamped transformStampedAtInput, transformStampedAtStamp;

    try {
        transformStampedAtStamp = tf_buffer.lookupTransform(world_frame, target_frame, input_stamp);
        //transformStampedAtStamp = tf_buffer.lookupTransform(world_frame, target_frame, input_stamp);
        transformStampedAtInput = tf_buffer.lookupTransform(world_frame, target_frame, ros::Time(0)); // 변환에 약 0.1초 시간이 좀 걸림
        
        //compareTransforms(transformStampedAtInput, transformStampedAtStamp);

    } catch (tf2::TransformException &ex) {
        // ROS_WARN("%s", ex.what());
        output_bbox_array = input_bbox_array;
        return;
    }

    tf2::Transform tfAtInput, tfAtStamp, deltaTransform;
    tf2::fromMsg(transformStampedAtInput.transform, tfAtInput);
    tf2::fromMsg(transformStampedAtStamp.transform, tfAtStamp);

    deltaTransform = tfAtStamp.inverse() * tfAtInput;
    geometry_msgs::TransformStamped deltaTransformStamped;
    deltaTransformStamped.transform = tf2::toMsg(deltaTransform);

    // 각 bounding box에 변환 적용
    for (const auto &box : input_bbox_array.boxes) {
        geometry_msgs::PoseStamped input_pose, transformed_pose;

        input_pose.pose = box.pose;
        tf2::doTransform(input_pose, transformed_pose, deltaTransformStamped);

        jsk_recognition_msgs::BoundingBox transformed_box;
        transformed_box.header = box.header;
        transformed_box.header.stamp = input_stamp;
        transformed_box.pose = transformed_pose.pose;
        transformed_box.dimensions = box.dimensions;
        transformed_box.value = box.value;
        transformed_box.label = box.label;
        output_bbox_array.boxes.push_back(transformed_box);

    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
}

void undistortPointCloud(const pcl::PointCloud<PointXYZIT>::Ptr &cloudIn,
                        const ros::Time time_stamp,
                        const std::deque<sensor_msgs::Imu::ConstPtr> &imu_buffer,
                        pcl::PointCloud<PointXYZIT>::Ptr &cloudOut,
                        double &time_taken)
{
    auto start_time = std::chrono::steady_clock::now();

    double start_time_sec = time_stamp.toSec();
    double end_time_sec = start_time_sec + 0.1;  // 0.1 seconds later

    // Initialize the final rotation as an identity quaternion
    Eigen::Quaterniond final_rotation(1, 0, 0, 0);

    // Accumulate rotation from IMU data
    for (const auto &imu : imu_buffer) {
        double imu_time = imu->header.stamp.toSec();
        if (imu_time >= start_time_sec && imu_time <= end_time_sec) {
            Eigen::Quaterniond imu_rotation(1.0, imu->orientation.x,
                                            imu->orientation.y, imu->orientation.z);
            final_rotation = imu_rotation * final_rotation;  // Order matters in quaternion multiplication
        }
    }

    // Convert final rotation to SO3 for easier manipulation
    Sophus::SO3d final_so3(final_rotation);

    // Process each point in the cloud
    cloudOut->points.resize(cloudIn->points.size());
    cloudOut->width = cloudIn->width;
    cloudOut->height = cloudIn->height;
    cloudOut->is_dense = cloudIn->is_dense;

    for (size_t i = 0; i < cloudIn->points.size(); ++i) {
        const auto &pt = cloudIn->points[i];
        double point_time = pt.timestamp;  // Assume pt.timestamp is in nanoseconds

        // Calculate the ratio of time elapsed for each point
        double duration = end_time_sec - start_time_sec;
        double time_ratio = (point_time - start_time_sec) / duration;

        // Calculate the rotation for the current point
        Sophus::SO3d point_rotation = Sophus::SO3d::exp(time_ratio * final_so3.log());

        // Apply the rotation
        Eigen::Vector3d pt_vec(pt.x, pt.y, pt.z);
        Eigen::Vector3d rotated_pt = point_rotation * pt_vec;

        PointXYZIT new_point;
        new_point.x = rotated_pt.x();
        new_point.y = rotated_pt.y();
        new_point.z = rotated_pt.z();
        cloudOut->points[i] = new_point;
    }

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    time_taken = elapsed_seconds.count();
}
