#include "params.h"
#include "utils.hpp"
// #include "undistortion/data_process/data_process.h"
#include "lshape/lshaped_fitting.h"
#include "track/track.h"


// Pandar 64
/*
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
*/

// OT 128
void projectPointCloud(const pcl::PointCloud<PointXYZIT>::Ptr &cloudIn, 
                        pcl::PointCloud<PointXYZIT>::Ptr &cloudOut, double &time_taken)
{
    auto start = std::chrono::steady_clock::now();

    if (cloudIn->points.empty()) {
        std::cerr << "Input cloud is empty! <- projectPointCloud" << std::endl;
        return;
    }

    cloudOut->clear();
    cloudOut->points.resize(V_SCAN * H_SCAN);

    float verticalAngle, horizonAngle;
    size_t rowIdn, columnIdn, index;
    
    for (const auto& inPoint : cloudIn->points)
    {   
        verticalAngle = atan2(inPoint.z, sqrt(inPoint.x * inPoint.x + inPoint.y * inPoint.y)) * 180 / M_PI;

        if (verticalAngle >= -25 && verticalAngle < -14) {
            rowIdn = (verticalAngle + 25) / 1.2;
        } else if (verticalAngle < -6) {
            rowIdn = 4 + (verticalAngle + 14) / 0.67;
        } else if (verticalAngle < -0.25) {
            rowIdn = 8 + (verticalAngle + 6) / 0.36;
        } else if (verticalAngle < 15) {
            rowIdn = 24 + (verticalAngle + 0.25) / 0.125;
        } else {
            continue;
        }

        if (rowIdn < 0 || rowIdn >= V_SCAN)
            continue;

        horizonAngle = atan2(inPoint.x, inPoint.y) * 180 / M_PI;
        columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + H_SCAN / 2;

        if (columnIdn >= H_SCAN)
            columnIdn -= H_SCAN;

        if (columnIdn < 0 || columnIdn >= H_SCAN)
            continue;

        index = columnIdn + rowIdn * H_SCAN;

        cloudOut->points[index] = inPoint;
    }
    
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
}

void convertPointCloudToImage(const pcl::PointCloud<PointXYZIT>::Ptr &cloudIn, cv::Mat &imageOut, double &time_taken)
{   
    auto start = std::chrono::steady_clock::now();

    if (cloudIn->points.empty()) {
        std::cerr << "Input cloud is empty! <- projectPointCloud" << std::endl;
        return;
    }
    cv::Mat temp_image = cv::Mat::zeros(V_SCAN, H_SCAN, CV_8UC1);

    for (int i = 0; i < V_SCAN; ++i)
    {
        int reversed_i = V_SCAN - 1 - i;  // 수직 인덱스를 반대로 계산하여 상하 반전
        for (int j = 0; j < H_SCAN; ++j)
        {
            int reversed_j = H_SCAN - 1 - j;  // 수평 인덱스를 반대로 계산하여 좌우 반전
            int index = i * H_SCAN + j;
            if (!std::isnan(cloudIn->points[index].intensity))
            {
                temp_image.at<uchar>(reversed_i, reversed_j) = static_cast<uchar>(cloudIn->points[index].intensity);
            }
        }
    }

    // Apply histogram equalization to enhance the image contrast
    cv::equalizeHist(temp_image, imageOut);

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

    if (cloudIn->points.empty()) {
        std::cerr << "Input cloud is empty! <- cropPointCloud" << std::endl;
        return;
    }
    
    cloudOut->clear();
    cloudOut->reserve(cloudIn->size());

    for (const auto& inPoint : cloudIn->points)
    {
        const PointType& point = inPoint;

        // ring filtering
        if (crop_ring == true && point.ring % ring == 0) { continue; }
        // intensity filtering
        if (crop_intensity == true && point.intensity < intensity) { continue; }
        
        // Car exclusion
        if (point.x >= min_x && point.x <= max_x &&
            point.y >= min_y && point.y <= max_y) { continue; }

        // Rectangle
        if (point.x >= MIN_X && point.x <= MAX_X &&
            point.y >= MIN_Y && point.y <= MAX_Y &&
            //point.z <= MAX_Z)
            point.z >= MIN_Z && point.z <= MAX_Z)
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

/*
template<typename PointT>
void cropPointCloudHDMap(const typename pcl::PointCloud<PointT>::Ptr &cloudIn, typename pcl::PointCloud<PointT>::Ptr &cloudOut, 
                    const ros::Time &input_stamp, tf2_ros::Buffer &tf_buffer, const std::string target_frame, 
                    const std::string world_frame, const std::vector<std::pair<float, float>> &global_path, double &time_taken)
{
    auto start = std::chrono::steady_clock::now();

    if (cloudIn->points.empty()) {
        std::cerr << "Input cloud is empty! <- cropPointCloudHDMap" << std::endl;
        return;
    }

    cloudOut->clear();
    cloudOut->reserve(cloudIn->size());

    pcl::KdTreeFLANN<PointT> kdtree;

    kdtree.setInputCloud(cloudIn);

    geometry_msgs::TransformStamped transformStamped;
    try {
        transformStamped = tf_buffer.lookupTransform(world_frame, target_frame, ros::Time(0)); // input_stamp
    } catch (tf2::TransformException &ex) {
        // ROS_WARN("%s", ex.what());
        cloudOut = cloudIn;
        return;  // Handle the exception or exit
    }

    double ego_x = transformStamped.transform.translation.x;
    double ego_y = transformStamped.transform.translation.y;
    tf2::Quaternion q(transformStamped.transform.rotation.x, 
                        transformStamped.transform.rotation.y,
                        transformStamped.transform.rotation.z, 
                        transformStamped.transform.rotation.w);

    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    double min_dis = 1000;
    int min_idx = 0;
    // int step_size = 4000;

    int path_len = global_path.size();
    
    for (int i = 0; i < path_len; ++i) {
        double dis = std::hypot(global_path[i].first - ego_x, global_path[i].second - ego_y);
        if (min_dis > dis) { min_dis = dis; min_idx = i; }
    }


    int curr_index = min_idx;
    
    double relative_node_x, relative_node_y;
    
    std::vector<int> indices;
    for (int k = 0; k < 150; ++k) {

        indices.push_back((curr_index + k) % path_len);
    }
    
    for (int idx : indices){
        if (idx%2==0){

            double dx = global_path[idx].first - ego_x;
            double dy = global_path[idx].second - ego_y;

            double cos_heading = std::cos(yaw);
            double sin_heading = std::sin(yaw);

            relative_node_x = cos_heading * dx + sin_heading * dy;
            relative_node_y = -sin_heading * dx + cos_heading * dy;


            PointT query;
            query.x = static_cast<float>(relative_node_x);
            query.y = static_cast<float>(relative_node_y);
            query.z = 0.0f;
            // query.intensity = 0.0f;

            std::vector<int> idxes;
            std::vector<float> sqr_dists;

            idxes.clear();
            sqr_dists.clear();
            
            kdtree.radiusSearch(query, radius, idxes, sqr_dists);
            
            
            for (const auto& idx : idxes) {
                cloudOut->points.push_back(cloudIn->points[idx]);
            }
        }
    }
    

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
}
*/

void undistortPointCloud(const pcl::PointCloud<PointXYZIT>::Ptr &cloudIn, const Eigen::Quaterniond &rotation,
                        pcl::PointCloud<PointXYZIT>::Ptr &cloudOut, double &time_taken)
{
    auto start_time = std::chrono::steady_clock::now();

    if (cloudIn->points.empty()) {
        std::cerr << "Input cloud is empty! <- undistortPointCloud" << std::endl;
        return;
    }

    // Convert the quaternion to SO3 for easier manipulation and calculate its inverse
    Sophus::SO3d final_so3(rotation);
    Sophus::SO3d inverse_so3 = final_so3.inverse();

    // Process each point in the cloud
    cloudOut->clear();
    cloudOut->points.resize(cloudIn->points.size());
    // cloudOut->width = cloudIn->width;
    // cloudOut->height = cloudIn->height;
    // cloudOut->is_dense = cloudIn->is_dense;

    for (size_t i = 0; i < cloudIn->points.size(); ++i) {
        const auto &pt = cloudIn->points[i];
        
        // Apply the inverse rotation
        Eigen::Vector3d pt_vec(pt.x, pt.y, pt.z);
        Eigen::Vector3d rotated_pt = inverse_so3 * pt_vec;

        PointXYZIT new_point;
        new_point = pt;
        new_point.x = rotated_pt.x();
        new_point.y = rotated_pt.y();
        new_point.z = rotated_pt.z();
        cloudOut->points[i] = new_point;
    }

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    time_taken = elapsed_seconds.count();
}

void downsamplingPointCloud(const pcl::PointCloud<PointType>::Ptr &cloudIn, 
                            pcl::PointCloud<pcl::PointXYZ>::Ptr &cloudOut, double &time_taken)
{
    auto start = std::chrono::steady_clock::now();

    if (cloudIn->points.empty()) {
        std::cerr << "Input cloud is empty! <- downsamplingPointCloud" << std::endl;
        return;
    }

    cloudOut->clear();
    // cloudOut->reserve(cloudIn->size());

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

void adaptiveClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloudIn, 
                        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &outputClusters, double &time_taken)
{
    auto start = std::chrono::high_resolution_clock::now();

    if (cloudIn->points.empty()) {
        std::cerr << "Input cloud is empty! <- adaptiveClustering" << std::endl;
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

void fittingLShape(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &inputClusters, const ros::Time &input_stamp, 
                jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double &time_taken)
{
    auto start = std::chrono::steady_clock::now();

    if (inputClusters.empty()) {
        std::cerr << "Input clusters is empty! <- fittingLShape" << std::endl;
        return;
    }

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

/*
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
*/
