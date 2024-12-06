#include "utils.hpp"

class Tracking {
public:
    Tracking() {};

    Tracking(ros::NodeHandle& nh) : nh_(nh) {
        // ROS Parameters
        nh_.getParam("Public/lidar_frame", lidar_frame);
        nh_.getParam("Public/target_frame", target_frame);
        nh_.getParam("Public/world_frame", world_frame);
        nh_.getParam("Tracking/integration/mode", mode);
        nh_.getParam("Tracking/integration/thresh_iou", thresh_iou);
        nh_.getParam("Tracking/crop_hd_map/number_front_node", number_front_node);
        nh_.getParam("Tracking/crop_hd_map/number_back_node", number_back_node);
        nh_.getParam("Tracking/crop_hd_map/radius", radius);

        nh_.getParam("Tracking/track/invisibleCnt", invisibleCnt);
        nh_.getParam("Tracking/track/deque/number_velocity", number_velocity_deque);
        nh_.getParam("Tracking/track/deque/number_orientation", number_orientation_deque);
        nh_.getParam("Tracking/track/deque/thresh_velocity", thresh_velocity);
        nh_.getParam("Tracking/track/deque/thresh_orientation", thresh_orientation);

        // mission
        nh_.getParam("Mission/tracking/cluster_distance", cluster_distance);
        nh_.getParam("Mission/tracking/ground_removal_distance", ground_removal_distance);
        nh_.getParam("Mission/tracking/cluster_size", cluster_size);
        nh_.getParam("Mission/tracking/cluster_ratio", cluster_ratio);
        nh_.getParam("Mission/tracking/deep_distance", deep_distance);
        nh_.getParam("Mission/tracking/deep_score", deep_score);
        
        global_path = map_reader(map.c_str());
        
        // integration
        last_timestamp_cluster = -1;
        last_timestamp_deep = -1;
        
        tracker.setParams(invisibleCnt, number_velocity_deque, number_orientation_deque, thresh_velocity, thresh_orientation);

        waypoints_cache.setCacheSize(600);

        clearLogFile(integration_time_log_path);
        clearLogFile(crophdmap_time_log_path);
        clearLogFile(tracking_time_log_path);
        clearLogFile(transform_time_log_path);
        clearLogFile(correction_time_log_path);
    }

    void updateWaypoints(const sensor_msgs::PointCloud2::ConstPtr &cloud_msg);
    void integrationBbox(jsk_recognition_msgs::BoundingBoxArray &cluster_bbox_array, 
                         jsk_recognition_msgs::BoundingBoxArray &deep_bbox_array,
                         jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double &time_taken);
    
    void cropHDMapBbox(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, 
                        jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, 
                        const ros::Time &input_stamp, double& time_taken);
    
    void tracking(const jsk_recognition_msgs::BoundingBoxArray &bbox_array, jsk_recognition_msgs::BoundingBoxArray &track_bbox_array, 
                    visualization_msgs::MarkerArray &track_text_array, const ros::Time &input_stamp, double &time_taken);

    void transformBbox(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, tf2_ros::Buffer &tf_buffer, 
                        jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double &time_taken);

    void correctionBboxRelativeSpeed(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, const ros::Time &input_stamp, 
                        const ros::Time &cur_stamp, jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double& time_taken);

    void correctionBboxTF(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, const ros::Time &input_stamp, 
                        const ros::Time &cur_stamp, tf2_ros::Buffer &tf_buffer, 
                        jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double &time_taken);
    
    void averageTime();

private:
    ros::NodeHandle nh_;
    std::string map;
    std::string lidar_frame;
    std::string target_frame;
    std::string world_frame;
    std::vector<std::pair<float, float>> global_path; // Initialization is Public
    int mode;
    float thresh_iou;    // IOU threshold for bounding box integration
    int number_front_node;
    int number_back_node;
    double radius; // Minimum distance threshold for HD map cropping
    
    int invisibleCnt;
    int number_velocity_deque;
    int number_orientation_deque;
    float thresh_velocity;
    float thresh_orientation;

    // mission
    int cluster_distance;
    int ground_removal_distance;
    float cluster_size;
    float cluster_ratio;
    int deep_distance;
    float deep_score;
    
    double last_timestamp_cluster;
    double last_timestamp_deep;

    Track tracker;

    message_filters::Cache<sensor_msgs::PointCloud2> waypoints_cache;
    
    // average time check
    std::string package_path = ros::package::getPath("lidar_tracking") + "/time_log/tracking/";
    std::string integration_time_log_path = package_path + "integration.txt";
    std::string crophdmap_time_log_path = package_path + "crophdmap.txt";
    std::string tracking_time_log_path = package_path + "tracking.txt";
    std::string transform_time_log_path = package_path + "transform.txt";
    std::string correction_time_log_path = package_path + "correction.txt";
};

void Tracking::updateWaypoints(const sensor_msgs::PointCloud2::ConstPtr &cloud_msg) 
{
    waypoints_cache.add(cloud_msg);
}


void Tracking::integrationBbox(jsk_recognition_msgs::BoundingBoxArray &cluster_bbox_array, 
                               jsk_recognition_msgs::BoundingBoxArray &deep_bbox_array,
                               jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double& time_taken) 
{
    auto start = std::chrono::steady_clock::now();

    if (cluster_bbox_array.header.stamp.toSec() == last_timestamp_cluster) {
        cluster_bbox_array.boxes.clear();
    }
    if (deep_bbox_array.header.stamp.toSec() == last_timestamp_deep) {
        deep_bbox_array.boxes.clear();
    }

    output_bbox_array.boxes.clear();
    
    if (cluster_bbox_array.boxes.empty()) { cluster_bbox_array.boxes.clear(); }
    if (deep_bbox_array.boxes.empty()) { deep_bbox_array.boxes.clear(); }
    
    // mode
    if (mode == 0) {
        for (const auto &cluster_bbox : cluster_bbox_array.boxes) {
            
            bool keep_cluster_bbox = true;
            for (const auto &deep_bbox : deep_bbox_array.boxes) {
                double overlap = getBBoxOverlap(cluster_bbox, deep_bbox);
                if (overlap > thresh_iou) {
                    keep_cluster_bbox = false;
                    break;
                }
            }
            if (keep_cluster_bbox) {
                output_bbox_array.boxes.push_back(cluster_bbox);
            }
        }
        output_bbox_array.boxes.insert(output_bbox_array.boxes.end(), deep_bbox_array.boxes.begin(), deep_bbox_array.boxes.end());
    } 
    else if (mode == 1) { output_bbox_array = cluster_bbox_array; } 
    else if (mode == 2) { output_bbox_array = deep_bbox_array; }

    last_timestamp_cluster = cluster_bbox_array.header.stamp.toSec();
    last_timestamp_deep = deep_bbox_array.header.stamp.toSec();

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
    saveTimeToFile(integration_time_log_path, time_taken);
}

void Tracking::cropHDMapBbox(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, 
                             jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, 
                             const ros::Time &input_stamp, double& time_taken) 
{
    auto start = std::chrono::steady_clock::now();

    output_bbox_array.boxes.clear();

    auto closest_waypoint = waypoints_cache.getElemAfterTime(input_stamp);
    if (!closest_waypoint) {
        closest_waypoint = waypoints_cache.getElemBeforeTime(input_stamp);
    }

    if (!closest_waypoint) {
        ROS_WARN("No waypoints found in cache for the specified timestamp.");
        output_bbox_array = input_bbox_array;
        return;
    }

    pcl::PointCloud<pcl::PointXY>::Ptr cloud(new pcl::PointCloud<pcl::PointXY>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*closest_waypoint, *temp_cloud);

    for (const auto& point : temp_cloud->points) {
        pcl::PointXY xy_point;
        xy_point.x = point.x;
        xy_point.y = point.y;
        cloud->points.push_back(xy_point);
    }

    pcl::KdTreeFLANN<pcl::PointXY> kdtree;
    kdtree.setInputCloud(cloud);

    for (const auto& box : input_bbox_array.boxes) {
        pcl::PointXY search_point;
        search_point.x = box.pose.position.x;
        search_point.y = box.pose.position.y;
        
        std::vector<int> point_indices;
        std::vector<float> point_distances;
        if (kdtree.radiusSearch(search_point, radius, point_indices, point_distances) > 0) {
            output_bbox_array.boxes.push_back(box);
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
    saveTimeToFile(crophdmap_time_log_path, time_taken);
}

void Tracking::tracking(const jsk_recognition_msgs::BoundingBoxArray &bbox_array, 
                        jsk_recognition_msgs::BoundingBoxArray &track_bbox_array, visualization_msgs::MarkerArray &track_text_array,
                        const ros::Time &input_stamp, double& time_taken)
{
    auto start = std::chrono::steady_clock::now();

    track_bbox_array.boxes.clear();
    track_text_array.markers.clear();
    tracker.predictNewLocationOfTracks(input_stamp);
    tracker.assignDetectionsTracks(bbox_array);
    tracker.assignedTracksUpdate(bbox_array);    
    tracker.unassignedTracksUpdate();
    tracker.deleteLostTracks();
    tracker.createNewTracks(bbox_array);
    auto bbox = tracker.displayTrack();
    track_bbox_array = bbox.first;
    track_text_array = bbox.second;

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
    saveTimeToFile(tracking_time_log_path, time_taken);
}

void Tracking::transformBbox(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, tf2_ros::Buffer &tf_buffer, 
                            jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double& time_taken) 
{
    auto start = std::chrono::high_resolution_clock::now();

    output_bbox_array.boxes.clear();

    geometry_msgs::TransformStamped transformStamped;
    try {
        transformStamped = tf_buffer.lookupTransform(target_frame, lidar_frame, ros::Time(0)); // static tf
    } catch (tf2::TransformException &ex) {
        output_bbox_array = input_bbox_array;
        return;
    }

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
    saveTimeToFile(transform_time_log_path, time_taken);
}

void Tracking::correctionBboxRelativeSpeed(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, const ros::Time &input_stamp, 
                            const ros::Time &cur_stamp, jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double& time_taken) 
{
    auto start = std::chrono::steady_clock::now();

    output_bbox_array.boxes.clear();

    double delta_time = (cur_stamp - input_stamp).toSec();

    for (const auto &box : input_bbox_array.boxes) {
        jsk_recognition_msgs::BoundingBox corrected_box = box; // 원래 box 복사
        corrected_box.header.stamp = cur_stamp;

        if (corrected_box.header.seq > invisibleCnt / 2 && corrected_box.label == 1) {
            
            double velocity = std::abs(box.value);
            double yaw = tf::getYaw(box.pose.orientation);
            double delta_x = velocity * 0.2 * cos(yaw);
            double delta_y = velocity * 0.2 * sin(yaw);
            // 100km/h & 0.1sec -> 2.76m 
            delta_x = std::copysign(std::min(std::abs(delta_x), 2.8), delta_x);
            delta_y = std::copysign(std::min(std::abs(delta_y), 2.8), delta_y);

            corrected_box.pose.position.x += delta_x; // x 방향으로 이동
            corrected_box.pose.position.y += delta_y; // y 방향으로 이동                    
        }
        
        output_bbox_array.boxes.push_back(corrected_box);
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
    saveTimeToFile(correction_time_log_path, time_taken);
}

void Tracking::correctionBboxTF(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, const ros::Time &input_stamp, 
                                const ros::Time &cur_stamp, tf2_ros::Buffer &tf_buffer, 
                                jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double& time_taken) 
{
    auto start = std::chrono::steady_clock::now();

    output_bbox_array.boxes.clear();

    geometry_msgs::TransformStamped transformStampedAtInput, transformStampedAtCur;

    try {
        // 로봇 좌표계에서 월드 좌표계로의 변환을 가져옵니다.
        transformStampedAtInput = tf_buffer.lookupTransform(world_frame, target_frame, input_stamp);
        transformStampedAtCur = tf_buffer.lookupTransform(world_frame, target_frame, cur_stamp);
    } catch (tf2::TransformException &ex) {
        output_bbox_array = input_bbox_array;
        return;
    }

    tf2::Transform tfAtInput, tfAtCur, deltaTransform;
    tf2::fromMsg(transformStampedAtInput.transform, tfAtInput);
    tf2::fromMsg(transformStampedAtCur.transform, tfAtCur);

    // 두 시점 간의 로봇의 움직임을 계산합니다.
    deltaTransform = tfAtCur.inverse() * tfAtInput;

    // deltaTransform을 geometry_msgs::TransformStamped로 변환합니다.
    geometry_msgs::TransformStamped deltaTransformStamped;
    deltaTransformStamped.header.stamp = input_stamp;
    deltaTransformStamped.header.frame_id = target_frame;
    deltaTransformStamped.child_frame_id = target_frame;
    deltaTransformStamped.transform = tf2::toMsg(deltaTransform);

    for (const auto &box : input_bbox_array.boxes) {
        geometry_msgs::PoseStamped input_pose, transformed_pose;

        input_pose.header = box.header;
        input_pose.pose = box.pose;

        // deltaTransform을 바운딩 박스에 적용합니다.
        tf2::doTransform(input_pose, transformed_pose, deltaTransformStamped);

        jsk_recognition_msgs::BoundingBox transformed_box;
        transformed_box.header = box.header;
        transformed_box.header.stamp = cur_stamp; // 보정된 시점으로 업데이트
        transformed_box.pose = transformed_pose.pose;
        transformed_box.dimensions = box.dimensions;
        transformed_box.value = box.value;
        transformed_box.label = box.label;
        output_bbox_array.boxes.push_back(transformed_box);
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
    saveTimeToFile(correction_time_log_path, time_taken);
}



/*
void Tracking::correctionBboxTF(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, const ros::Time &input_stamp, 
                              const ros::Time &cur_stamp, tf2_ros::Buffer &tf_buffer, 
                              jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double& time_taken) 
{
    auto start = std::chrono::steady_clock::now();

    output_bbox_array.boxes.clear();

    geometry_msgs::TransformStamped transformStampedAtInput, transformStampedAtStamp;

    try {
        transformStampedAtStamp = tf_buffer.lookupTransform(world_frame, target_frame, input_stamp);
        transformStampedAtInput = tf_buffer.lookupTransform(world_frame, target_frame, cur_stamp); // enu -> tf 변환 시 약 0.1초 소요 ros::Time::now() 사용 불가
    } catch (tf2::TransformException &ex) {
        output_bbox_array = input_bbox_array;
        return;
    }

    tf2::Transform tfAtInput, tfAtStamp, deltaTransform;
    tf2::fromMsg(transformStampedAtInput.transform, tfAtInput);
    tf2::fromMsg(transformStampedAtStamp.transform, tfAtStamp);

    deltaTransform = tfAtStamp.inverse() * tfAtInput;
    // deltaTransform = tfAtInput.inverse() * tfAtStamp;
    geometry_msgs::TransformStamped deltaTransformStamped;
    deltaTransformStamped.transform = tf2::toMsg(deltaTransform);

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
    saveTimeToFile(correction_time_log_path, time_taken);
}
*/