#include "utils.hpp"

class Tracking {
public:
    Tracking() {};

    Tracking(ros::NodeHandle& nh) : nh_(nh) {
        // ROS Parameters
        nh_.getParam("Public/map", map);
        nh_.getParam("Tracking/integration/threshIOU", threshIOU);
        nh_.getParam("Tracking/crop_hd_map/number_front_node", number_front_node);
        nh_.getParam("Tracking/crop_hd_map/number_back_node", number_back_node);
        nh_.getParam("Tracking/crop_hd_map/radius", crop_hd_map_radius);

        global_path = map_reader(map.c_str());

        // integration
        last_timestamp_cluster = -1;
        last_timestamp_deep = -1;
        
        clearLogFile(integration_time_log_path);
        clearLogFile(crophdmap_time_log_path);
        clearLogFile(tracking_time_log_path);
        clearLogFile(transform_time_log_path);
        clearLogFile(correction_time_log_path);
        clearLogFile(average_time_log_path);
    }

    void integrationBbox(jsk_recognition_msgs::BoundingBoxArray &cluster_bbox_array, 
                         jsk_recognition_msgs::BoundingBoxArray &deep_bbox_array,
                         jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double &time_taken);
    
    void cropHDMapBbox(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, 
                       jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, 
                       const ros::Time &input_stamp, tf2_ros::Buffer &tf_buffer, const std::string target_frame, 
                       const std::string world_frame, double &time_taken);
    
    void tracking(Track &tracker, const jsk_recognition_msgs::BoundingBoxArray &bbox_array, 
                  jsk_recognition_msgs::BoundingBoxArray &track_bbox_array, visualization_msgs::MarkerArray &track_text_array,
                  const ros::Time &input_stamp, double &time_taken);

    void transformBbox(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, const std::string &frame_id, 
                       const std::string &target_frame, tf2_ros::Buffer &tf_buffer,
                       jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double &time_taken);

    void correctionBbox(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, const ros::Time &input_stamp, 
                        const std::string &target_frame, const std::string &world_frame, tf2_ros::Buffer &tf_buffer,
                        jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double &time_taken);
    
    void averageTime();

private:
    ros::NodeHandle nh_;
    std::string map;
    std::vector<std::pair<float, float>> global_path; // Initialization is Public
    float threshIOU;    // IOU threshold for bounding box integration
    int number_front_node;
    int number_back_node;
    double crop_hd_map_radius; // Minimum distance threshold for HD map cropping
    double last_timestamp_cluster;
    double last_timestamp_deep;

    // average time check
    std::string package_path = ros::package::getPath("lidar_tracking") + "/time_log/tracking/";
    std::string integration_time_log_path = package_path + "integration.txt";
    std::string crophdmap_time_log_path = package_path + "crophdmap.txt";
    std::string tracking_time_log_path = package_path + "tracking.txt";
    std::string transform_time_log_path = package_path + "transform.txt";
    std::string correction_time_log_path = package_path + "correction.txt";
    std::string average_time_log_path = package_path + "average.txt";
};

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
    for (const auto &cluster_bbox : cluster_bbox_array.boxes) {
        bool keep_cluster_bbox = true;
        for (const auto &deep_bbox : deep_bbox_array.boxes) {
            double overlap = getBBoxOverlap(cluster_bbox, deep_bbox);
            if (overlap > threshIOU) {
                keep_cluster_bbox = false;
                break;
            }
        }
        if (keep_cluster_bbox) {
            output_bbox_array.boxes.push_back(cluster_bbox);
        }
    }
    output_bbox_array.boxes.insert(output_bbox_array.boxes.end(), deep_bbox_array.boxes.begin(), deep_bbox_array.boxes.end());

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
    saveTimeToFile(integration_time_log_path, time_taken);
}
/*
void Tracking::cropHDMapBbox(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, 
                             jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, 
                             const ros::Time &input_stamp, tf2_ros::Buffer &tf_buffer, const std::string target_frame, 
                             const std::string world_frame, double& time_taken) 
{
    auto start = std::chrono::steady_clock::now();

    output_bbox_array.boxes.clear();

    geometry_msgs::TransformStamped transformStamped;
    try {
        transformStamped = tf_buffer.lookupTransform(world_frame, target_frame, ros::Time(0)); // input_stamp
    } catch (tf2::TransformException &ex) {
        output_bbox_array = input_bbox_array;
        return;
    }  

    for (const auto& box : input_bbox_array.boxes) {
        geometry_msgs::Point transformed_point;
        tf2::doTransform(box.pose.position, transformed_point, transformStamped);

        bool within_range = false;
        for (const auto& path_point : global_path) {
            double distance = std::hypot(path_point.first - transformed_point.x, path_point.second - transformed_point.y);
            if (distance <= crop_hd_map_radius) {
                within_range = true;
                break;
            }
        }

        if (within_range) {
            output_bbox_array.boxes.push_back(box);
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
    saveTimeToFile(crophdmap_time_log_path, time_taken);
}
*/

void Tracking::cropHDMapBbox(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, 
                             jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, 
                             const ros::Time &input_stamp, tf2_ros::Buffer &tf_buffer, const std::string target_frame, 
                             const std::string world_frame, double& time_taken) 
{
    auto start = std::chrono::steady_clock::now();

    output_bbox_array.boxes.clear();

    geometry_msgs::TransformStamped transformStamped;
    try {
        transformStamped = tf_buffer.lookupTransform(world_frame, target_frame, ros::Time(0)); // input_stamp
    } catch (tf2::TransformException &ex) {
        output_bbox_array = input_bbox_array;
        return;
    }

    // 1. transformStamped에서 현재 로봇 위치를 가져오기
    geometry_msgs::Point current_position;
    current_position.x = transformStamped.transform.translation.x;
    current_position.y = transformStamped.transform.translation.y;

    int closest_idx = -1;
    double min_distance = std::numeric_limits<double>::max();
    for (size_t i = 0; i < global_path.size(); ++i) {
        double distance = std::hypot(global_path[i].first - current_position.x, global_path[i].second - current_position.y);
        if (distance < min_distance) {
            min_distance = distance;
            closest_idx = i;
        }
    }

    // 3. 전방 20개, 후방 20개의 노드 선택
    int start_idx = std::max(0, closest_idx - number_back_node);
    int end_idx = std::min(static_cast<int>(global_path.size()) - 1, closest_idx + number_front_node);

    for (const auto& box : input_bbox_array.boxes) {
        geometry_msgs::Point transformed_point;
        tf2::doTransform(box.pose.position, transformed_point, transformStamped);

        bool within_range = false;
        for (int i = start_idx; i <= end_idx; ++i) {
            double distance = std::hypot(global_path[i].first - transformed_point.x, global_path[i].second - transformed_point.y);
            if (distance <= crop_hd_map_radius) {
                within_range = true;
                break;
            }
        }

        if (within_range) {
            output_bbox_array.boxes.push_back(box);
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
    saveTimeToFile(crophdmap_time_log_path, time_taken);
}



void Tracking::tracking(Track &tracker, const jsk_recognition_msgs::BoundingBoxArray &bbox_array, 
                        jsk_recognition_msgs::BoundingBoxArray &track_bbox_array, visualization_msgs::MarkerArray &track_text_array,
                        const ros::Time &input_stamp, double& time_taken)
{
    auto start = std::chrono::steady_clock::now();

    track_bbox_array.boxes.clear();
    track_text_array.markers.clear();
    tracker.predictNewLocationOfTracks();
    tracker.assignDetectionsTracks(bbox_array);
    tracker.assignedTracksUpdate(bbox_array, input_stamp);
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

void Tracking::transformBbox(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, const std::string &frame_id, 
                             const std::string &target_frame, tf2_ros::Buffer &tf_buffer,
                             jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double& time_taken) 
{
    auto start = std::chrono::high_resolution_clock::now();

    output_bbox_array.boxes.clear();

    geometry_msgs::TransformStamped transformStamped;
    try {
        transformStamped = tf_buffer.lookupTransform(target_frame, frame_id, ros::Time(0)); // static tf
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

void Tracking::correctionBbox(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, const ros::Time &input_stamp, 
                              const std::string &target_frame, const std::string &world_frame, tf2_ros::Buffer &tf_buffer,
                              jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double& time_taken) 
{
    auto start = std::chrono::steady_clock::now();

    output_bbox_array.boxes.clear();

    geometry_msgs::TransformStamped transformStampedAtInput, transformStampedAtStamp;

    try {
        transformStampedAtStamp = tf_buffer.lookupTransform(world_frame, target_frame, input_stamp);
        transformStampedAtInput = tf_buffer.lookupTransform(world_frame, target_frame, ros::Time(0)); // enu -> tf 변환 시 약 0.1초 소요 ros::Time::now() 사용 불가
    } catch (tf2::TransformException &ex) {
        output_bbox_array = input_bbox_array;
        return;
    }

    tf2::Transform tfAtInput, tfAtStamp, deltaTransform;
    tf2::fromMsg(transformStampedAtInput.transform, tfAtInput);
    tf2::fromMsg(transformStampedAtStamp.transform, tfAtStamp);

    deltaTransform = tfAtStamp.inverse() * tfAtInput;
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

void Tracking::averageTime()
{
    std::ofstream file(average_time_log_path, std::ios::app);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << average_time_log_path << std::endl;
        return;
    }

    file << "integration : " << calculateAverageTime(integration_time_log_path) << "\n";
    file << "crophdmap : " << calculateAverageTime(crophdmap_time_log_path) << "\n";
    file << "tracking : " << calculateAverageTime(tracking_time_log_path) << "\n";
    file << "transform : " << calculateAverageTime(transform_time_log_path) << "\n";
    file << "correction : " << calculateAverageTime(correction_time_log_path) << "\n";

    file.close();
}