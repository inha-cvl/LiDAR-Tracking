#include "utils.hpp"

class Tracking {
public:
    Tracking() {};

    Tracking(ros::NodeHandle& nh) : nh_(nh) {
        // ROS Parameters
        nh_.getParam("Tracking/integration/threshIOU", threshIOU);
        nh_.getParam("Tracking/crop_hd_map/radius", crop_hd_map_radius);
    }

    void integrationBbox(const jsk_recognition_msgs::BoundingBoxArray &cluster_bbox_array, 
                         const jsk_recognition_msgs::BoundingBoxArray &deep_bbox_array,
                         jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double &time_taken);
    
    void cropBboxHDMap(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, 
                       jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, 
                       const ros::Time &input_stamp, tf2_ros::Buffer &tf_buffer, const std::string target_frame, 
                       const std::string world_frame, const std::vector<std::pair<float, float>> &global_path, double &time_taken);
    
    void tracking(Track &tracker, const jsk_recognition_msgs::BoundingBoxArray &bbox_array, 
                  jsk_recognition_msgs::BoundingBoxArray &track_bbox_array, visualization_msgs::MarkerArray &track_text_array,
                  const ros::Time &stamp, double &time_taken);

    void transformBbox(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, const std::string &frame_id, 
                       const std::string &target_frame, tf2_ros::Buffer &tf_buffer,
                       jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double &time_taken);

    void correctionBbox(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, const ros::Time &input_stamp, 
                        const std::string &target_frame, const std::string &world_frame, tf2_ros::Buffer &tf_buffer,
                        jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double &time_taken);

private:
    ros::NodeHandle nh_;
    float threshIOU;    // IOU threshold for bounding box integration
    double crop_hd_map_radius; // Minimum distance threshold for HD map cropping
};

void Tracking::integrationBbox(const jsk_recognition_msgs::BoundingBoxArray &cluster_bbox_array, 
                               const jsk_recognition_msgs::BoundingBoxArray &deep_bbox_array,
                               jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double &time_taken) 
{
    auto start = std::chrono::steady_clock::now();

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
}

void Tracking::cropBboxHDMap(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, 
                             jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, 
                             const ros::Time &input_stamp, tf2_ros::Buffer &tf_buffer, const std::string target_frame, 
                             const std::string world_frame, const std::vector<std::pair<float, float>> &global_path, double &time_taken) 
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
}

void Tracking::tracking(Track &tracker, const jsk_recognition_msgs::BoundingBoxArray &bbox_array, 
                        jsk_recognition_msgs::BoundingBoxArray &track_bbox_array, visualization_msgs::MarkerArray &track_text_array,
                        const ros::Time &stamp, double &time_taken)
{
    auto start = std::chrono::steady_clock::now();

    track_bbox_array.boxes.clear();
    track_text_array.markers.clear();
    tracker.predictNewLocationOfTracks();
    tracker.assignDetectionsTracks(bbox_array);
    tracker.assignedTracksUpdate(bbox_array, stamp);
    tracker.unassignedTracksUpdate();
    tracker.deleteLostTracks();
    tracker.createNewTracks(bbox_array);
    auto bbox = tracker.displayTrack();
    track_bbox_array = bbox.first;
    track_text_array = bbox.second;

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    time_taken = elapsed_seconds.count();
}

void Tracking::transformBbox(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, const std::string &frame_id, 
                             const std::string &target_frame, tf2_ros::Buffer &tf_buffer,
                             jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double &time_taken) 
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
}

void Tracking::correctionBbox(const jsk_recognition_msgs::BoundingBoxArray &input_bbox_array, const ros::Time &input_stamp, 
                              const std::string &target_frame, const std::string &world_frame, tf2_ros::Buffer &tf_buffer,
                              jsk_recognition_msgs::BoundingBoxArray &output_bbox_array, double &time_taken) 
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
}
