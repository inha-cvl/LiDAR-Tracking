#include <iostream>
#include <csignal>
#include "tracking/tracking.hpp"

ros::Publisher pub_track_box;
ros::Publisher pub_track_text;
ros::Publisher pub_track_model;

// Track tracker;
boost::shared_ptr<Tracking> Tracking_;  // Tracking 클래스의 객체를 boost::shared_ptr로 관리

double t9, t10, t11, t12, t13, total;
std::string fixed_frame;

tf2_ros::Buffer tf_buffer;

jsk_recognition_msgs::BoundingBoxArray cluster_bbox_array, deep_bbox_array, integration_bbox_array, filtered_bbox_array, track_bbox_array, transformed_bbox_array, corrected_bbox_array, output_bbox_array;
visualization_msgs::MarkerArray track_text_array, track_model_array;

std::string lidar_frame, target_frame, world_frame;

void callbackCluster(const jsk_recognition_msgs::BoundingBoxArray::Ptr &bba_msg)
{   
    if (bba_msg->boxes.empty()) { return; }

    cluster_bbox_array = *bba_msg;

    Tracking_->integrationBbox(cluster_bbox_array, deep_bbox_array, integration_bbox_array, t9);
    // Tracking_->correctionBboxRelativeSpeed(track_bbox_array, bba_msg->header.stamp, ros::Time::now(), corrected_bbox_array, t12);
    
    if (checkTransform(tf_buffer, world_frame, target_frame)) {
        Tracking_->transformBbox(integration_bbox_array, tf_buffer, transformed_bbox_array, t10);
        Tracking_->cropHDMapBbox(transformed_bbox_array, filtered_bbox_array, bba_msg->header.stamp, t11);
        Tracking_->tracking(filtered_bbox_array, track_bbox_array, track_text_array, bba_msg->header.stamp, t12);
        Tracking_->correctionBboxTF(track_bbox_array, bba_msg->header.stamp, ros::Time::now(), tf_buffer, corrected_bbox_array, t13);
        fixed_frame = target_frame;
        output_bbox_array = corrected_bbox_array;
    } else {
        Tracking_->tracking(integration_bbox_array, track_bbox_array, track_text_array, bba_msg->header.stamp, t12);
        fixed_frame = lidar_frame;
        output_bbox_array = track_bbox_array;
    }

    pub_track_box.publish(bba2msg(output_bbox_array, ros::Time::now(), fixed_frame));
    pub_track_model.publish(bba2ma(output_bbox_array, ros::Time::now(), fixed_frame));
    pub_track_text.publish(ta2msg(track_text_array, ros::Time::now(), fixed_frame));

    total = ros::Time::now().toSec() - cluster_bbox_array.boxes[0].header.stamp.toSec();

    std::cout << "\033[" << 18 << ";" << 30 << "H" << std::endl;
    std::cout << "integration & crophdmap : " << t9+t10+t11 << "sec" << std::endl;
    std::cout << "tracking : " << t12 << "sec" << std::endl;
    std::cout << "correction : " << t13 << "sec" << std::endl;
    // std::cout << "transform : " << t13 << "sec" << std::endl;
    std::cout << "total : " << total << " sec" << std::endl;
    std::cout << "fixed frame : " << fixed_frame << std::endl;
    
}

void callbackDeep(const jsk_recognition_msgs::BoundingBoxArray::Ptr &bba_msg)
{
    if (bba_msg->boxes.empty()) { return; }

    deep_bbox_array = *bba_msg;
}

void callbackWaypoints(const sensor_msgs::PointCloud2::Ptr &cloud_msg)
{
    Tracking_->updateWaypoints(cloud_msg);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "tracking");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    tf2_ros::TransformListener tf_listener(tf_buffer);

    pnh.param<std::string>("lidar_frame", lidar_frame, "hesai_lidar");
    pnh.param<std::string>("target_frame", target_frame, "ego_car");
    pnh.param<std::string>("world_frame", world_frame, "world");

    pub_track_box = pnh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/mobinha/perception/lidar/track_box", 10);
    pub_track_text = pnh.advertise<visualization_msgs::MarkerArray>("/mobinha/visualize/visualize/track_text", 10);
    pub_track_model = pnh.advertise<visualization_msgs::MarkerArray>("/mobinha/visualize/visualize/track_model", 10);

    // Tracking 객체를 초기화
    Tracking_ = boost::make_shared<Tracking>(pnh);

    ros::Subscriber sub_cluster_box = nh.subscribe("/cloud_segmentation/cluster_box", 1, callbackCluster);
    ros::Subscriber sub_deep_box = nh.subscribe("/deep_box", 1, callbackDeep);
    ros::Subscriber sub_waypoints = nh.subscribe("/waypoints", 1, callbackWaypoints);

    ros::spin();
    return 0;
}

