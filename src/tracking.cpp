#include <iostream>
#include "cloud_processor/cloud_processor.hpp"

ros::Publisher pub_track_box, pub_track_test;
ros::Publisher pub_track_text;
ros::Publisher pub_track_model;

Track tracker;

double t8, t9, t10, t11, total;
std::string fixed_frame;

// correction
tf2_ros::Buffer tf_buffer;

jsk_recognition_msgs::BoundingBoxArray cluster_bbox_array, deep_bbox_array, integration_bbox_array, track_bbox_array, transformed_bbox_array, corrected_bbox_array, output_bbox_array;
visualization_msgs::MarkerArray track_text_array, track_model_array;

// clustering이 더 오래 걸려서 callbackCluster에서 integration
void callbackCluster(const jsk_recognition_msgs::BoundingBoxArray::Ptr &bba_msg)
{   
    if (bba_msg->boxes.empty()) { return; }

    cluster_bbox_array = *bba_msg;
    integrationBbox(cluster_bbox_array, deep_bbox_array, integration_bbox_array, t8);
    tracking(tracker, integration_bbox_array, track_bbox_array, track_text_array, bba_msg->header.stamp, t9);
    
    if (checkTransform(tf_buffer, frameID, target_frame) == true) {
        transformBbox(track_bbox_array, frameID, target_frame, tf_buffer, transformed_bbox_array, t10);
        correctionBbox(transformed_bbox_array, bba_msg->header.stamp, target_frame, world_frame, tf_buffer, corrected_bbox_array, t11);
        fixed_frame = target_frame;
        output_bbox_array = corrected_bbox_array;
    }
    else {
        fixed_frame = frameID;
        output_bbox_array = track_bbox_array;
    }
    
    pub_track_box.publish(bba2msg(output_bbox_array, ros::Time::now(), fixed_frame));
    pub_track_model.publish(bba2ma(output_bbox_array, ros::Time::now(), fixed_frame));
    pub_track_text.publish(ta2msg(track_text_array, ros::Time::now(), fixed_frame));
    
    total = ros::Time::now().toSec() - cluster_bbox_array.boxes[0].header.stamp.toSec();

    // std::cout << "\033[" << 17 << ";" << 30 << "H" << std::endl;
    // std::cout << "integration : " << t8 << "sec" << std::endl;
    // std::cout << "tracking : " << t9 << "sec" << std::endl;
    // std::cout << "transform & correction : " << t10+t11 << "sec" << std::endl;
    // std::cout << "total : " << total << " sec" << std::endl;
    // std::cout << "fixed frame : " << fixed_frame << std::endl;
}

void callbackDeep(const jsk_recognition_msgs::BoundingBoxArray::Ptr &bba_msg)
{
    deep_bbox_array = *bba_msg;
    // tracking(tracker, deep_bbox_array, track_bbox_array, track_text_array, bba_msg->header.stamp, t9);
    // pub_track_box.publish(bba2msg(track_bbox_array, ros::Time::now(), frameID));
    // pub_track_model.publish(bba2ma(track_bbox_array, ros::Time::now(), frameID));
    // pub_track_text.publish(ta2msg(track_text_array, ros::Time::now(), frameID));
}

int main(int argc, char**argv)
{
    ros::init(argc, argv, "tracking");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    tf2_ros::TransformListener tf_listener(tf_buffer);

    pub_track_box = pnh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/mobinha/perception/lidar/track_box", 1);
    pub_track_text = pnh.advertise<visualization_msgs::MarkerArray>("/mobinha/visualize/visualize/track_text", 1);
    pub_track_model = pnh.advertise<visualization_msgs::MarkerArray>("/mobinha/visualize/visualize/track_model", 1);

    ros::Subscriber sub_cluster_box = nh.subscribe("/cloud_segmentation/cluster_box", 1, callbackCluster);
    ros::Subscriber sub_deep_box = nh.subscribe("/deep_box", 1, callbackDeep);

    ros::spin();
    return 0;
}
