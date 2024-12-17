#include <iostream>
#include <csignal>
#include "tracking/tracking.hpp"


ros::Publisher pub_track_box, pub_track_text, pub_track_model, pub_track_test;

// Track tracker;
boost::shared_ptr<Tracking> Tracking_;

double t9, t10, t11, t12, t13, total;
std::string fixed_frame;

tf2_ros::Buffer tf_buffer;

jsk_recognition_msgs::BoundingBoxArray cluster_bbox_array, deep_bbox_array, integration_bbox_array, filtered_bbox_array, 
                                        output_bbox_array, track_bbox_array, transformed_bbox_array, corrected_bbox_array;
visualization_msgs::MarkerArray track_text_array, track_model_array;

std::string lidar_frame, target_frame, world_frame;

ros::Time publish_stamp;

void callbackSynchronized(const jsk_recognition_msgs::BoundingBoxArray::ConstPtr &cluster_bba_msg,
                          const jsk_recognition_msgs::BoundingBoxArray::ConstPtr &deep_bba_msg)
{
    cluster_bbox_array = *cluster_bba_msg;
    deep_bbox_array = *deep_bba_msg;

    Tracking_->integrationBbox(cluster_bbox_array, deep_bbox_array, integration_bbox_array, t9);

    if (checkTransform(tf_buffer, world_frame, target_frame)) {
        Tracking_->transformBbox(integration_bbox_array, tf_buffer, transformed_bbox_array, t10);
        Tracking_->cropHDMapBbox(transformed_bbox_array, filtered_bbox_array, cluster_bba_msg->header.stamp, t11);
        fixed_frame = target_frame;
        output_bbox_array = filtered_bbox_array;
    } else {
        fixed_frame = lidar_frame;
        output_bbox_array = integration_bbox_array;
    }

    Tracking_->tracking(output_bbox_array, track_bbox_array, track_text_array, cluster_bba_msg->header.stamp, t12);
    Tracking_->correctionBboxRelativeSpeed(track_bbox_array, cluster_bba_msg->header.stamp, publish_stamp, corrected_bbox_array, t13);

    publish_stamp = ros::Time::now();

    pub_track_box.publish(bba2msg(track_bbox_array, publish_stamp, fixed_frame));
    pub_track_model.publish(bba2ma(track_bbox_array, publish_stamp, fixed_frame));
    pub_track_text.publish(ta2msg(track_text_array, publish_stamp, fixed_frame));

    total = ros::Time::now().toSec() - cluster_bbox_array.header.stamp.toSec();

    std::cout << "\033[" << 18 << ";" << 30 << "H" << std::endl;
    std::cout << "integration & transform : " << t9 + t10 << " sec" << std::endl;
    std::cout << "tracking : " << t12 << " sec" << std::endl;
    std::cout << "total : " << total << " sec" << std::endl;
    std::cout << "fixed frame : " << fixed_frame << std::endl;
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

    Tracking_ = boost::make_shared<Tracking>(pnh);

    // message_filters
    message_filters::Subscriber<jsk_recognition_msgs::BoundingBoxArray> sub_cluster_box(nh, "/cloud_segmentation/cluster_box", 10);
    message_filters::Subscriber<jsk_recognition_msgs::BoundingBoxArray> sub_deep_box(nh, "/deep_box", 10);

    typedef message_filters::sync_policies::ApproximateTime<jsk_recognition_msgs::BoundingBoxArray, jsk_recognition_msgs::BoundingBoxArray> SyncPolicy;
    message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10), sub_cluster_box, sub_deep_box);
    sync.registerCallback(boost::bind(&callbackSynchronized, _1, _2));

    ros::Subscriber sub_waypoints = nh.subscribe("/waypoints", 1, &Tracking::updateWaypoints, Tracking_.get());

    ros::spin();
    return 0;
}


/*
#include <iostream>
#include <csignal>
#include "tracking/tracking.hpp"

ros::Publisher pub_track_box, pub_track_text, pub_track_model, pub_track_test;

// Track tracker;
boost::shared_ptr<Tracking> Tracking_;  // Tracking 클래스의 객체를 boost::shared_ptr로 관리

double t9, t10, t11, t12, t13, total;
std::string fixed_frame;

tf2_ros::Buffer tf_buffer;

jsk_recognition_msgs::BoundingBoxArray cluster_bbox_array, deep_bbox_array, integration_bbox_array, filtered_bbox_array, 
                                        output_bbox_array, track_bbox_array, transformed_bbox_array, corrected_bbox_array;
visualization_msgs::MarkerArray track_text_array, track_model_array;

std::string lidar_frame, target_frame, world_frame;

ros::Time publish_stamp;

void callbackCluster(const jsk_recognition_msgs::BoundingBoxArray::Ptr &bba_msg)
{   
    if (bba_msg->boxes.empty()) { return; }

    cluster_bbox_array = *bba_msg;

    Tracking_->integrationBbox(cluster_bbox_array, deep_bbox_array, integration_bbox_array, t9);
    if (checkTransform(tf_buffer, world_frame, target_frame)) {
        Tracking_->transformBbox(integration_bbox_array, tf_buffer, transformed_bbox_array, t10);
        // Tracking_->cropHDMapBbox(transformed_bbox_array, filtered_bbox_array, bba_msg->header.stamp, t11);
        // Tracking_->correctionBboxTF(track_bbox_array, bba_msg->header.stamp, ros::Time::now(), tf_buffer, corrected_bbox_array, t13);
        fixed_frame = target_frame;
        output_bbox_array = transformed_bbox_array;
    } else {
        fixed_frame = lidar_frame;
        output_bbox_array = integration_bbox_array;
    }

    Tracking_->tracking(output_bbox_array, track_bbox_array, track_text_array, bba_msg->header.stamp, t12);
    publish_stamp = ros::Time::now();
    // Tracking_->correctionBboxRelativeSpeed(track_bbox_array, bba_msg->header.stamp, publish_stamp, corrected_bbox_array, t13);
    
    // pub_track_test.publish(bba2msg(filtered_bbox_array, publish_stamp, fixed_frame));
    pub_track_box.publish(bba2msg(track_bbox_array, publish_stamp, fixed_frame));
    pub_track_model.publish(bba2ma(track_bbox_array, publish_stamp, fixed_frame));
    pub_track_text.publish(ta2msg(track_text_array, publish_stamp, fixed_frame));

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
    // pub_track_test = pnh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/mobinha/perception/lidar/track_test", 10);
    pub_track_text = pnh.advertise<visualization_msgs::MarkerArray>("/mobinha/visualize/visualize/track_text", 10);
    pub_track_model = pnh.advertise<visualization_msgs::MarkerArray>("/mobinha/visualize/visualize/track_model", 10);

    // Tracking 객체를 초기화
    Tracking_ = boost::make_shared<Tracking>(pnh);

    ros::Subscriber sub_cluster_box = nh.subscribe("/cloud_segmentation/cluster_box", 10, callbackCluster);
    ros::Subscriber sub_deep_box = nh.subscribe("/deep_box", 10, callbackDeep);
    ros::Subscriber sub_waypoints = nh.subscribe("/waypoints", 1, callbackWaypoints);

    ros::spin();
    return 0;
}
*/