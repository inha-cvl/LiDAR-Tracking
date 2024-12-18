#!/usr/bin/env python3
import os
import sys
import math
import numpy as np
import pymap3d
import rospy
import roslib
import sensor_msgs.point_cloud2 as pc2
import tf
import tf2_ros

from geometry_msgs.msg import Point, Vector3, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from jsk_rviz_plugins.msg import OverlayText
from jsk_recognition_msgs.msg import BoundingBox
from novatel_oem7_msgs.msg import INSPVA
from sensor_msgs.msg import PointCloud2
from scipy.spatial import KDTree

package_path = roslib.packages.get_pkg_dir('lidar_tracking')
sys.path.append(package_path+'/scripts/lib')
from utils import (
    rotate_quaternion_yaw,
    query_local_waypoints,
    LaneletMap,
    LaneletMapViz,
    create_car_marker,
    publish_static_tfs,
    calculate_velocity_and_heading,
    create_text_marker,
    create_ego_info_overlay
)

# parameters
lidar_frame = rospy.get_param('/Public/lidar_frame', "hesai_lidar")
ins_frame = rospy.get_param('/Public/ins_frame', "gps")
target_frame = rospy.get_param('/Public/target_frame', "ego_car")
world_frame = rospy.get_param('/Public/world_frame', "world")
map_name = rospy.get_param('/Public/map', "map/songdo.json")
dae_path = os.path.join(package_path, 'urdf/car.dae')
map_path = os.path.join(package_path, map_name)

crop_distance = 80.0
interp_distance = 2.0

# ioniq
t_gps_ego = np.array([1.527, 0, 0])
q_gps_ego = rotate_quaternion_yaw((0, 0, 0, 1), -0.3)
t_gps_lidar = np.array([1.06, 0, 2.1])
q_gps_lidar = rotate_quaternion_yaw((0, 0, 0, 1), -1.5)

# i30
t_gps_target = np.array([1.4, 0, 0])
q_gps_target = rotate_quaternion_yaw((0, 0, 0, 1), 0.0)

static_transforms = [
    (t_gps_ego, q_gps_ego, target_frame, ins_frame),
    (t_gps_lidar, q_gps_lidar, lidar_frame, ins_frame),
    (t_gps_target, q_gps_target, 'target_car', "gps2")
]

EGO_CAR_COLOR = (0.7, 0.7, 0.7, 1.0)
TARGET_CAR_COLOR = (0.0, 0.98, 1.0, 0.7)

class Integration:
    def __init__(self):
        rospy.init_node('Integration')

        self.lmap = LaneletMap(map_path, interp_distance)
        lanelet_map_viz = LaneletMapViz(self.lmap.lanelets, self.lmap.for_viz)
        pub_lanelet_map = rospy.Publisher('/lanelet_map', MarkerArray, queue_size=1, latch=True)
        pub_lanelet_map.publish(lanelet_map_viz)

        self.use_waypoints = True
        self.r = crop_distance
        self.build_waypoint_kdtree()
        self.pub_waypoints = rospy.Publisher('/waypoints', PointCloud2, queue_size=1)

        self.ego_car = create_car_marker(target_frame, True, EGO_CAR_COLOR, dae_path)
        self.pub_ego_car = rospy.Publisher('/ego_model', Marker, queue_size=1)
        self.target_car = create_car_marker('target_car', False, TARGET_CAR_COLOR, dae_path)
        self.pub_target_car = rospy.Publisher('/target_model', Marker, queue_size=1)
        self.pub_target_box = rospy.Publisher('/target_box', BoundingBox, queue_size=1)

        self.pub_ego_speed_marker = rospy.Publisher('/ego_speed', Marker, queue_size=1)
        self.pub_target_speed_marker = rospy.Publisher('/target_speed', Marker, queue_size=1)
        self.pub_ego_info = rospy.Publisher('/car_info', OverlayText, queue_size=1)

        self.br = tf.TransformBroadcaster()
        self.static_br = tf2_ros.StaticTransformBroadcaster()
        publish_static_tfs(self.static_br, static_transforms)

        rospy.Subscriber('/novatel/oem7/inspva', INSPVA, self.novatel_cb)
        rospy.Subscriber('/novatel/oem7/inspva2', INSPVA, self.novatel_cb2)

        self.x,self.y,self.z,self.azimuth,self.vx,self.vy = 0,0,0,0,0,0
        self.timestamp = rospy.Time(0)

        rospy.Timer(rospy.Duration(0.05), self.periodic_update_cb) # 20Hz

        rospy.loginfo("Initialized")

    def build_waypoint_kdtree(self):
        all_waypoints = []
        for _, lanelet in self.lmap.lanelets.items():
            all_waypoints.extend(lanelet['interp_waypoints'])
        self.waypoints_np = np.array(all_waypoints)
        self.kdtree = KDTree(self.waypoints_np) if len(all_waypoints) > 0 else None

    def periodic_update_cb(self, event):
        if self.use_waypoints and self.kdtree is not None and hasattr(self, 'waypoints_np'):
            transformed_waypoints = query_local_waypoints(self.kdtree, self.waypoints_np, self.x, self.y, self.azimuth, self.r)
            if transformed_waypoints is not None and self.timestamp:
                point_cloud = pc2.create_cloud_xyz32(
                    header=rospy.Header(frame_id=target_frame, stamp=self.timestamp),
                    points=transformed_waypoints
                )
                self.pub_waypoints.publish(point_cloud)

        ego_overlay = create_ego_info_overlay(self.x, self.y, self.azimuth, self.vx, self.vy)
        self.pub_ego_info.publish(ego_overlay)

    def novatel_cb(self, msg):
        self.timestamp = msg.header.stamp
        self.x, self.y, self.z = pymap3d.geodetic2enu(
            msg.latitude, msg.longitude, 0,
            self.lmap.base_lla[0], self.lmap.base_lla[1], 0)
        self.roll = msg.roll
        self.pitch = msg.pitch
        self.azimuth = (90 - msg.azimuth) % 360

        quaternion = tf.transformations.quaternion_from_euler(
            math.radians(self.roll), math.radians(self.pitch), math.radians(self.azimuth))
        self.br.sendTransform(
            (self.x, self.y, self.z),
            (quaternion[0], quaternion[1], quaternion[2], quaternion[3]),
            self.timestamp,
            ins_frame,
            world_frame
        )

        self.pub_ego_car.publish(self.ego_car)
        vx, vy, v = calculate_velocity_and_heading(msg)
        self.vx, self.vy = vx, vy  # 상태 변수 업데이트

        speed_marker = create_text_marker(target_frame, "Speed: {:.2f} km/h".format(v*3.6), self.timestamp, Point(0, 0, 3.0), EGO_CAR_COLOR)
        self.pub_ego_speed_marker.publish(speed_marker)

    # integration bag 사용 시
    def novatel_cb2(self, msg):
        timestamp = msg.header.stamp
        x, y, z = pymap3d.geodetic2enu(
            msg.latitude, msg.longitude, 0,
            self.lmap.base_lla[0], self.lmap.base_lla[1], 0)
        roll = msg.roll
        pitch = msg.pitch
        azimuth = (90 - msg.azimuth) % 360

        quaternion = tf.transformations.quaternion_from_euler(
            math.radians(roll), math.radians(pitch), math.radians(azimuth))
        self.br.sendTransform(
            (x, y, z),
            (quaternion[0], quaternion[1], quaternion[2], quaternion[3]),
            timestamp,
            'gps2',
            world_frame
        )

        self.pub_target_car.publish(self.target_car)

        target_box = BoundingBox()
        target_box.header.stamp = timestamp
        target_box.header.frame_id = "target_car"
        target_box.pose.position.z = 1.06
        target_box.dimensions = Vector3(x=4.34, y=1.795, z=1.455)  # i30
        self.pub_target_box.publish(target_box)

        vx, vy, v = calculate_velocity_and_heading(msg)
        speed_marker = create_text_marker('target_car', "Speed: {:.2f} km/h".format(v*3.6), timestamp, Point(0, 0, 3.0), TARGET_CAR_COLOR)
        self.pub_target_speed_marker.publish(speed_marker)

def main():
    integration = Integration()
    rospy.spin()

if __name__ == '__main__':
    main()
