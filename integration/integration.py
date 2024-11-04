import os
import copy
from scipy.interpolate import interp1d
from scipy.spatial import KDTree
import numpy as np
import pymap3d
import json
import math
import tf
import tf2_ros
import rospy
import roslib
from std_msgs.msg import Header, ColorRGBA
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from novatel_oem7_msgs.msg import INSPVA
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, TransformStamped, Pose, Vector3, Quaternion

from utils import *

package_path = roslib.packages.get_pkg_dir('lidar_tracking')
dae_path = os.path.join(package_path, 'urdf/car.dae')  # car.dae 파일 경로 설정
map_path = os.path.join(package_path, 'map/songdo.json')

class Integration:
    def __init__(self):
        rospy.init_node('Integration')
        
        # lanelet
        self.interp_distance = 2.0
        self.lmap = LaneletMap(map_path, self.interp_distance)
        lanelet_map_viz = LaneletMapViz(self.lmap.lanelets, self.lmap.for_viz)
        pub_lanelet_map = rospy.Publisher('/lanelet_map', MarkerArray, queue_size=1, latch=True)
        pub_lanelet_map.publish(lanelet_map_viz)
        
        # microlanelet
        # self.mlgraph = MicroLaneletGraph(self.lmap, 15.0)
        # micro_lanelet_graph_viz = MicroLaneletGraphViz(self.lmap.lanelets, self.mlgraph.graph)
        # pub_micro_lanelet_graph = rospy.Publisher('/micro_lanelet_graph', MarkerArray, queue_size=1, latch=True)
        # pub_micro_lanelet_graph.publish(micro_lanelet_graph_viz)

        # waypoints
        self.use_waypoints = False
        self.r = 100.0
        self.build_waypoint_kdtree()
        self.pub_waypoints = rospy.Publisher('/waypoints', PointCloud2, queue_size=1)

        # ego car marker
        self.ego_car = self.egoCar()
        self.pub_ego_car = rospy.Publisher('/car_model', Marker, queue_size=1)
        
        # calibration
        self.br = tf.TransformBroadcaster()
        self.static_br = tf2_ros.StaticTransformBroadcaster()
        static_transforms = [
            # ioniq
            ((1.5275, 0.0, 0.0), rotate_quaternion_yaw((0, 0, 0, 1), -0.3), 'ego_car', 'gps'),
            ((1.06, 0, 1.22), rotate_quaternion_yaw((0, 0, 0, 1), -2.1), 'hesai_lidar', 'gps')

            # avente
            # ((1.5275, -0.3, 0.0), rotate_quaternion_yaw((0, 0, 0, 1), 0.0), 'ego_car', 'gps'),
            # ((1.06, 0, 1.22), rotate_quaternion_yaw((0, 0, 0, 1), -2.1), 'hesai_lidar', 'gps')
        ]
        self.publish_static_tfs(static_transforms)

        rospy.Subscriber('/novatel/oem7/inspva', INSPVA, self.novatel_cb)

        rospy.loginfo("Initialized")

    def egoCar(self):
        marker = Marker(
            header=Header(frame_id='ego_car'),
            ns='ego_car',
            id=0,
            type=Marker.MESH_RESOURCE,
            mesh_resource="file://" + dae_path,
            action=Marker.ADD,
            lifetime=rospy.Duration(0.05),
            scale=Vector3(x=2.0, y=2.0, z=2.0),
            color=ColorRGBA(r=0.7, g=0.7, b=0.7, a=1.0),
            pose=Pose(
                position=Point(x=0, y=0, z=1.0),
                orientation=Quaternion(*tf.transformations.quaternion_from_euler(0, 0, math.radians(90)))
            )
        )
        return marker

    def publish_static_tfs(self, transforms):
        static_transformStamped_vec = []
        for translation, rotation, child_frame, parent_frame in transforms:
            static_transformStamped = TransformStamped()
            static_transformStamped.header.frame_id = parent_frame
            static_transformStamped.child_frame_id = child_frame
            static_transformStamped.transform.translation.x = translation[0]
            static_transformStamped.transform.translation.y = translation[1]
            static_transformStamped.transform.translation.z = translation[2]
            static_transformStamped.transform.rotation.x = rotation[0]
            static_transformStamped.transform.rotation.y = rotation[1]
            static_transformStamped.transform.rotation.z = rotation[2]
            static_transformStamped.transform.rotation.w = rotation[3]
            static_transformStamped_vec.append(static_transformStamped)
        
        self.static_br.sendTransform(static_transformStamped_vec)

    def novatel_cb(self, msg):
        self.timestamp = msg.header.stamp
        self.latitude = msg.latitude
        self.longitude = msg.longitude
        self.altitude = msg.height
        self.x, self.y, self.z = pymap3d.geodetic2enu(
            msg.latitude, msg.longitude, 0, self.lmap.base_lla[0], self.lmap.base_lla[1], 0)
        self.roll = msg.roll
        self.pitch = msg.pitch
        self.yaw = 90 - msg.azimuth + 360 if (-270 <= 90 - msg.azimuth <= -180) else 90 - msg.azimuth
        # self.yaw = self.yaw - 0.3 # 설치 오차

        quaternion = tf.transformations.quaternion_from_euler(
            math.radians(self.roll), math.radians(self.pitch), math.radians(self.yaw))  # RPY
        self.br.sendTransform(
            (self.x, self.y, self.z),
            (quaternion[0], quaternion[1],
                quaternion[2], quaternion[3]),
            self.timestamp,
            'gps',
            'world'
        )
        
        self.pub_ego_car.publish(self.ego_car)
        self.update_local_waypoints(self.r)

    def build_waypoint_kdtree(self):
        all_waypoints = []
        for id_, lanelet in self.lmap.lanelets.items():
            waypoints = lanelet['interp_waypoints']
            all_waypoints.extend(waypoints)
        self.waypoints_np = np.array(all_waypoints)
        self.kdtree = KDTree(self.waypoints_np)

    def update_local_waypoints(self, r):
        if self.use_waypoints == False:
            return

        if not hasattr(self, 'kdtree'):
            rospy.logerr("KD-Tree is not built yet.")
            return

        indices = self.kdtree.query_ball_point([self.x, self.y], r)

        if not indices:
            return

        nearby_waypoints = self.waypoints_np[indices]

        yaw_vehicle = self.yaw
        yaw_rad = math.radians(yaw_vehicle)
        cos_yaw = math.cos(-yaw_rad)
        sin_yaw = math.sin(-yaw_rad)

        dx = nearby_waypoints[:, 0] - self.x
        dy = nearby_waypoints[:, 1] - self.y

        x_e = dx * cos_yaw - dy * sin_yaw
        y_e = dx * sin_yaw + dy * cos_yaw

        transformed_waypoints = list(zip(x_e, y_e, np.zeros_like(x_e)))

        point_cloud = pc2.create_cloud_xyz32(
            header=rospy.Header(frame_id='ego_car', stamp=self.timestamp),
            points=transformed_waypoints
        )

        self.pub_waypoints.publish(point_cloud)

def main():
    integration = Integration()
    rospy.spin()

if __name__ == '__main__':
    main()
