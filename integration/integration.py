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
from novatel_oem7_msgs.msg import INSPVA
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, TransformStamped

package_path = roslib.packages.get_pkg_dir('lidar_tracking')
dae_path = os.path.join(package_path, 'urdf/car.dae')  # car.dae 파일 경로 설정
map_path = os.path.join(package_path, 'map/songdo.json')


def EgoCarViz():
    marker = Marker()
    marker.header.frame_id = 'ego_car'
    marker.ns = 'car'
    marker.id = 0
    marker.type = Marker.MESH_RESOURCE
    marker.mesh_resource = "file://" + dae_path
    marker.action = Marker.ADD
    marker.lifetime = rospy.Duration(0)
    marker.scale.x = 2.0
    marker.scale.y = 2.0
    marker.scale.z = 2.0
    marker.color.r = 0.7
    marker.color.g = 0.7
    marker.color.b = 0.7
    marker.color.a = 1.0
    marker.pose.position.x = 0
    marker.pose.position.y = 0
    marker.pose.position.z = 1.0
    quaternion = tf.transformations.quaternion_from_euler(
        0, 0, math.radians(90))
    marker.pose.orientation.x = quaternion[0]
    marker.pose.orientation.y = quaternion[1]
    marker.pose.orientation.z = quaternion[2]
    marker.pose.orientation.w = quaternion[3]
    return marker

def rotate_quaternion_yaw(quaternion, yaw_degrees):
    yaw_radians = math.radians(yaw_degrees)
    q_yaw = (0, 0, math.sin(yaw_radians / 2), math.cos(yaw_radians / 2))
    x1, y1, z1, w1 = quaternion
    x2, y2, z2, w2 = q_yaw
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    )

def Bound(ns, id_, n, points, type_, color):
    if type_ == 'solid':
        marker = Line('%s_%s' % (ns, id_), n, 0.15, color)
        for pt in points:
            marker.points.append(Point(x=pt[0], y=pt[1], z=0.0))

    elif type_ == 'dotted':
        marker = Points('%s_%s' % (ns, id_), n, 0.15, color)
        for pt in points:
            marker.points.append(Point(x=pt[0], y=pt[1], z=0.0))

    return marker

def Points(ns, id_, scale, color):
    marker = Marker()
    marker.type = Marker.POINTS
    marker.action = Marker.ADD
    marker.header.frame_id = 'world'
    marker.ns = ns
    marker.id = id_
    marker.lifetime = rospy.Duration(0)
    marker.scale.x = scale
    marker.scale.y = scale
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    return marker

def Line(ns, id_, scale, color):
    marker = Marker()
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.header.frame_id = 'world'
    marker.ns = ns
    marker.id = id_
    marker.lifetime = rospy.Duration(0)
    marker.scale.x = scale
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    return marker

def Sphere(ns, id_, data, scale, color):
    marker = Marker()
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.header.frame_id = 'world'
    marker.ns = ns
    marker.id = id_
    marker.lifetime = rospy.Duration(0)
    marker.scale.x = scale
    marker.scale.y = scale
    marker.scale.z = scale
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    marker.pose.position.x = data[0]
    marker.pose.position.y = data[1]
    marker.pose.position.z = 1.0
    return marker

def Node(id_, n, pt, color):
    marker = Text('graph_id', n, 2.5, color, id_)
    marker.pose.position = Point(x=pt[0], y=pt[1], z=1.0)
    return marker

def Edge(n, points, color):
    if len(points) == 2:
        wx, wy = zip(*points)
        itp = QuadraticSplineInterpolate(list(wx), list(wy))
        pts = []
        for ds in np.arange(0.0, itp.s[-1], 0.5):
            pts.append(itp.calc_position(ds))
        points = pts

    marker1 = Line('edge_line', n, 0.5, color)
    for pt in points:
        marker1.points.append(Point(x=pt[0], y=pt[1], z=0.0))

    marker2 = Arrow('edge_arrow', n, (1.0, 2.0, 4.0), color)
    num = len(points)
    if num > 2:
        marker2.points.append(
            Point(x=points[-min(max(num, 3), 5)][0], y=points[-min(max(num, 3), 5)][1]))
    else:
        marker2.points.append(Point(x=points[-2][0], y=points[-2][1]))
    marker2.points.append(Point(x=points[-1][0], y=points[-1][1]))
    return marker1, marker2

def Text(ns, id_, scale, color, text):
    marker = Marker()
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD
    marker.header.frame_id = 'world'
    marker.ns = ns
    marker.id = id_
    marker.lifetime = rospy.Duration(0)
    marker.text = text
    marker.scale.z = scale
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    return marker

def Arrow(ns, id_, scale, color):
    marker = Marker()
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    marker.header.frame_id = 'world'
    marker.ns = ns
    marker.id = id_
    marker.lifetime = rospy.Duration(0)
    marker.scale.x = scale[0]
    marker.scale.y = scale[1]
    marker.scale.z = scale[2]
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    return marker

def VectorMapVis(map_data):
    lanelet = map_data['lanelets']
    stoplines = map_data['stoplines']
    safetysigns = map_data['safetysigns']
    surfacemarks = map_data['surfacemarks']
    trafficlights = map_data['trafficlights']
    vehicleprotectionsafetys = map_data['vehicleprotectionsafetys']
    postpoints = map_data['postpoints']

    array = MarkerArray()
    for id_, data in lanelet.items():
        for n, (leftBound, leftType) in enumerate(zip(data['leftBound'], data['leftType'])):
            marker = Bound('leftBound', id_, n, leftBound,
                           leftType, (1.0, 1.0, 1.0, 0.5))
            array.markers.append(marker)

        for n, (rightBound, rightType) in enumerate(zip(data['rightBound'], data['rightType'])):
            marker = Bound('rightBound', id_, n, rightBound,
                           rightType, (1.0, 1.0, 1.0, 0.5))
            array.markers.append(marker)

    for id_, data in safetysigns.items():
        marker = Bound('safetysign', id_, n, data,
                       'solid', (1.0, 1.0, 1.0, 0.5))
        array.markers.append(marker)

    for id_, data in stoplines.items():
        marker = Bound('stopline', id_, 0, data, 'solid', (1.0, 1.0, 1.0, 0.5))
        array.markers.append(marker)

    for id_, data in surfacemarks.items():
        marker = Bound('surfacemark', id_, 0, data,
                       'solid', (1.0, 1.0, 1.0, 0.5))
        array.markers.append(marker)

    for id_, data in trafficlights.items():
        marker = Sphere('traifficlight_%s' %
                        (id_), 0, data, 0.1, (1.0, 1.0, 1.0, 0.5))
        array.markers.append(marker)

    for id_, data in vehicleprotectionsafetys.items():
        marker = Bound('vehicleprotectionsafety', id_, 0,
                       data, 'solid', (1.0, 1.0, 1.0, 0.5))
        array.markers.append(marker)

    for id_, data in postpoints.items():
        marker = PostPoint('postpoint_%s' % (id_), 0, data,
                           0.2, 4.0, (1.0, 1.0, 1.0, 0.5))
        array.markers.append(marker)

    return array

def LaneletMapViz(lanelet, for_viz):
    array = MarkerArray()
    for id_, data in lanelet.items():
        for n, (leftBound, leftType) in enumerate(zip(data['leftBound'], data['leftType'])):
            marker = Bound('leftBound', id_, n, leftBound,
                           leftType, (1.0, 1.0, 1.0, 1.0))
            array.markers.append(marker)

        for n, (rightBound, rightType) in enumerate(zip(data['rightBound'], data['rightType'])):
            marker = Bound('rightBound', id_, n, rightBound,
                           rightType, (1.0, 1.0, 1.0, 1.0))
            array.markers.append(marker)

    for n, (points, type_) in enumerate(for_viz):
        if type_ == 'stop_line':
            marker = Bound('for_viz', n, n, points,
                           'solid', (1.0, 1.0, 1.0, 1.0))
            array.markers.append(marker)
        else:
            marker = Bound('for_viz', n, n, points,
                           type_, (1.0, 1.0, 1.0, 1.0))
            array.markers.append(marker)

    return array

def MicroLaneletGraphViz(lanelet, graph):
    array = MarkerArray()

    for n, (node_id, data) in enumerate(graph.items()):
        split = node_id.split('_')

        if len(split) == 1:
            id_ = split[0]
            from_idx = lanelet[id_]['idx_num'] // 2
            from_pts = lanelet[id_]['waypoints']
            marker = Node(node_id, n, from_pts[from_idx], (1.0, 1.0, 1.0, 1.0))
            array.markers.append(marker)

            for m, target_node_id in enumerate(data.keys()):
                split = target_node_id.split('_')
                pts = []

                if len(split) == 1:
                    target_id = split[0]
                    to_pts = lanelet[target_id]['waypoints']
                    to_idx = lanelet[target_id]['idx_num'] // 2
                    pts.extend(from_pts[from_idx:])
                    pts.extend(to_pts[:to_idx])
                else:
                    target_id = split[0]
                    cut_n = int(split[1])
                    to_pts = lanelet[target_id]['waypoints']
                    to_idx = sum(lanelet[target_id]['cut_idx'][cut_n]) // 2
                    pts.extend(from_pts[from_idx:])
                    pts.extend(to_pts[:to_idx])

                marker1, marker2 = Edge(n*100000+m, pts, (0.0, 1.0, 0.0, 0.5))
                array.markers.append(marker1)
                array.markers.append(marker2)

        else:
            id_ = split[0]
            cut_n = int(split[1])
            from_idx = sum(lanelet[id_]['cut_idx'][cut_n]) // 2
            from_pts = lanelet[id_]['waypoints']
            marker = Node(node_id, n, from_pts[from_idx], (1.0, 1.0, 1.0, 1.0))
            array.markers.append(marker)

            for m, target_node_id in enumerate(data.keys()):
                split = target_node_id.split('_')
                pts = []

                if len(split) == 1:
                    target_id = split[0]
                    to_pts = lanelet[target_id]['waypoints']
                    to_idx = lanelet[target_id]['idx_num'] // 2
                    pts.extend(from_pts[from_idx:])
                    pts.extend(to_pts[:to_idx])
                else:
                    target_id = split[0]
                    cut_n = int(split[1])
                    to_pts = lanelet[target_id]['waypoints']
                    to_idx = sum(lanelet[target_id]['cut_idx'][cut_n]) // 2
                    pts = [from_pts[from_idx], to_pts[to_idx]]

                marker1, marker2 = Edge(n*100000+m, pts, (0.0, 1.0, 0.0, 0.5))
                array.markers.append(marker1)
                array.markers.append(marker2)

    return array

def euc_distance(pt1, pt2):
    return np.sqrt((pt2[0]-pt1[0])**2+(pt2[1]-pt1[1])**2)

def find_nearest_idx(pts, pt):
    min_dist = float('inf')
    min_idx = 0

    for idx, pt1 in enumerate(pts):
        dist = euc_distance(pt1, pt)
        if dist < min_dist:
            min_dist = dist
            min_idx = idx

    return min_idx

class QuadraticSplineInterpolate:
    def __init__(self, x, y):
        self.s = self.calc_s(x, y)
        self.sx = interp1d(self.s, x, fill_value="extrapolate")
        self.sy = interp1d(self.s, y, fill_value="extrapolate")

    def calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = [math.sqrt(idx ** 2 + idy ** 2)
                   for (idx, idy) in zip(dx, dy)]
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_d(self, sp, x):
        dx = 1.0
        dp = sp(x+dx)
        dm = sp(x-dx)
        d = (dp - dm) / dx
        return d

    def calc_dd(self, sp, x):
        dx = 2.0
        ddp = self.calc_d(sp, x+dx)
        ddm = self.calc_d(sp, x-dx)
        dd = (ddp - ddm) / dx
        return dd

    def calc_yaw(self, s):
        dx = self.calc_d(self.sx, s)
        dy = self.calc_d(self.sy, s)
        yaw = math.atan2(dy, dx)
        return yaw

    def calc_position(self, s):
        x = self.sx(s)
        y = self.sy(s)
        return x, y

    def calc_curvature(self, s):
        dx = self.calc_d(self.sx, s)
        ddx = self.calc_dd(self.sx, s)
        dy = self.calc_d(self.sy, s)
        ddy = self.calc_dd(self.sy, s)
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
        return k

class LaneletMap:
    def __init__(self, map_path, interp_distance=0.5):
        with open(map_path, 'r') as f:
            map_data = json.load(f)
        self.map_data = map_data
        self.lanelets = map_data['lanelets']
        self.groups = map_data['groups']
        self.precision = map_data['precision']
        self.for_viz = map_data['for_vis']
        self.base_lla = map_data['base_lla']

        self.interp_distance = interp_distance
        self.preprocess_lanelets()
    
    def preprocess_lanelets(self):
        for id_, lanelet in self.lanelets.items():
            waypoints = np.array(lanelet['waypoints'])
            if len(waypoints) < 2:
                continue  # 웨이포인트가 2개 미만인 경우 스킵
            wp_x = waypoints[:, 0]
            wp_y = waypoints[:, 1]

            delta_wp = np.diff(waypoints, axis=0)
            segment_lengths = np.hypot(delta_wp[:, 0], delta_wp[:, 1])
            distances_along_wp = np.concatenate(([0], np.cumsum(segment_lengths)))

            total_length = distances_along_wp[-1]
            num_points = int(total_length / self.interp_distance) + 1
            if num_points < 2:
                num_points = 2  # 최소 2개의 포인트 유지
            s_interp = np.linspace(0, total_length, num_points)

            x_interp = np.interp(s_interp, distances_along_wp, wp_x)
            y_interp = np.interp(s_interp, distances_along_wp, wp_y)

            lanelet['interp_waypoints'] = np.stack((x_interp, y_interp), axis=-1)

class MicroLaneletGraph:
    def __init__(self, lmap, cut_dist):
        self.cut_dist = cut_dist
        self.precision = lmap.precision
        self.lanelets = lmap.lanelets
        self.groups = lmap.groups
        self.generate_micro_lanelet_graph()

        self.pub_micro_lanelet_graph = rospy.Publisher(
            '/micro_lanelet_graph', MarkerArray, queue_size=1, latch=True)
        micro_lanelet_graph_viz = MicroLaneletGraphViz(
            self.lanelets, self.graph)
        self.pub_micro_lanelet_graph.publish(micro_lanelet_graph_viz)

    def generate_micro_lanelet_graph(self):
        self.graph = {}
        cut_idx = int(self.cut_dist / self.precision)

        for group in self.groups:
            group = copy.copy(group)

            if self.lanelets[group[0]]['length'] > self.lanelets[group[-1]]['length']:
                group.reverse()

            idx_num = self.lanelets[group[0]]['idx_num']

            cut_num = idx_num // cut_idx
            if idx_num % cut_idx != 0:
                cut_num += 1

            for n, id_ in enumerate(group):
                self.lanelets[id_]['cut_idx'] = []

                if n == 0:
                    for i in range(cut_num):
                        start_idx = i * cut_idx

                        if i == cut_num - 1:
                            end_idx = idx_num
                        else:
                            end_idx = start_idx + cut_idx

                        self.lanelets[id_]['cut_idx'].append(
                            [start_idx, end_idx])

                else:
                    for i in range(cut_num):
                        pre_id = group[n-1]
                        pre_end_idx = self.lanelets[pre_id]['cut_idx'][i][1] - 1

                        pt = self.lanelets[pre_id]['waypoints'][pre_end_idx]

                        if i == 0:
                            start_idx = 0
                            end_idx = find_nearest_idx(
                                self.lanelets[id_]['waypoints'], pt)

                        elif i == cut_num - 1:
                            start_idx = self.lanelets[id_]['cut_idx'][i-1][1]
                            end_idx = self.lanelets[id_]['idx_num']

                        else:
                            start_idx = self.lanelets[id_]['cut_idx'][i-1][1]
                            end_idx = find_nearest_idx(
                                self.lanelets[id_]['waypoints'], pt)

                        self.lanelets[id_]['cut_idx'].append(
                            [start_idx, end_idx])

        for id_, data in self.lanelets.items():
            if data['group'] is None:
                if self.graph.get(id_) is None:
                    self.graph[id_] = {}

                for p_id in data['successor']:
                    if self.lanelets[p_id]['group'] is None:
                        self.graph[id_][p_id] = data['length']
                    else:
                        self.graph[id_][p_id+'_0'] = data['length']

            else:
                last = len(data['cut_idx']) - 1
                for n in range(len(data['cut_idx'])):
                    new_id = '%s_%s' % (id_, n)
                    if self.graph.get(new_id) is None:
                        self.graph[new_id] = {}

                    if n == last:
                        for p_id in data['successor']:
                            if self.lanelets[p_id]['group'] is None:
                                self.graph[new_id][p_id] = self.cut_dist
                            else:
                                self.graph[new_id][p_id+'_0'] = self.cut_dist

                    else:
                        self.graph[new_id]['%s_%s' %
                                           (id_, n+1)] = self.cut_dist

                        s_idx, e_idx = self.lanelets[id_]['cut_idx'][n]

                        left_id = data['adjacentLeft']
                        if left_id is not None:
                            if sum(self.lanelets[id_]['leftChange'][s_idx:s_idx+(e_idx-s_idx)//2]) == (e_idx - s_idx)//2:
                                self.graph[new_id]['%s_%s' % (
                                    left_id, n+1)] = self.cut_dist + 10.0 + n * 0.1

                        right_id = data['adjacentRight']
                        if right_id is not None:
                            if sum(self.lanelets[id_]['rightChange'][s_idx:s_idx+(e_idx-s_idx)//2]) == (e_idx - s_idx)//2:
                                self.graph[new_id]['%s_%s' % (
                                    right_id, n+1)] = self.cut_dist + 10.0 + n * 0.1

        self.reversed_graph = {}
        for from_id, data in self.graph.items():
            for to_id in data:
                if self.reversed_graph.get(to_id) is None:
                    self.reversed_graph[to_id] = []

                self.reversed_graph[to_id].append(from_id)

class LocalizerHDMap:
    def __init__(self):

        self.interp_distance = 2.0

        rospy.init_node('Localizer')
        pub_lanelet_map = rospy.Publisher('/lanelet_map', MarkerArray, queue_size=1, latch=True)
        # map_path = "/home/q/software/Localizer_hdmap/songdo.json"
        self.lmap = LaneletMap(map_path, self.interp_distance)
        lanelet_map_viz = LaneletMapViz(self.lmap.lanelets, self.lmap.for_viz)
        pub_lanelet_map.publish(lanelet_map_viz)
        
        # microlanelet
        self.graph = MicroLaneletGraph(self.lmap, 15.0).graph

        self.pub_waypoints_marker = rospy.Publisher('/local_waypoints', MarkerArray, queue_size=1)

        rospy.loginfo("Lanelet map published on lanelet_map")

        self.ego_car = EgoCarViz()
        self.br = tf.TransformBroadcaster()

        # calibration
        self.static_br = tf2_ros.StaticTransformBroadcaster()
        static_transforms = [
            ((1.5275, 0.0, 0.0), (0, 0, 0, 1), 'ego_car', 'gps'),   # center
            ((1.06, 0, 1.22), rotate_quaternion_yaw((0, 0, 0, 1), -2.1), 'hesai_lidar', 'gps')
        ]
        self.publish_static_tfs(static_transforms)

        rospy.Subscriber('/novatel/oem7/inspva', INSPVA, self.novatel_cb)

        self.pub_ego_car = rospy.Publisher('/car_model', Marker, queue_size=1)

        self.build_waypoint_kdtree()

    
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
        self.yaw = self.yaw - 0.3 # control 에서 수정하도록 바꿔야 함

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
        self.update_local_waypoints()

    def publish_waypoints_marker(self, waypoints):
        marker = Marker()
        marker.header.frame_id = 'ego_car'
        marker.header.stamp = self.timestamp
        marker.ns = 'local_waypoints'
        marker.id = 0
        marker.type = Marker.POINTS  # 포인트로 표시
        marker.action = Marker.ADD
        marker.scale.x = 0.2  # 포인트 크기
        marker.scale.y = 0.2
        marker.color.r = 0.0
        marker.color.g = 1.0  # 녹색으로 표시
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.points = [Point(x=pt[0], y=pt[1], z=0.0) for pt in waypoints]

        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.pub_waypoints_marker.publish(marker_array)

    def build_waypoint_kdtree(self):
        all_waypoints = []
        for id_, lanelet in self.lmap.lanelets.items():
            waypoints = lanelet['interp_waypoints']
            all_waypoints.extend(waypoints)
        self.waypoints_np = np.array(all_waypoints)
        self.kdtree = KDTree(self.waypoints_np)

    
    def update_local_waypoints(self):
        # 차량의 현재 위치 가져오기
        x_vehicle = self.x
        y_vehicle = self.y

        if not hasattr(self, 'kdtree'):
            rospy.logerr("KD-Tree is not built yet.")
            self.publish_waypoints_marker([])
            return

        # 반경 100m 이내의 웨이포인트 인덱스 검색
        indices = self.kdtree.query_ball_point([x_vehicle, y_vehicle], r=100.0)

        if not indices:
            # 주변에 웨이포인트가 없는 경우
            self.publish_waypoints_marker([])
            return

        # 해당 웨이포인트들을 가져옵니다.
        nearby_waypoints = self.waypoints_np[indices]

        # 웨이포인트를 변환합니다.
        yaw_vehicle = self.yaw
        yaw_rad = math.radians(yaw_vehicle)
        cos_yaw = math.cos(-yaw_rad)
        sin_yaw = math.sin(-yaw_rad)

        dx = nearby_waypoints[:, 0] - x_vehicle
        dy = nearby_waypoints[:, 1] - y_vehicle

        x_e = dx * cos_yaw - dy * sin_yaw
        y_e = dx * sin_yaw + dy * cos_yaw

        transformed_waypoints = list(zip(x_e, y_e))

        # 변환된 노드를 퍼블리시
        self.publish_waypoints_marker(transformed_waypoints)

def main():
    localizer = LocalizerHDMap()
    rospy.spin()

if __name__ == '__main__':
    main()
