import copy
import json
import math
import numpy as np
import rospy
import tf
from scipy.interpolate import interp1d
from geometry_msgs.msg import Point, TransformStamped, Pose, Vector3, Quaternion
from jsk_rviz_plugins.msg import OverlayText
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

#evaluation
def gpsTime(gps_week_number, gps_week_milliseconds):
    gps_epoch_unix = 315964800  # UNIX 타임스탬프로 GPS 에포크 시간 (1980-01-06)
    gps_seconds = gps_week_number * 604800 + gps_week_milliseconds / 1000.0
    gps_time = gps_epoch_unix + gps_seconds
    return gps_time

def rotate_quaternion_yaw(quaternion, yaw_degrees):
    yaw_radians = math.radians(yaw_degrees)
    q_yaw = tf.transformations.quaternion_from_euler(0, 0, yaw_radians)
    return tf.transformations.quaternion_multiply(quaternion, q_yaw)

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

def create_car_marker(frame_id, use_embedded_materials, color, dae_path):
    return Marker(
        header=Header(frame_id=frame_id),
        ns=frame_id,
        id=0,
        type=Marker.MESH_RESOURCE,
        mesh_resource="file://" + dae_path,
        mesh_use_embedded_materials=use_embedded_materials,
        action=Marker.ADD,
        lifetime=rospy.Duration(0.05),
        scale=Vector3(x=2.0, y=2.0, z=2.0),
        color=ColorRGBA(*color),
        pose=Pose(
            position=Point(x=0, y=0, z=1.0),
            orientation=Quaternion(*tf.transformations.quaternion_from_euler(0, 0, math.radians(90)))
        )
    )

def create_ego_info_overlay(x, y, azimuth, vx, vy):
    v = math.sqrt(vx**2 + vy**2)
    text = ("Position:\nx: {:.2f}\ny: {:.2f}\nazimuth: {:.2f}\n\n"
            "Speed: {:.2f} km/h").format(x, y, azimuth, v*3.6)

    overlay_text = OverlayText()
    overlay_text.action = OverlayText.ADD
    overlay_text.width = 400
    overlay_text.height = 200
    overlay_text.left = 10
    overlay_text.top = 10
    overlay_text.text_size = 14
    overlay_text.line_width = 2
    overlay_text.font = "DejaVu Sans Mono"
    overlay_text.text = text
    overlay_text.fg_color = ColorRGBA(0.0, 1.0, 0.0, 1.0)
    overlay_text.bg_color = ColorRGBA(0.0, 0.0, 0.0, 0.5)
    return overlay_text

def create_text_marker(frame_id, text, timestamp, position, color):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = timestamp
    marker.ns = frame_id + "_speed_text"
    marker.id = 0
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD
    marker.pose.position = position
    marker.pose.orientation.w = 1.0
    marker.scale.z = 1.2
    marker.color = ColorRGBA(color[0], color[1], color[2], color[3])
    marker.text = text
    marker.lifetime = rospy.Duration(0.1)
    return marker

def publish_static_tfs(static_br, transforms):
    static_transform_stamped_vec = []
    for translation, rotation, child_frame, parent_frame in transforms:
        st = TransformStamped()
        st.header.frame_id = parent_frame
        st.child_frame_id = child_frame
        st.transform.translation.x = translation[0]
        st.transform.translation.y = translation[1]
        st.transform.translation.z = translation[2]
        st.transform.rotation.x = rotation[0]
        st.transform.rotation.y = rotation[1]
        st.transform.rotation.z = rotation[2]
        st.transform.rotation.w = rotation[3]
        static_transform_stamped_vec.append(st)
    static_br.sendTransform(static_transform_stamped_vec)

def calculate_velocity_and_heading(msg):
    # INS 메세지에서 속도와 헤딩을 계산하는 유틸 함수
    azimuth_rad_original = math.radians(msg.azimuth)
    cos_azimuth = math.cos(azimuth_rad_original)
    sin_azimuth = math.sin(azimuth_rad_original)

    vx = msg.north_velocity * cos_azimuth + msg.east_velocity * sin_azimuth
    vy = -msg.north_velocity * sin_azimuth + msg.east_velocity * cos_azimuth
    v = math.sqrt(vx**2 + vy**2)
    return vx, vy, v

def query_local_waypoints(kdtree, waypoints_np, x, y, azimuth, r):
    if kdtree is None:
        return None

    indices = kdtree.query_ball_point([x, y], r)
    if not indices:
        return None

    nearby_waypoints = waypoints_np[indices]

    azimuth_rad = math.radians(azimuth)
    cos_azimuth = math.cos(-azimuth_rad)
    sin_azimuth = math.sin(-azimuth_rad)

    dx = nearby_waypoints[:, 0] - x
    dy = nearby_waypoints[:, 1] - y

    x_e = dx * cos_azimuth - dy * sin_azimuth
    y_e = dx * sin_azimuth + dy * cos_azimuth

    transformed_waypoints = list(zip(x_e, y_e, np.zeros_like(x_e)))
    return transformed_waypoints

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
