#!/usr/bin/env python

import os
import csv
import math
import numpy as np
import rosbag
import pymap3d
import tf.transformations
from novatel_oem7_msgs.msg import INSPVA

def rotate_quaternion_yaw(quaternion, yaw_angle):
    # Quaternion을 yaw 축으로 주어진 각도만큼 회전시킵니다.
    q_rotation = tf.transformations.quaternion_from_euler(0, 0, math.radians(yaw_angle))
    q_new = tf.transformations.quaternion_multiply(q_rotation, quaternion)
    return q_new

def gpsTime(week_number, week_milliseconds):
    # GPS 시간을 초 단위로 계산합니다.
    return week_number * 7 * 24 * 3600 + week_milliseconds / 1000.0

def main():
    bag_file = 'undistortion_test.bag'  # 실제 bag 파일의 경로로 변경해주세요.
    bag = rosbag.Bag(bag_file)
    
    ego_file = open('ego.csv', 'w', newline='')
    target_file = open('target.csv', 'w', newline='')
    ego_writer = csv.writer(ego_file)
    target_writer = csv.writer(target_file)

    # CSV 파일 헤더 작성
    ego_writer.writerow(['rostime', 'gpstime', 'world_x', 'world_y', 'azimuth', 'vx', 'vy'])
    target_writer.writerow(['rostime', 'gpstime', 'world_x', 'world_y', 'azimuth', 'vx', 'vy'])

    # 기준 위도, 경도 설정 (필요에 따라 실제 값으로 수정)
    base_lat = 37.38435
    base_lon = 126.65303
    base_alt = 0  # 해발 고도 (필요에 따라 수정)

    # 차량의 변환 설정
    t_gps_lidar = np.array([1.06, 0, 2.1])  # ego 차량의 변환
    q_gps_lidar = rotate_quaternion_yaw((0, 0, 0, 1), -1.5)
    
    t_gps_target = np.array([1.4, 0, 0])  # target 차량의 변환
    q_gps_target = rotate_quaternion_yaw((0, 0, 0, 1), 0.0)

    for topic, msg, t in bag.read_messages(topics=['/novatel/oem7/inspva', '/novatel/oem7/inspva2']):
        if topic == '/novatel/oem7/inspva':
            # ego 차량 데이터 처리
            timestamp = t.to_sec()
            # ENU 좌표계로 변환
            x, y, z = pymap3d.geodetic2enu(
                msg.latitude, msg.longitude, 0, base_lat, base_lon, base_alt)
            roll = msg.roll
            pitch = msg.pitch
            azimuth = (90 - msg.azimuth) % 360
            quaternion = tf.transformations.quaternion_from_euler(
                math.radians(roll), math.radians(pitch), math.radians(azimuth))  # RPY

            # 차량 좌표계에서의 속도 계산
            azimuth_rad_original = math.radians(msg.azimuth)
            cos_azimuth = math.cos(azimuth_rad_original)
            sin_azimuth = math.sin(azimuth_rad_original)
            vx = msg.north_velocity * cos_azimuth + msg.east_velocity * sin_azimuth
            vy = -msg.north_velocity * sin_azimuth + msg.east_velocity * cos_azimuth

            # 평가를 위한 위치 및 방향 계산
            gps_time = gpsTime(msg.nov_header.gps_week_number, msg.nov_header.gps_week_milliseconds)
            t_world_gps = np.array([x, y, z])
            q_world_gps = quaternion
            R_world_gps = tf.transformations.quaternion_matrix(q_world_gps)[:3, :3]
            t_gps_lidar_in_world = R_world_gps.dot(t_gps_lidar)
            t_world_lidar = t_world_gps + t_gps_lidar_in_world
            q_world_lidar = tf.transformations.quaternion_multiply(q_world_gps, q_gps_lidar)
            _, _, yaw_lidar = tf.transformations.euler_from_quaternion(q_world_lidar)
            azimuth_lidar = (math.degrees(yaw_lidar) + 360) % 360

            # ego.csv에 데이터 저장
            ego_writer.writerow([timestamp, gps_time, t_world_lidar[0], t_world_lidar[1], azimuth_lidar, vx, vy])

        elif topic == '/novatel/oem7/inspva2':
            # target 차량 데이터 처리
            timestamp = t.to_sec()
            # ENU 좌표계로 변환
            x, y, z = pymap3d.geodetic2enu(
                msg.latitude, msg.longitude, 0, base_lat, base_lon, base_alt)
            roll = msg.roll
            pitch = msg.pitch
            azimuth = (90 - msg.azimuth) % 360
            quaternion = tf.transformations.quaternion_from_euler(
                math.radians(roll), math.radians(pitch), math.radians(azimuth))  # RPY

            # 차량 좌표계에서의 속도 계산
            azimuth_rad_original = math.radians(msg.azimuth)
            cos_azimuth = math.cos(azimuth_rad_original)
            sin_azimuth = math.sin(azimuth_rad_original)
            vx = msg.north_velocity * cos_azimuth + msg.east_velocity * sin_azimuth
            vy = -msg.north_velocity * sin_azimuth + msg.east_velocity * cos_azimuth

            # 평가를 위한 위치 및 방향 계산
            gps_time = gpsTime(msg.nov_header.gps_week_number, msg.nov_header.gps_week_milliseconds)
            t_world_gps = np.array([x, y, z])
            q_world_gps = quaternion
            R_world_gps = tf.transformations.quaternion_matrix(q_world_gps)[:3, :3]
            t_gps_target_in_world = R_world_gps.dot(t_gps_target)
            t_world_target = t_world_gps + t_gps_target_in_world
            q_world_target = tf.transformations.quaternion_multiply(q_world_gps, q_gps_target)
            _, _, yaw_target = tf.transformations.euler_from_quaternion(q_world_target)
            azimuth_target = (math.degrees(yaw_target) + 360) % 360

            # target.csv에 데이터 저장
            target_writer.writerow([timestamp, gps_time, t_world_target[0], t_world_target[1], azimuth_target, vx, vy])

    # 파일 닫기
    ego_file.close()
    target_file.close()
    bag.close()

if __name__ == '__main__':
    main()
