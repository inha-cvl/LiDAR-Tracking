#!/usr/bin/env python
import rosbag
from novatel_oem7_msgs.msg import INSPVA
import rospy
import os

def create_integration_bag(ioniq_bag_path, avente_bag_path, output_bag_path):
    # 먼저 ioniq_bag과 avente_bag의 INSPVA 메시지를 읽어옵니다.
    ioniq_msgs = []
    avente_msgs = []

    print("Reading ioniq bag...")
    with rosbag.Bag(ioniq_bag_path, 'r') as ioniq_bag:
        for topic, msg, t in ioniq_bag.read_messages(topics=['/novatel/oem7/inspva']):
            ioniq_msgs.append((msg, t))

    print("Reading avente bag...")
    with rosbag.Bag(avente_bag_path, 'r') as avente_bag:
        for topic, msg, t in avente_bag.read_messages(topics=['/novatel/oem7/inspva']):
            avente_msgs.append((msg, t))

    # ioniq_msgs와 avente_msgs를 gps_time을 기준으로 동기화합니다.
    # avente_msgs의 header.stamp를 ioniq_msgs의 header.stamp와 동일한 gps_time에 해당하도록 조정합니다.

    # ioniq_msgs와 avente_msgs의 gps_time 목록을 가져옵니다.
    ioniq_gps_times = []
    for msg, t in ioniq_msgs:
        gps_week = msg.nov_header.gps_week_number
        gps_seconds = msg.nov_header.gps_week_milliseconds / 1000.0
        gps_time = gps_week * 604800 + gps_seconds
        ioniq_gps_times.append(gps_time)

    avente_gps_times = []
    for msg, t in avente_msgs:
        gps_week = msg.nov_header.gps_week_number
        gps_seconds = msg.nov_header.gps_week_milliseconds / 1000.0
        gps_time = gps_week * 604800 + gps_seconds
        avente_gps_times.append(gps_time)

    # ioniq_msgs의 gps_time과 ROS 시간(header.stamp)의 매핑을 만듭니다.
    ioniq_time_map = {}
    for (msg, t), gps_time in zip(ioniq_msgs, ioniq_gps_times):
        ioniq_time_map[gps_time] = t

    # avente_msgs의 gps_time에 대응하는 ROS 시간을 찾습니다.
    adjusted_avente_msgs = []

    print("Adjusting avente messages...")
    for (msg, t), gps_time in zip(avente_msgs, avente_gps_times):
        # 가장 가까운 ioniq의 gps_time을 찾습니다.
        closest_gps_time = min(ioniq_time_map.keys(), key=lambda x: abs(x - gps_time))

        # 해당하는 ROS 시간(header.stamp)을 가져옵니다.
        adjusted_stamp = ioniq_time_map[closest_gps_time]

        # 메시지의 header.stamp를 조정합니다.
        msg.header.stamp = adjusted_stamp

        # 토픽 이름을 변경합니다.
        topic = '/novatel/oem7/inspva2'

        adjusted_avente_msgs.append((topic, msg, adjusted_stamp))

    # 새로운 bag 파일을 생성하고, ioniq_msgs와 adjusted_avente_msgs를 저장합니다.
    print("Writing integration bag...")
    with rosbag.Bag(output_bag_path, 'w') as outbag:
        # ioniq_msgs를 그대로 저장
        with rosbag.Bag(ioniq_bag_path, 'r') as ioniq_bag:
            for topic, msg, t in ioniq_bag.read_messages():
                outbag.write(topic, msg, t)

        # adjusted_avente_msgs를 저장
        for topic, msg, t in adjusted_avente_msgs:
            outbag.write(topic, msg, t)

    print("Integration bag created at:", output_bag_path)

def main():
    # 입력 bag 파일의 경로를 설정합니다.
    ioniq_bag_path = 'songdo_ioniq.bag'
    avente_bag_path = 'songdo_avente.bag'
    output_bag_path = 'integration.bag'

    if not os.path.exists(ioniq_bag_path):
        print("Error: ioniq bag file not found at", ioniq_bag_path)
        return

    if not os.path.exists(avente_bag_path):
        print("Error: avente bag file not found at", avente_bag_path)
        return

    create_integration_bag(ioniq_bag_path, avente_bag_path, output_bag_path)

if __name__ == '__main__':
    main()

