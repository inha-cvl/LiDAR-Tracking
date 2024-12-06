#!/usr/bin/env python
import rospy
import csv
import math
import json
from jsk_recognition_msgs.msg import BoundingBoxArray
from tf.transformations import euler_from_quaternion

# bba_topic = '/cloud_segmentation/cluster_box'
bba_topic = '/deep_box'
#bba_topic = '/mobinha/perception/lidar/track_box'

class BoundingBoxArraySaver:
    def __init__(self):
        rospy.init_node('bounding_box_array_saver')

        # CSV 파일 열기
        self.csv_file = open('bounding_boxes.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        # CSV 헤더 작성
        self.csv_writer.writerow(['rostime', 'bounding_boxes'])

        # /track_box 토픽 구독
        rospy.Subscriber(bba_topic, BoundingBoxArray, self.callback)

        rospy.on_shutdown(self.shutdown_hook)
        rospy.loginfo("BoundingBoxArraySaver initialized and listening to " + bba_topic)

    def shutdown_hook(self):
        self.csv_file.close()
        rospy.loginfo("CSV file closed.")

    def callback(self, msg):
        rostime = msg.header.stamp.to_sec()

        bounding_boxes_data = []

        for bbox in msg.boxes:
            # Quaternion -> Yaw 각도 변환
            orientation_q = bbox.pose.orientation
            quaternion = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            _, _, yaw_angle_rad = euler_from_quaternion(quaternion)
            yaw_angle_deg = math.degrees(yaw_angle_rad)

            bbox_data = {
                'center_x': bbox.pose.position.x,
                'center_y': bbox.pose.position.y,
                'center_z': bbox.pose.position.z,
                'size_x': bbox.dimensions.x,
                'size_y': bbox.dimensions.y,
                'size_z': bbox.dimensions.z,
                'yaw_angle': yaw_angle_deg,
                'age': bbox.header.seq,
                'value': bbox.value
            }

            bounding_boxes_data.append(bbox_data)

        # 바운딩 박스 데이터를 JSON 문자열로 직렬화
        bounding_boxes_json = json.dumps(bounding_boxes_data)

        # CSV 파일에 기록
        self.csv_writer.writerow([rostime, bounding_boxes_json])

def main():
    saver = BoundingBoxArraySaver()
    rospy.spin()

if __name__ == '__main__':
    main()
