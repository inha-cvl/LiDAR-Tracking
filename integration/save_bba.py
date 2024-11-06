#!/usr/bin/env python
import rospy
import csv
import math
import json
from jsk_recognition_msgs.msg import BoundingBoxArray
from tf.transformations import euler_from_quaternion

class BoundingBoxArraySaver:
    def __init__(self):
        rospy.init_node('bounding_box_array_saver')

        # CSV 파일 열기
        self.csv_file = open('bounding_boxes.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        # CSV 헤더 작성
        self.csv_writer.writerow(['rostime', 'bounding_boxes'])

        # /track_box 토픽 구독
        rospy.Subscriber('/deep_box', BoundingBoxArray, self.callback)

        rospy.on_shutdown(self.shutdown_hook)
        rospy.loginfo("BoundingBoxArraySaver initialized and listening to /track_box")

    def shutdown_hook(self):
        self.csv_file.close()
        rospy.loginfo("CSV file closed.")

    def callback(self, msg):
        rostime = msg.header.stamp.to_sec()

        bounding_boxes_data = []

        for bbox in msg.boxes:
            # 바운딩 박스 정보 추출
            center_x = bbox.pose.position.x
            center_y = bbox.pose.position.y
            center_z = bbox.pose.position.z

            size_x = bbox.dimensions.x
            size_y = bbox.dimensions.y
            size_z = bbox.dimensions.z

            # Quaternion -> Yaw 각도 변환
            orientation_q = bbox.pose.orientation
            quaternion = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            _, _, yaw_angle_rad = euler_from_quaternion(quaternion)
            yaw_angle_deg = math.degrees(yaw_angle_rad)

            # value 필드 추출 (상대 속도)
            value = bbox.value

            bbox_data = {
                'center_x': center_x,
                'center_y': center_y,
                'center_z': center_z,
                'size_x': size_x,
                'size_y': size_y,
                'size_z': size_z,
                'yaw_angle': yaw_angle_deg,
                'value': value
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
