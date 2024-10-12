import rospy
from ublox_msgs.msg import NavPVT
from jsk_recognition_msgs.msg import BoundingBoxArray
from visualization_msgs.msg import Marker, MarkerArray

class BoundingBoxTextMarkerPublisher:
    def __init__(self):
        rospy.init_node('bbox_text_marker_publisher', anonymous=True)

        # Publishers
        self.marker_pub = rospy.Publisher('/absolute_speed', MarkerArray, queue_size=10)

        # Subscribers
        rospy.Subscriber("/ublox/navpvt", NavPVT, self.gspeed_callback)
        rospy.Subscriber("/mobinha/perception/lidar/track_box", BoundingBoxArray, self.bbox_callback)

        self.gspeed = 0

    def gspeed_callback(self, navpvt_msg):
        # gSpeed 값을 받아와서 처리 (m/s 단위로 변환)
        self.gspeed = (navpvt_msg.gSpeed / 1000.0) * 3.6  # km/h로 변환

    def bbox_callback(self, bbox_array_msg):
        marker_array = MarkerArray()  # MarkerArray 생성

        for idx, bbox in enumerate(bbox_array_msg.boxes):
            # 텍스트 마커 생성
            marker = Marker()
            marker.lifetime = rospy.Duration(0.05)
            marker.header.frame_id = bbox.header.frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = "bounding_box_text"
            marker.id = idx  # 각 텍스트 마커에 고유 ID 할당
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD

            # 텍스트 마커의 위치 (바운딩 박스 상단에 표시)
            marker.pose.position.x = bbox.pose.position.x
            marker.pose.position.y = bbox.pose.position.y
            marker.pose.position.z = bbox.pose.position.z + bbox.dimensions.z / 2 + 0.5  # 바운딩 박스 위에 표시

            # 텍스트 마커의 텍스트 내용 (gSpeed 값 추가)
            marker.text = "{:.2f} km/h".format(self.gspeed + bbox.value)

            # 텍스트 크기 설정
            marker.scale.z = 1.5  # 텍스트 크기 설정 (Z축 스케일로 조정)

            # 텍스트 색상 설정
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0  # 불투명하게 설정

            # MarkerArray에 텍스트 마커 추가
            marker_array.markers.append(marker)

        # MarkerArray 퍼블리시
        self.marker_pub.publish(marker_array)

if __name__ == '__main__':
    try:
        bbox_text_marker_publisher = BoundingBoxTextMarkerPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
