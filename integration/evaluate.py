#!/usr/bin/env python
import csv
import json
import math
import numpy as np
from shapely.geometry import Polygon

def read_csv_file(filename):
    data = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

def synchronize_data(data1, data2, key='gpstime', tolerance=0.01):
    """
    공통 키(key)를 기준으로 두 데이터 세트를 주어진 오차 범위(tolerance) 내에서 동기화합니다.
    """
    data1_sorted = sorted(data1, key=lambda x: float(x[key]))
    data2_sorted = sorted(data2, key=lambda x: float(x[key]))

    synchronized_data = []

    idx2 = 0
    len2 = len(data2_sorted)

    for item1 in data1_sorted:
        time1 = float(item1[key])

        while idx2 < len2:
            time2 = float(data2_sorted[idx2][key])
            time_diff = time2 - time1
            if abs(time_diff) <= tolerance:
                synchronized_data.append((item1, data2_sorted[idx2]))
                idx2 += 1
                break
            elif time2 > time1 + tolerance:
                # data2에서 더 이상 매칭되는 시간이 없음
                break
            else:
                idx2 += 1

    return synchronized_data

def compute_iou(boxA, boxB):
    """
    두 2D 바운딩 박스의 Intersection over Union (IoU)을 계산합니다.
    각 박스는 [center_x, center_y, size_x, size_y, yaw_angle]로 표현됩니다.
    """
    # 박스의 모서리 좌표 계산
    def get_corners(box):
        cx, cy, size_x, size_y, yaw = box
        cos_yaw = math.cos(math.radians(yaw))
        sin_yaw = math.sin(math.radians(yaw))
        half_size_x = size_x / 2.0
        half_size_y = size_y / 2.0

        corners = []
        for dx, dy in [(-half_size_x, -half_size_y), (-half_size_x, half_size_y),
                       (half_size_x, half_size_y), (half_size_x, -half_size_y)]:
            x = cx + dx * cos_yaw - dy * sin_yaw
            y = cy + dx * sin_yaw + dy * cos_yaw
            corners.append((x, y))
        return corners

    # 폴리곤 생성
    polyA = Polygon(get_corners(boxA))
    polyB = Polygon(get_corners(boxB))

    if not polyA.is_valid or not polyB.is_valid:
        return 0.0

    intersection = polyA.intersection(polyB).area
    union = polyA.union(polyB).area

    if union == 0:
        return 0.0

    iou = intersection / union
    return iou

def evaluate(ioniq_data, avente_data, bounding_boxes_data, avente_specs):
    # avente_specs: {'length': float, 'width': float}

    # gpstime을 기준으로 ioniq와 avente 데이터 동기화
    synchronized_data = synchronize_data(ioniq_data, avente_data, key='gpstime', tolerance=0.01)

    evaluation_results = []

    for (ioniq_item, avente_item) in synchronized_data:
        gpstime = float(ioniq_item['gpstime'])
        rostime = float(ioniq_item['rostime'])

        # ioniq 차량의 위치 및 방향
        ioniq_x = float(ioniq_item['world_x'])
        ioniq_y = float(ioniq_item['world_y'])
        ioniq_azimuth = float(ioniq_item['azimuth'])
        ioniq_vx = float(ioniq_item['vx'])
        ioniq_vy = float(ioniq_item['vy'])

        # avente 차량의 위치 및 방향
        avente_x = float(avente_item['world_x'])
        avente_y = float(avente_item['world_y'])
        avente_azimuth = float(avente_item['azimuth'])
        avente_vx = float(avente_item['vx'])
        avente_vy = float(avente_item['vy'])

        # avente 차량의 속도 계산
        avente_speed = math.sqrt(avente_vx**2 + avente_vy**2)

        # ioniq 차량의 속도 계산
        ioniq_speed = math.sqrt(ioniq_vx**2 + ioniq_vy**2)

        # 상대 속도 계산 (avente 속도 벡터 - ioniq 속도 벡터)
        relative_vx = avente_vx - ioniq_vx
        relative_vy = avente_vy - ioniq_vy
        relative_speed = math.sqrt(relative_vx**2 + relative_vy**2)

        # avente 차량의 ioniq 좌표계에서의 상대 위치 계산
        dx = avente_x - ioniq_x
        dy = avente_y - ioniq_y

        distance = math.sqrt(dx**2 + dy**2)  # 두 차량 사이의 거리 계산

        theta = math.radians(ioniq_azimuth)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        rel_x = dx * cos_theta + dy * sin_theta
        rel_y = -dx * sin_theta + dy * cos_theta

        # avente 차량의 ioniq 좌표계에서의 방향
        rel_yaw = (avente_azimuth - ioniq_azimuth + 360) % 360

        # avente 차량의 Ground Truth 바운딩 박스 생성
        gt_bbox = [rel_x, rel_y, avente_specs['length'], avente_specs['width'], rel_yaw]

        # 해당 rostime에 대한 바운딩 박스 찾기
        detected_bboxes = []
        for bbox_item in bounding_boxes_data:
            bbox_rostime = float(bbox_item['rostime'])
            if abs(bbox_rostime - rostime) <= 0.01:  # 오차 범위 내에 있으면
                bounding_boxes_json = bbox_item['bounding_boxes']
                bboxes = json.loads(bounding_boxes_json)
                for bbox in bboxes:
                    center_x = bbox['center_x']
                    center_y = bbox['center_y']
                    size_x = bbox['size_x']
                    size_y = bbox['size_y']
                    yaw_angle = bbox['yaw_angle']
                    value = bbox['value']  # 상대 속도 값
                    detected_bboxes.append({
                        'center_x': center_x,
                        'center_y': center_y,
                        'size_x': size_x,
                        'size_y': size_y,
                        'yaw_angle': yaw_angle,
                        'value': value
                    })
                break  # 일치하는 타임스탬프에 대해서만 처리

        # 검출된 바운딩 박스 중 중심 거리가 2m 이내인 것만 선택
        filtered_bboxes = []
        for det_bbox in detected_bboxes:
            det_center_x = det_bbox['center_x']
            det_center_y = det_bbox['center_y']
            position_error = math.sqrt((det_center_x - gt_bbox[0])**2 + (det_center_y - gt_bbox[1])**2)
            if position_error <= 2.0:
                filtered_bboxes.append((det_bbox, position_error))

        # 필터링된 바운딩 박스들에 대해 오차 계산
        if filtered_bboxes:
            for det_bbox, position_error in filtered_bboxes:
                # IOU 계산
                det_bbox_list = [det_bbox['center_x'], det_bbox['center_y'], det_bbox['size_x'], det_bbox['size_y'], det_bbox['yaw_angle']]
                iou = compute_iou(gt_bbox, det_bbox_list)

                # 헤딩 오차 계산
                heading_error = abs(det_bbox['yaw_angle'] - gt_bbox[4])
                heading_error = min(heading_error, 360 - heading_error)  # 최소 각도 차이로 계산

                # 속도 오차 계산
                lidar_relative_speed = det_bbox['value']  # 라이다로 측정한 상대 속도
                speed_error = abs(lidar_relative_speed - relative_speed)

                # 평가 결과 저장
                result = {
                    'gpstime': gpstime,
                    'rostime': rostime,
                    'distance': distance,
                    'position_error': position_error,
                    'iou': iou,
                    'heading_error': heading_error,
                    'speed_error': speed_error,
                    'gt_bbox': gt_bbox,
                    'detected_bbox': det_bbox
                }

                evaluation_results.append(result)
        else:
            # 매칭되는 바운딩 박스가 없는 경우
            result = {
                'gpstime': gpstime,
                'rostime': rostime,
                'distance': distance,
                'position_error': None,
                'iou': None,
                'heading_error': None,
                'speed_error': None,
                'gt_bbox': gt_bbox,
                'detected_bbox': None
            }
            evaluation_results.append(result)

    # 각 지표별로 값을 수집
    position_errors = [res['position_error'] for res in evaluation_results if res['position_error'] is not None]
    ious = [res['iou'] for res in evaluation_results if res['iou'] is not None]
    heading_errors = [res['heading_error'] for res in evaluation_results if res['heading_error'] is not None]
    speed_errors = [res['speed_error'] for res in evaluation_results if res['speed_error'] is not None]

    # 평균, 최대, 최소값 계산
    def calculate_stats(values):
        if values:
            avg = sum(values) / len(values)
            max_val = max(values)
            min_val = min(values)
            return avg, max_val, min_val
        else:
            return None, None, None

    position_error_stats = calculate_stats(position_errors)
    iou_stats = calculate_stats(ious)
    heading_error_stats = calculate_stats(heading_errors)
    speed_error_stats = calculate_stats(speed_errors)

    # 결과 출력
    print("\nEvaluation Results:")
    print(f"Position Error - Avg: {position_error_stats[0]:.2f}, Max: {position_error_stats[1]:.2f}, Min: {position_error_stats[2]:.2f}")
    print(f"IOU - Avg: {iou_stats[0]:.2f}, Max: {iou_stats[1]:.2f}, Min: {iou_stats[2]:.2f}")
    print(f"Heading Error - Avg: {heading_error_stats[0]:.2f}, Max: {heading_error_stats[1]:.2f}, Min: {heading_error_stats[2]:.2f}")
    print(f"Speed Error - Avg: {speed_error_stats[0]:.2f}, Max: {speed_error_stats[1]:.2f}, Min: {speed_error_stats[2]:.2f}")

    return evaluation_results

def main():
    # CSV 파일 읽기
    ioniq_data = read_csv_file('ioniq.csv')
    avente_data = read_csv_file('avente.csv')
    bounding_boxes_data = read_csv_file('bounding_boxes.csv')

    # avente 차량의 스펙 (미터 단위)
    avente_specs = {
        'length': 4.650,  # 전장 (미터)
        'width': 1.825    # 전폭 (미터)
    }

    evaluation_results = evaluate(ioniq_data, avente_data, bounding_boxes_data, avente_specs)

    # 평가 결과를 CSV 파일로 저장
    with open('evaluation_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['gpstime', 'rostime', 'distance', 'position_error', 'iou', 'heading_error', 'speed_error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for res in evaluation_results:
            writer.writerow({
                'gpstime': res['gpstime'],
                'rostime': res['rostime'],
                'distance': res['distance'],
                'position_error': res['position_error'],
                'iou': res['iou'],
                'heading_error': res['heading_error'],
                'speed_error': res['speed_error']
            })

if __name__ == '__main__':
    main()
