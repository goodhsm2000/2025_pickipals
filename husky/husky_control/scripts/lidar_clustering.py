#!/usr/bin/env python
# -*- coding:utf-8 -*-

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from sklearn.cluster import DBSCAN

# 전방 각도(도 단위)
FRONT_ANGLE_DEG = 150.0

def lidar_callback(scan):
    # 원본 LaserScan에서 각도/거리 배열 생성
    ranges = np.array(scan.ranges)
    angles = np.linspace(scan.angle_min, scan.angle_max, len(ranges))

    # 유효 데이터 필터링 (무한대 값 및 최대 범위 초과 제거)
    valid_indices = (~np.isinf(ranges)) & (ranges < scan.range_max)
    ranges = ranges[valid_indices]
    angles = angles[valid_indices]

    # -----------------------------
    # 1) 전방 160도(±80도) 필터링
    # -----------------------------
    half_front_rad = np.deg2rad(FRONT_ANGLE_DEG / 2.0)  # 80도 -> 라디안
    front_indices = (angles >= -half_front_rad) & (angles <= half_front_rad)

    ranges = ranges[front_indices]
    angles = angles[front_indices]

    # 만약 전방 160도에 데이터가 거의 없다면 그냥 반환할 수도 있음
    if len(ranges) == 0:
        empty_scan = LaserScan()
        empty_scan.header = scan.header
        # 전방 160도에 해당하는 범위만 세팅
        empty_scan.angle_min = -half_front_rad
        empty_scan.angle_max = half_front_rad
        empty_scan.angle_increment = scan.angle_increment
        empty_scan.time_increment = scan.time_increment
        empty_scan.scan_time = scan.scan_time
        empty_scan.range_min = scan.range_min
        empty_scan.range_max = scan.range_max

        # front_angle 범위에 맞게 points 개수 계산
        angle_range = empty_scan.angle_max - empty_scan.angle_min
        num_points = int(round(angle_range / empty_scan.angle_increment)) + 1
        empty_scan.ranges = [float('inf')] * num_points
        empty_scan.intensities = [0.0] * num_points

        pub.publish(empty_scan)
        return

    # (x, y) 좌표로 변환
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    points = np.stack((x, y), axis=-1)

    # -----------------------------
    # 2) DBSCAN 클러스터링
    # -----------------------------
    clustering = DBSCAN(eps=0.15, min_samples=6).fit(points)
    labels = clustering.labels_

    # 유효 클러스터(-1: 노이즈 제외) 포인트만 추출
    valid_points = points[labels != -1]

    # -----------------------------
    # 3) 전방 160도 범위로 LaserScan 생성
    # -----------------------------
    filtered_scan = LaserScan()
    filtered_scan.header = scan.header

    # 전방 160도에 해당하는 min/max를 설정
    filtered_scan.angle_min = -half_front_rad
    filtered_scan.angle_max = half_front_rad
    filtered_scan.angle_increment = scan.angle_increment
    filtered_scan.time_increment = scan.time_increment
    filtered_scan.scan_time = scan.scan_time
    filtered_scan.range_min = scan.range_min
    filtered_scan.range_max = scan.range_max

    # front_angle 범위에 해당하는 총 스캔 포인트 수 계산
    angle_range = filtered_scan.angle_max - filtered_scan.angle_min
    num_points = int(round(angle_range / filtered_scan.angle_increment)) + 1

    # 초기화
    filtered_scan.ranges = [float('inf')] * num_points
    filtered_scan.intensities = [0.0] * num_points

    # 유효 포인트를 새 LaserScan에 매핑
    for point in valid_points:
        angle = np.arctan2(point[1], point[0])
        distance = np.linalg.norm(point)

        # 혹시라도 angle이 -80도~+80도 범위를 벗어났다면 무시
        if angle < filtered_scan.angle_min or angle > filtered_scan.angle_max:
            continue

        index = int((angle - filtered_scan.angle_min) / filtered_scan.angle_increment)
        # 인덱스 범위 체크
        if 0 <= index < num_points:
            # 더 가까운 거리만 업데이트 (DBSCAN 후 같은 인덱스에 여러 점이 겹칠 수도 있으므로)
            if distance < filtered_scan.ranges[index]:
                filtered_scan.ranges[index] = distance

    # 퍼블리시
    pub.publish(filtered_scan)

if __name__ == "__main__":
    rospy.init_node("lidar_filter_node")

    # Subscriber와 Publisher 설정
    sub = rospy.Subscriber("/scan", LaserScan, lidar_callback)
    pub = rospy.Publisher("/cluster_scan", LaserScan, queue_size=10)

    rospy.spin()
