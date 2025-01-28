#! /usr/bin/env python
# -*- coding:utf-8 -*-
   
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from sklearn.cluster import DBSCAN

def lidar_callback(scan):
    # LaserScan 데이터를 numpy 배열로 변환
    ranges = np.array(scan.ranges)
    angles = np.linspace(scan.angle_min, scan.angle_max, len(ranges))

    # 유효 데이터 필터링 (무한대 값 및 최대 범위 초과 제거)
    valid_indices = ~np.isinf(ranges) & (ranges < scan.range_max)
    ranges = ranges[valid_indices]
    angles = angles[valid_indices]

    # (x, y) 좌표로 변환
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    points = np.stack((x, y), axis=-1)

    # DBSCAN 클러스터링
    clustering = DBSCAN(eps=0.15, min_samples=6).fit(points)
    labels = clustering.labels_

    # 유효 클러스터 포인트 필터링 (노이즈 제외)
    valid_points = points[labels != -1]

    # 새로운 LaserScan 메시지 생성
    filtered_scan = LaserScan()
    filtered_scan.header = scan.header
    filtered_scan.angle_min = scan.angle_min
    filtered_scan.angle_max = scan.angle_max
    filtered_scan.angle_increment = scan.angle_increment
    filtered_scan.time_increment = scan.time_increment
    filtered_scan.scan_time = scan.scan_time
    filtered_scan.range_min = scan.range_min
    filtered_scan.range_max = scan.range_max
    filtered_scan.ranges = [float('inf')] * len(scan.ranges)  # 초기화
    filtered_scan.intensities = scan.intensities
    
    # 유효 포인트를 LaserScan으로 매핑
    for point in valid_points:
        # (x, y)에서 각도 및 거리 계산
        angle = np.arctan2(point[1], point[0])
        distance = np.linalg.norm(point)

        # 각도 기반으로 인덱스 계산
        index = int((angle - scan.angle_min) / scan.angle_increment)
        if 0 <= index < len(filtered_scan.ranges):
            filtered_scan.ranges[index] = distance

    # 결과 전송
    pub.publish(filtered_scan)

if __name__ == "__main__":
    rospy.init_node("lidar_filter_node")

    # Subscriber와 Publisher 설정
    sub = rospy.Subscriber("/scan", LaserScan, lidar_callback)
    pub = rospy.Publisher("/cluster_scan", LaserScan, queue_size=10)

    rospy.spin()
