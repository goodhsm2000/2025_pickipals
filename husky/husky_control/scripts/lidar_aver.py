#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import LaserScan
import numpy as np

class Initial100Averager:
    def __init__(self):
        # 저장용
        self.avg_list = []
        self.got_100 = False

        rospy.Subscriber('/scan', LaserScan, self.scan_callback)

    def scan_callback(self, msg):
        if self.got_100:
            return  # 이미 100개 수집 완료 시, 더 이상 처리 X

        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_increment = msg.angle_increment

        # 관심 각도 범위 -45 ~ 45
        min_angle = np.deg2rad(-15)
        max_angle = np.deg2rad(15)

        min_index = int((min_angle - angle_min) / angle_increment)
        max_index = int((max_angle - angle_min) / angle_increment)

        filtered_ranges = list(msg.ranges)
        for i in range(len(filtered_ranges)):
            if not (min_index <= i <= max_index):
                filtered_ranges[i] = float('inf')

        valid_ranges = [r for r in filtered_ranges if not np.isinf(r)]
        avg_distance = np.mean(valid_ranges) if len(valid_ranges) > 0 else 0

        self.avg_list.append(avg_distance)
        rospy.loginfo(f"{len(self.avg_list)}번째 평균 거리: {avg_distance:.3f} m")

        if len(self.avg_list) >= 100:
            self.got_100 = True
            # 100개 평균 계산
            final_avg = np.mean(self.avg_list)
            rospy.loginfo(f"==== 초기 100개의 평균 라이다 값 평균: {final_avg:.3f} m ====")

            # 노드 종료
            rospy.signal_shutdown("100개 평균 계산 완료")

if __name__ == '__main__':
    rospy.init_node("initial_100_averager", anonymous=True)
    node = Initial100Averager()
    rospy.spin()
