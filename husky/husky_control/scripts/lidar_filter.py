#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy 
from sensor_msgs.msg import LaserScan
import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from threading import Lock, Thread
import math
import sys

class Scanfilter:
    def __init__(self):
        # ROS 관련 설정은 이미 메인 스레드에서 초기화했으므로,
        # 여기서는 ROS 관련 작업(구독, 발행 등)만 스레드로 실행합니다.
        self.ros_thread = Thread(target=self.ros_loop, daemon=True)
        self.ros_thread.start()

        # 데이터 보호 및 저장용 변수
        self.is_scan = False
        self.filter_scan = None
        self.average_distances = []
        self.lock = Lock()

        # Matplotlib 초기화 (메인 스레드)
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Time Step')
        self.ax.set_ylabel('Distance (m)')
        self.ax.set_title('Real-Time Average Distance')
        self.line, = self.ax.plot([], [], label='Average Distance')
        self.ax.legend()

        plt.show(block=False)

        # 메인 스레드에서 무한 플롯 업데이트 루프 실행
        self.run_plot_loop()

    def ros_loop(self):
        # ROS 관련 작업은 이미 메인 스레드에서 init_node() 호출 후 이곳에서 실행됩니다.
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.filter_scan_pub = rospy.Publisher('/filtered_scan', LaserScan, queue_size=1)

        rate = rospy.Rate(30)  # 30 Hz
        while not rospy.is_shutdown():
            if self.is_scan and self.filter_scan is not None:
                # LaserScan 메시지에서 각도 정보 추출
                angle_min = self.filter_scan.angle_min 
                angle_max = self.filter_scan.angle_max  
                angle_increment = self.filter_scan.angle_increment

                # 관심 각도 범위 (-45도 ~ 45도)
                min_angle = np.deg2rad(-45)
                max_angle = np.deg2rad(45)

                # 해당 각도 범위 인덱스 계산
                min_index = int((min_angle - angle_min) / angle_increment)
                max_index = int((max_angle - angle_min) / angle_increment)
                filtered_ranges = list(self.filter_scan.ranges)

                # 관심 범위 외의 값은 inf로 설정
                for i, range_value in enumerate(filtered_ranges):
                    if not (min_index <= i <= max_index):
                        filtered_ranges[i] = float('inf')

                # 유효한 값들에 대해 평균 계산
                valid_ranges = [r for r in filtered_ranges if not np.isinf(r)]
                avg_distance = np.mean(valid_ranges) if valid_ranges else 0
                print("Average distance:", avg_distance)

                # 안전하게 데이터 추가
                with self.lock:
                    self.average_distances.append(avg_distance)

                # LaserScan 메시지 수정 후 발행
                self.filter_scan.ranges = filtered_ranges                
                self.filter_scan_pub.publish(self.filter_scan)

            rate.sleep()

    def scan_callback(self, msg):
        self.is_scan = True
        self.filter_scan = msg

    def run_plot_loop(self):
        try:
            while not rospy.is_shutdown():
                with self.lock:
                    # 최근 100 데이터만 사용 (너무 많은 데이터 누적 방지)
                    ydata = list(self.average_distances)
                xdata = list(range(len(ydata)))

                self.line.set_data(xdata, ydata)
                self.ax.relim()              # 데이터 범위 재계산
                self.ax.autoscale_view()     # 축 자동 조정

                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()

                plt.pause(0.2)
        except KeyboardInterrupt:
            print("종료합니다.")
            sys.exit()

if __name__ == '__main__':
    try:
        # 메인 스레드에서 rospy.init_node() 호출
        rospy.init_node('scan_filter', anonymous=True)
        Scanfilter()
    except rospy.ROSInterruptException:
        pass
