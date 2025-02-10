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
        # 시간 기준 설정 (노드 시작 시점)
        self.start_time = rospy.get_time()

        # ROS 관련
        self.ros_thread = Thread(target=self.ros_loop, daemon=True)
        self.ros_thread.start()

        # 동기화, 데이터 저장용
        self.is_scan = False
        self.filter_scan = None

        # (time, avg_distance)를 저장할 리스트
        self.time_distance_list = []
        self.lock = Lock()

        # Matplotlib 초기화
        plt.ion()  # 인터랙티브 모드 on
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Time (s)')  # 초 단위
        self.ax.set_ylabel('Distance (m)')
        self.ax.set_title('Real-Time Average Distance (-15deg ~ 15deg)')
        self.line, = self.ax.plot([], [], label='Average Distance')
        self.ax.legend()
        plt.show(block=False)

        # 메인 스레드에서 그래프 업데이트 루프 실행
        self.run_plot_loop()

    def ros_loop(self):
        """ROS 관련 loop: /scan 구독 + /filtered_scan 발행"""
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.filter_scan_pub = rospy.Publisher('/filtered_scan', LaserScan, queue_size=1)

        rate = rospy.Rate(30)  # 30 Hz
        while not rospy.is_shutdown():
            if self.is_scan and self.filter_scan is not None:
                angle_min = self.filter_scan.angle_min 
                angle_max = self.filter_scan.angle_max  
                angle_increment = self.filter_scan.angle_increment

                # 관심 각도 범위 설정: -45도 ~ 45도
                min_angle = np.deg2rad(-15)
                max_angle = np.deg2rad(15)

                min_index = int((min_angle - angle_min) / angle_increment)
                max_index = int((max_angle - angle_min) / angle_increment)

                filtered_ranges = list(self.filter_scan.ranges)
                for i in range(len(filtered_ranges)):
                    if not (min_index <= i <= max_index):
                        filtered_ranges[i] = float('inf')

                valid_ranges = [r for r in filtered_ranges if not np.isinf(r)]
                avg_distance = np.mean(valid_ranges) if len(valid_ranges) > 0 else 0

                # 시간(초) 계산 (노드 시작 후 경과 시간)
                current_time = rospy.get_time() - self.start_time

                # 데이터 저장
                with self.lock:
                    self.time_distance_list.append((current_time, avg_distance))

                # 발행할 Scan 수정
                self.filter_scan.ranges = filtered_ranges
                self.filter_scan_pub.publish(self.filter_scan)

            rate.sleep()

    def scan_callback(self, msg):
        """LaserScan 콜백"""
        self.is_scan = True
        self.filter_scan = msg

    def run_plot_loop(self):
        """메인 스레드에서 주기적으로 그래프 업데이트"""
        try:
            while not rospy.is_shutdown():
                current_time = rospy.get_time() - self.start_time
                # 20초 이후에는 그래프 업데이트/수집 종료
                if current_time > 20.0:
                    print("==== 20초가 지났으므로 그래프 업데이트를 멈춥니다. ====")
                    # 최종 데이터 개수 출력
                    with self.lock:
                        data_count = len(self.time_distance_list)
                    print(f"수집된 평균거리 데이터 개수: {data_count} 개")

                    # 그래프 업데이트를 중단하기 위해 break
                    break

                with self.lock:
                    local_data = self.time_distance_list[:]
                if len(local_data) > 0:
                    xdata = [d[0] for d in local_data]
                    ydata = [d[1] for d in local_data]

                    self.line.set_data(xdata, ydata)
                    self.ax.relim()              # 범위 재계산
                    self.ax.autoscale_view()     # 축 자동 스케일
                    self.fig.canvas.draw_idle()
                    self.fig.canvas.flush_events()

                plt.pause(0.2)

            # ========= 20초 후 while 루프를 벗어남 =========
            print("정상 종료합니다.")
            
            # 인터랙티브 모드 off 후, 최종 그래프를 block=True로 표시
            plt.ioff()
            plt.show(block=True)   # 여기서 사용자가 그래프 창을 닫을 때까지 유지

        except KeyboardInterrupt:
            print("KeyboardInterrupt: 종료합니다.")
            sys.exit()

if __name__ == '__main__':
    try:
        rospy.init_node('scan_filter', anonymous=True)
        Scanfilter()
    except rospy.ROSInterruptException:
        pass
