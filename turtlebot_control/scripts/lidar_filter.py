#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy 
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point32
import numpy as np 
import matplotlib.pyplot as plt
from threading import Thread
import math

class Scanfilter:
    def __init__(self):
        rospy.init_node('scan_filter', anonymous=True)

        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.filter_scan_pub = rospy.Publisher('/filtered_scan', LaserScan, queue_size=1)

        self.is_scan = False
        self.average_distances = []
        self.graph_thread = Thread(target=self.plot_graph)
        self.graph_thread.daemon = True
        self.graph_thread.start()
        
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():

            if self.is_scan:

                # 각도 범위
                angle_min = self.filter_scan.angle_min 
                angle_max = self.filter_scan.angle_max  
                angle_increment = self.filter_scan.angle_increment
                
                min_angle = np.deg2rad(-45)
                max_angle = np.deg2rad(45)

                # 인덱스
                min_index = int((min_angle - angle_min) / angle_increment)
                max_index = int((max_angle - angle_min) / angle_increment)
                # print(min_index, max_index)
                filtered_ranges = list(self.filter_scan.ranges)
                
                for i, range_value in enumerate(filtered_ranges):
                    # if i > min_index and i < max_index:
                    if min_index <= i <= max_index:
                        continue
                    else:
                        filtered_ranges[i] = float('inf')

                valid_ranges = [r for r in filtered_ranges if not np.isinf(r)]
                avg_distance = np.mean(valid_ranges) if valid_ranges else 0
                self.average_distances.append(avg_distance)
                
                self.filter_scan.ranges = filtered_ranges
                
                self.filter_scan_pub.publish(self.filter_scan)

            rate.sleep()

    def scan_callback(self, msg):
        self.is_scan = True
        self.filter_scan = msg

    def plot_graph(self):
        plt.ion()
        fig, ax = plt.subplots()
        while not rospy.is_shutdown():
            ax.clear()
            ax.plot(self.average_distances, label='Average Distance')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Distance (m)')
            ax.set_title('Real-Time Average Distance')
            ax.legend()
            plt.pause(0.1)

if __name__ == '__main__':
    try:
        Scanfilter()
    except rospy.ROSInterruptException:
        pass