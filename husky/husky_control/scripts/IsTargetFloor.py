#! /usr/bin/env python
# -*- coding:utf-8 -*-

import rospy 
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point32
import numpy as np 

class IsTargetFloor(object):
    def __init__(self):
        rospy.Subscriber('/cluster_scan', LaserScan, self.scan_callback)
        self.is_scan = False
        self.distance_list = []

    def __call__(self, min_ang, max_ang, threshold):
        self.distance_list = []
        if self.is_scan:
            angle_min = self.filter_scan.angle_min 
            angle_max = self.filter_scan.angle_max  
            angle_increment = self.filter_scan.angle_increment
            
            min_angle = np.deg2rad(min_ang)
            max_angle = np.deg2rad(max_ang)

            # 인덱스
            min_index = int((min_angle - angle_min) / angle_increment)
            max_index = int((max_angle - angle_min) / angle_increment)
            # print(min_index, max_index)
            
            for i, range_value in enumerate(self.filter_scan.ranges):
                # if i > min_index and i < max_index:
                if min_index <= i <= max_index:
                    if not np.isinf(range_value): 
                        self.distance_list.append(range_value)

            average_distance = np.mean(self.distance_list)
            print("distance: ", average_distance)

            if average_distance > threshold:
                print("Door is open")
                return True
            else:
                print("Door is closed")
                return False
            

    def scan_callback(self, msg):
        self.is_scan = True
        self.filter_scan = msg
