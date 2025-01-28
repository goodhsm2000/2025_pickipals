#! /usr/bin/env python
# -*- coding:utf-8 -*-
import rospy
from sensor_msgs.msg import LaserScan

def callback(scan_msg):
    # 거리 배열과 intensity 배열 가져오기
    ranges = scan_msg.ranges
    intensities = scan_msg.intensities

    if not ranges or not intensities:
        rospy.logwarn("Received empty ranges or intensities array.")
        return

    # 최소 거리와 해당 조건을 만족하는 index 찾기
    valid_distances = [(distance, idx) for idx, distance in enumerate(ranges) if distance > 0.25]
    if not valid_distances:
        rospy.logwarn("No distances greater than 0.25 meters found.")
        return

    min_distance, min_index = min(valid_distances, key=lambda x: x[0])
    # 해당 index의 intensity 값 가져오기
    corresponding_intensity = intensities[min_index]

    # f-string 대신 format 사용
    rospy.loginfo("Min Distance: {:.2f} m, Intensity: {:.2f}".format(min_distance, corresponding_intensity))

def listener():
    rospy.init_node('scan_intensity_listener', anonymous=True)
    rospy.Subscriber('/scan', LaserScan, callback)
    rospy.loginfo("Subscribed to /scan topic.")
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass

