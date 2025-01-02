#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class MoveToLaserDistance:
    def __init__(self):
        rospy.init_node('move_to_laser_distance', anonymous=True)
        
        # /cmd_vel 퍼블리셔 생성
        self.velocity_publisher = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)
        
        # /scan 구독자 생성
        self.scan_subscriber = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        
        # 전진 명령을 위해 Twist 메시지 초기화
        self.vel_msg = Twist()
        
        # 이동 플래그 설정
        self.moving = False

    def scan_callback(self, msg):
        # 250번째 요소의 거리 값을 가져옴
        distance = msg.ranges[250]-0.1

        # 거리 값 확인 (예: 최소 거리 및 최대 거리 체크)
        if distance > 0.2 and distance < msg.range_max:
            rospy.loginfo(f"Moving forward by {distance} meters")

            # 선형 속도를 설정 (0.2 m/s)
            self.vel_msg.linear.x = 0.2
            self.vel_msg.angular.z = 0.0

            # 이동 시간 계산 (거리 / 속도)
            time_to_move = distance / self.vel_msg.linear.x
            
            # 현재 시간 기록
            start_time = rospy.Time.now().to_sec()
            
            # 이동 플래그 설정
            self.moving = True

            # 주어진 시간 동안 전진
            while rospy.Time.now().to_sec() - start_time < time_to_move:
                self.velocity_publisher.publish(self.vel_msg)
                rospy.sleep(0.1)
            
            # 정지 명령
            self.vel_msg.linear.x = 0.0
            self.velocity_publisher.publish(self.vel_msg)
            rospy.loginfo("TurtleBot has stopped")
            
            # 한 번만 이동하도록 구독자 종료
            self.scan_subscriber.unregister()
        
        else:
            rospy.logwarn(f"Invalid distance: {distance} meters")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        move_to_laser_distance = MoveToLaserDistance()
        move_to_laser_distance.run()
    except rospy.ROSInterruptException:
        pass
