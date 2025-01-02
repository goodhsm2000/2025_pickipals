#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist

def move_turtlebot():
    # 노드 초기화
    rospy.init_node('move_turtlebot', anonymous=True)
    
    # /mobile_base/commands/velocity 토픽에 메시지를 발행할 퍼블리셔 생성
    pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)
    
    # Twist 메시지 인스턴스 생성
    twist = Twist()
    twist.angular.z = 0.2  # 회전 속도 설정 (양수는 시계 방향, 음수는 반시계 방향)
    
    # 메시지 발행 주기 설정 (0.1초)
    rate = rospy.Rate(10)  # 10 Hz
    
    while not rospy.is_shutdown():
        pub.publish(twist)  # Twist 메시지 발행
        #rate.sleep()  # 0.1초 대기

if __name__ == '__main__':
    try:
        move_turtlebot()
    except rospy.ROSInterruptException:
        pass