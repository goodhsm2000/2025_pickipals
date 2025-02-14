#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import tf
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

class YawController:
    def __init__(self):
        rospy.init_node("yaw_controller", anonymous=True)

        # ★ 목표 각도(도 단위) 파라미터. 예: 90도 (정면 기준 좌회전 90도)
        self.target_yaw_deg = rospy.get_param("~target_yaw_deg", 45.0)
        # 내부 계산은 라디안
        self.target_yaw_rad = math.radians(self.target_yaw_deg)

        # ★ 제어 게인 (P, I)
        self.Kp = rospy.get_param("~Kp", 0.8)
        self.Ki = rospy.get_param("~Ki", 0)  # 기본값 0 → P 제어만 사용

        # 현재 로봇 각도(yaw) 저장 변수
        self.yaw_current = 0.0

        # 적분항
        self.integral_error = 0.0
        self.last_time = rospy.Time.now()

        # cmd_vel 퍼블리셔
        self.cmd_vel_pub = rospy.Publisher("/husky_velocity_controller/cmd_vel",
                                           Twist, queue_size=1)

        # 현재 로봇의 자세를 알기 위해 /odom 구독
        rospy.Subscriber("/odometry/filtered", Odometry, self.odom_callback)

        # 루프 주기 설정 (예: 10Hz)
        self.rate = rospy.Rate(10)

    def odom_callback(self, msg):
        """
        /odom의 orientation(쿼터니언) → euler 변환 후 yaw(회전각)만 추출
        """
        orientation_q = msg.pose.pose.orientation
        # tf.transformations.euler_from_quaternion 사용
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
            [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        )
        self.yaw_current = yaw  # 라디안
        print("현재 각도: ")
        print(math.degrees(self.yaw_current))
        

    def run(self):
        """
        루프 돌며 목표각도 - 현재각도 오차를 계산해서 회전 명령(P/PI 제어) 발행
        """
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = (now - self.last_time).to_sec()
            self.last_time = now

            # yaw 오차 (목표 - 현재)
            error = self.normalize_angle(self.target_yaw_rad - self.yaw_current)

            # 적분 항 업데이트 (Ki != 0일 때만 효과 있음)
            self.integral_error += error * dt

            # 제어(토크) = P + I
            control = self.Kp * error + self.Ki * self.integral_error

            # 회전 명령어 구성
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = control  # 제어 출력

            # 발행
            self.cmd_vel_pub.publish(twist)

            self.rate.sleep()

    def normalize_angle(self, angle):
        """각도를 -pi ~ +pi 범위로 정규화"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle <= -math.pi:
            angle += 2.0 * math.pi
        return angle

if __name__ == "__main__":
    node = YawController()
    node.run()
