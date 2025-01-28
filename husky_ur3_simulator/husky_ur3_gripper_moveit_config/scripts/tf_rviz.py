#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import tf2_ros
import tf.transformations as t
from geometry_msgs.msg import PoseStamped, TransformStamped

class DetectedObjectTFBroadcaster:
    def __init__(self):
        # 노드 초기화
        rospy.init_node('tf_rviz_broadcaster', anonymous=True)

        # TF 브로드캐스터 생성
        self.br = tf2_ros.TransformBroadcaster()

        # PoseStamped 구독 (토픽 이름은 기존 코드와 맞추어 수정)
        self.pose_sub = rospy.Subscriber(
            "/detected_object_pose",  # ← 기존 코드에서 퍼블리시하는 토픽명
            PoseStamped,
            self.pose_callback
        )

    def pose_callback(self, msg: PoseStamped):
        """
        /detected_object_pose로부터 받은 PoseStamped를 TF로 퍼블리시.
        즉, parent_frame(기본: odom) → child_frame(기본: detected_object).
        """
        # Transform 메시지 생성
        tf_msg = TransformStamped()

        # header 설정
        tf_msg.header.stamp = rospy.Time.now()
        tf_msg.header.frame_id = msg.header.frame_id  # 보통 "odom"
        tf_msg.child_frame_id = "detected_object"     # TF에서 보이게 될 이름

        # 위치(translation)
        tf_msg.transform.translation.x = msg.pose.position.x
        tf_msg.transform.translation.y = msg.pose.position.y
        tf_msg.transform.translation.z = msg.pose.position.z

        # 회전(orientation)
        tf_msg.transform.rotation.x = msg.pose.orientation.x
        tf_msg.transform.rotation.y = msg.pose.orientation.y
        tf_msg.transform.rotation.z = msg.pose.orientation.z
        tf_msg.transform.rotation.w = msg.pose.orientation.w

        # 브로드캐스트
        self.br.sendTransform(tf_msg)

    def spin(self):
        rospy.loginfo("tf_rviz_broadcaster node started. Waiting for /detected_object_pose...")
        rospy.spin()

if __name__ == "__main__":
    node = DetectedObjectTFBroadcaster()
    node.spin()
