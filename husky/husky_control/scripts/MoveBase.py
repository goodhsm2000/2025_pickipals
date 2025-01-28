#! /usr/bin/env python
# -*- coding:utf-8 -*-

import rospy

import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, Twist, PoseWithCovarianceStamped
from actionlib_msgs.msg import GoalStatus

class MoveBase(object):

    def __init__(self):
        self.check_state = True
        self.move_base = None
        
    def done_callback(self, status, result):
        if status == GoalStatus.SUCCEEDED:
            rospy.loginfo("Goal reached successfully!")
            self.goal_status = "success"  # 성공

        elif status in [GoalStatus.PREEMPTED, GoalStatus.ABORTED]:
            # recovery mode 때문에 ABORTED가 늦게 뜰 수도 있음
            # recovery mode를 포기할지 엘리베이터 문이 열렸는지를 파약하는 다른 방법을 쓸지 선택 필요
            rospy.logwarn("Goal was not completed: %s", GoalStatus.to_string(status))
            self.goal_status = "failure"  # 실패

        else:
            rospy.logwarn("Goal status: %s", GoalStatus.to_string(status))
            self.goal_status = "failure"  # 실패

        self.check_state = False

    def cancel_goal(self):
        rospy.loginfo("Cancelling the current goal...")
        self.move_base.cancel_goal()
        self.check_state = False

    def wait_for_result(self):
        rospy.loginfo("Waiting for goal result...")

        # 계속해서 상태 체크
        while self.check_state:
            rospy.sleep(1)  # 1초 간격으로 주기적으로 확인
            if self.goal_status == "failure":  # 실패 시
                self.cancel_goal()
                rospy.logwarn("Goal failed, stopping the robot.")
                break
            if self.goal_status == "success":  # 성공 시
                rospy.loginfo("Goal reached successfully.")
                break

        return self.goal_status == "success"  # 성공 여부 반환

    def goal_pub_func(self, gp):
    
        goal_msg = MoveBaseGoal()
        goal_msg.target_pose.header.frame_id = 'map'
        goal_msg.target_pose.header.stamp = rospy.Time.now()
        goal_msg.target_pose.pose.position.x = gp['x']
        goal_msg.target_pose.pose.position.y = gp['y']
        goal_msg.target_pose.pose.orientation.x = gp['qx']
        goal_msg.target_pose.pose.orientation.y = gp['qy']
        goal_msg.target_pose.pose.orientation.z = gp['qz']
        goal_msg.target_pose.pose.orientation.w = gp['qw']
        self.move_base.send_goal(goal_msg, done_cb=self.done_callback)
        self.check_state = True
        self.goal_status = "pending"

        rospy.loginfo("goal pub")


    def __call__(self, target_pose, move_base_client):
        self.move_base = move_base_client
        self.goal_pub_func(target_pose)
        result = self.wait_for_result()
    
        return result

