#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import rospy
from moveit_commander import (
    MoveGroupCommander,
    RobotCommander,
    PlanningSceneInterface,
    roscpp_initialize,
    roscpp_shutdown
)
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import String, Float64

# ROS 노드 초기화 및 MoveIt 객체 생성
roscpp_initialize(sys.argv)
rospy.init_node('control_Husky_UR3_with_base_link', anonymous=True)
robot = RobotCommander()
scene = PlanningSceneInterface()

group_name = "manipulator"
move_group = MoveGroupCommander(group_name)
FIXED_FRAME = 'base_link'

error_pub = rospy.Publisher('/error_topic', String, queue_size=10)  # 에러 토픽 퍼블리셔

def move_ee(Px, Py, Pz, Rx, Ry, Rz, Rw):
    pose_target = Pose()
    pose_target.position.x = Px
    pose_target.position.y = Py
    pose_target.position.z = Pz
    pose_target.orientation.x = Rx
    pose_target.orientation.y = Ry
    pose_target.orientation.z = Rz
    pose_target.orientation.w = Rw

    move_group.set_pose_target(pose_target)
    success = move_group.go(wait=True)
    if not success:
        rospy.logwarn("No motion plan found.")
        error_message1 = "Error: Motion plan failed."
        rospy.loginfo(error_message1)
        error_pub.publish(error_message1)
    else:
        rospy.loginfo("Plan executed successfully!")

def set_default_pose():
    rospy.loginfo("Moving to default pose for vertical grasp...")
    move_Joint(1.5783, -1.977, 1.651, -1.843, -1.592, 0.034)
    rospy.loginfo("Default pose reached.")

def set_home_pose():
    rospy.loginfo("Moving to home pose...")
    move_Joint(-1.57, -2.276, 1.924, -1.57, -1.57, 0)
    rospy.loginfo("Home pose reached.")

def move_to_above_object(x, y, z):
    rospy.loginfo("Moving to position above object...")

    move_ee(
        x,
        y,
        z + 0.2,  # 물체 위 0.2m 높이
        1,
        0,
        0,
        0
    )

def grasp_object(x, y, z):
    rospy.loginfo("Starting grasp sequence...")

    # Step 1: Default Pose
    # set_default_pose()
    # input("Press Enter to move above the object...")

    # Step 2: Move to position above object
    move_to_above_object(x, y, z)
    input("Press Enter to move to grasp position...")

    # Step 3: Move to grasp position
    rospy.loginfo("Moving to grasp position...")
    move_ee(
        x,
        y,
        z,
        1,
        0,
        0,
        0
    )

    input("Press Enter to close the gripper...")

    # Step 4: Close gripper (단발성 퍼블리시)
    #publish_position_close()
    #input("Press Enter to return to home pose...")

    # Step 5: Return to home pose
    set_home_pose()
    input("Press Enter to open the gripper and release the object...")

    # Step 6: Open gripper (단발성 퍼블리시)
    #publish_position_open()
    #input("Press Enter to return to default pose...")

    # Step 7: Default pose
    set_default_pose()

def move_Joint(q1, q2, q3, q4, q5, q6):
    joint_goal = move_group.get_current_joint_values()
    joint_goal_list = [q1, q2, q3, q4, q5, q6]

    if len(joint_goal) != len(joint_goal_list):
        rospy.logerr("Joint goal list size mismatch.")
        return

    for i in range(6):
        joint_goal[i] = joint_goal_list[i]

    move_group.go(joint_goal, wait=True)


if __name__ == '__main__':
    try:
        rospy.loginfo("UR3 Base Link Control Program Initialized.")

        # 물체의 좌표값 (base_link 기준, 예제 값)
        x_dir = -(0.7-0.4)
        y_dir = 0
        z_dir = 0.2
        result = move_group.get_current_pose()
        print(result)
        grasp_object(x_dir, y_dir, z_dir)

    except rospy.ROSInterruptException:
        pass
    finally:
        roscpp_shutdown()
