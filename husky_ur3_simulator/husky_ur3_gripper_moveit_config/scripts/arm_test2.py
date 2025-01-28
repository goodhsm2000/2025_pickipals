#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import rospy
from moveit_commander import (
    MoveGroupCommander,
    RobotCommander,
    PlanningSceneInterface,
    roscpp_initialize,
    roscpp_shutdown
)
from geometry_msgs.msg import PoseStamped, Pose
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose
from std_msgs.msg import Float64

# ROS 노드 초기화 및 MoveIt 객체 생성
roscpp_initialize(sys.argv)
rospy.init_node('control_Husky_UR3_with_base_link', anonymous=True)
robot = RobotCommander()
scene = PlanningSceneInterface()

# MoveIt 관련 설정
group_name = "ur3_manipulator"
move_group = MoveGroupCommander(group_name)
FIXED_FRAME = 'base_link'

# 오류 상태 변수
motion_plan_error = 0

# TF 초기화
tf_buffer = Buffer()
tf_listener = TransformListener(tf_buffer)

def transform_pose(target_frame, pose):
    try:
        transform = tf_buffer.lookup_transform(target_frame, pose.header.frame_id, rospy.Time(0), rospy.Duration(2.0))
        transformed_pose = do_transform_pose(pose, transform)
        return transformed_pose
    except Exception as e:
        rospy.logwarn(f"Transform failed: {e}")
        return None

def move_ee(Px, Py, Pz, Rx, Ry, Rz, Rw):
    global motion_plan_error

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
        motion_plan_error = 1  # 오류 상태 설정
        rospy.sleep(0.1)  # 잠시 동안 유지
        motion_plan_error = 0  # 상태 초기화
    else:
        rospy.loginfo("Plan executed successfully!")


def move_ee_with_current_orientation(Px, Py, Pz):
    """
    현재 그리퍼의 orientation(자세)을 유지하면서 목표 위치로 이동합니다.
    """
    global motion_plan_error    

    # 현재 자세 가져오기
    current_pose = move_group.get_current_pose().pose
    Rx = current_pose.orientation.x
    Ry = current_pose.orientation.y
    Rz = current_pose.orientation.z
    Rw = current_pose.orientation.w

    # 목표 위치와 현재 orientation으로 이동
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
        motion_plan_error = 1  # 오류 상태 설정
        rospy.sleep(0.1)  # 상태 유지
        motion_plan_error = 0  # 상태 초기화
    else:
        rospy.loginfo("Plan executed successfully!")


def move_Joint(q1, q2, q3, q4, q5, q6):
    joint_goal = move_group.get_current_joint_values()
    joint_goal_list = [q1, q2, q3, q4, q5, q6]

    if len(joint_goal) != len(joint_goal_list):
        rospy.logerr("Joint goal list size mismatch.")
        return

    for i in range(6):
        joint_goal[i] = joint_goal_list[i]

    move_group.go(joint_goal, wait=True)

def set_default_pose():
    rospy.loginfo("Moving to default pose...")
    move_Joint(1.5783, -1.977, 1.651, -1.843, -1.592, 0.034)
    rospy.loginfo("Default pose reached.")

def set_home_pose():
    rospy.loginfo("Moving to home pose...")
    move_Joint(0, -1.57, 1.57, -1.57, -1.57, 0)
    rospy.loginfo("Home pose reached.")

def set_updown_pose():
    rospy.loginfo("Moving to default pose...")
    move_Joint(1.5783, -1.573, 0.522, -1.900, -1.429, 3.141)
    rospy.loginfo("Default pose reached.")    

def set_number_pose():
    rospy.loginfo("Moving to default pose...")
    move_Joint(1.573, 0, -1.665, -1.564, -1.554, 3.141)
    rospy.loginfo("Default pose reached.")    

def move_to_above_object(x, y, z):
    rospy.loginfo("Transforming position to ur3_base_link...")

    # Transform base_link coordinates to ur3_base_link
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = 'odom'
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.pose.position.x = x
    pose_stamped.pose.position.y = y
    pose_stamped.pose.position.z = z + 0.2 # Above the object
    pose_stamped.pose.orientation.x = 1
    pose_stamped.pose.orientation.y = 0
    pose_stamped.pose.orientation.z = 0
    pose_stamped.pose.orientation.w = 0

    transformed_pose = transform_pose('base_link', pose_stamped)
    if not transformed_pose:
        rospy.logerr("Failed to transform pose to ur3_base_link.")
        return

    print(transformed_pose.pose.position.x, transformed_pose.pose.position.y, transformed_pose.pose.position.z)
    rospy.loginfo("Moving to transformed position above object...")
    move_ee(
        pose_stamped.pose.position.x,
        pose_stamped.pose.position.y,
        pose_stamped.pose.position.z,
        pose_stamped.pose.orientation.x,
        pose_stamped.pose.orientation.y,
        pose_stamped.pose.orientation.z,
        pose_stamped.pose.orientation.w
    )

def move_to_front_object(x, y, z):
    rospy.loginfo("Transforming position to ur3_base_link...")

    # Transform base_link coordinates to ur3_base_link
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = 'odom'
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.pose.position.x = x - 0.2
    pose_stamped.pose.position.y = y
    pose_stamped.pose.position.z = z
    pose_stamped.pose.orientation.x = 0
    pose_stamped.pose.orientation.y = 0
    pose_stamped.pose.orientation.z = 0
    pose_stamped.pose.orientation.w = 0

    transformed_pose = transform_pose('base_link', pose_stamped)
    if not transformed_pose:
        rospy.logerr("Failed to transform pose to ur3_base_link.")
        return

    rospy.loginfo("Moving to transformed position above object...")
    move_ee_with_current_orientation(
        transformed_pose.pose.position.x,
        transformed_pose.pose.position.y,
        transformed_pose.pose.position.z
    )



def publish_position_open():
    pub = rospy.Publisher('/rh_p12_rn_position/command', Float64, queue_size=10)
    rospy.loginfo("Opening the gripper...")
    pub.publish(0.0)

def publish_position_close():
    pub = rospy.Publisher('/rh_p12_rn_position/command', Float64, queue_size=10)
    rospy.loginfo("Closing the gripper...")
    pub.publish(1.05)

def publish_position_grasp():
    pub = rospy.Publisher('/rh_p12_rn_position/command', Float64, queue_size=10)
    rospy.loginfo("Closing the gripper...")
    pub.publish(0.5)


def grasp_object(x, y, z):
    rospy.loginfo("Starting grasp sequence...")

    # Step 1: Default Pose
    set_default_pose()
    time.sleep(10)

    # Step 2: Move to position above object
    move_to_above_object(x, y, z)
    time.sleep(10)

    # Step 3: Move to grasp position
    rospy.loginfo("Transforming position to ur3_base_link...")

    # Transform base_link coordinates to ur3_base_link
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = 'odom'
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.pose.position.x = x
    pose_stamped.pose.position.y = y
    pose_stamped.pose.position.z = z
    pose_stamped.pose.orientation.x = 1
    pose_stamped.pose.orientation.y = 0
    pose_stamped.pose.orientation.z = 0
    pose_stamped.pose.orientation.w = 0

    transformed_pose = transform_pose('base_link', pose_stamped)
    if not transformed_pose:
        rospy.logerr("Failed to transform pose to ur3_base_link.")
        return

    rospy.loginfo("Moving to transformed grasp position...")
    move_ee(
        transformed_pose.pose.position.x,
        transformed_pose.pose.position.y,
        transformed_pose.pose.position.z,
        transformed_pose.pose.orientation.x,
        transformed_pose.pose.orientation.y,
        transformed_pose.pose.orientation.z,
        transformed_pose.pose.orientation.w
    )

    time.sleep(10)


    # Step 4: grasp gripper
    publish_position_grasp()
    time.sleep(10)

    # Step 5: Return to home pose
    set_home_pose()
    time.sleep(10)

    # Step 6: Open gripper
    publish_position_open()
    time.sleep(10)

    # Step 7: Default pose
    set_default_pose()
    time.sleep(10)



    #updown 버튼 누르기
def press_updown(x, y, z):
    rospy.loginfo("Starting updown sequence...")
    publish_position_close()
    time.sleep(5)
    set_number_pose()
    time.sleep(10)

    move_to_front_object(x, y, z)
    time.sleep(10)

    #transform base_link coordinates to ur3_base_link
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = 'odom'
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.pose.position.x = x
    pose_stamped.pose.position.y = y
    pose_stamped.pose.position.z = z
    pose_stamped.pose.orientation.x = 0
    pose_stamped.pose.orientation.y = 0
    pose_stamped.pose.orientation.z = 0
    pose_stamped.pose.orientation.w = 0

    transformed_pose = transform_pose('base_link', pose_stamped)
    if not transformed_pose:
        rospy.logerr("Failed to transform pose to ur3_base_link.")
        return

    rospy.loginfo("Moving to transformed grasp position...")
    move_ee_with_current_orientation(
        transformed_pose.pose.position.x,
        transformed_pose.pose.position.y,
        transformed_pose.pose.position.z,
    )



def press_number(x, y, z):
    rospy.loginfo("Starting number sequence...")
    publish_position_close()
    time.sleep(5)
    set_number_pose()
    time.sleep(10)

    move_to_front_object(x, y, z)
    time.sleep(10)

    #transform base_link coordinates to ur3_base_link
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = 'odom'
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.pose.position.x = x
    pose_stamped.pose.position.y = y
    pose_stamped.pose.position.z = z
    pose_stamped.pose.orientation.x = 0
    pose_stamped.pose.orientation.y = 0
    pose_stamped.pose.orientation.z = 0
    pose_stamped.pose.orientation.w = 0

    transformed_pose = transform_pose('base_link', pose_stamped)
    if not transformed_pose:
        rospy.logerr("Failed to transform pose to ur3_base_link.")
        return

    rospy.loginfo("Moving to transformed grasp position...")
    move_ee_with_current_orientation(
        transformed_pose.pose.position.x,
        transformed_pose.pose.position.y,
        transformed_pose.pose.position.z,
    )

    






if __name__ == '__main__':
    try:
        rospy.loginfo("UR3 Control Program Initialized.")

        while not rospy.is_shutdown():
            rospy.loginfo("Choose an action: 1 (Grasp Object), 2 (Elevator Up/Down), 3 (Elevator Number)")
            action = input("Enter your choice: ")

            if action == "1":
                x, y, z = 0.7, 0.0, 0.3
                grasp_object(x, y, z)

            elif action == "2":
                x, y, z = 0.9, 0.0, 0.7
                press_updown(x, y, z)

            elif action == "3":
                x, y, z = 0.9, 0.0, 0.5
                press_number(x, y, z)

            else:
                rospy.logwarn("Invalid action. Please choose 1, 2, or 3.")

    except rospy.ROSInterruptException:
        pass
    finally:
        roscpp_shutdown()