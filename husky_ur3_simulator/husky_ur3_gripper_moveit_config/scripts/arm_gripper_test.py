#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import rospy
import pyrealsense2 as rs  # Intel Realsense 카메라 라이브러리
import numpy as np
import cv2
import time
import threading
from pyzbar.pyzbar import decode  # QR 코드 디코딩 라이브러리
from ultralytics import YOLO  # YOLO 객체 검출 라이브러리

# 그리퍼 제어를 위한 메시지 타입
from robotis_controller_msgs.msg import SyncWriteItem

# === ROS / MoveIt 관련 임포트 ===
from moveit_commander import (
    MoveGroupCommander,  # 특정 로봇 그룹(팔 등)을 제어하는 클래스
    RobotCommander,  # 로봇의 현재 상태와 전체 구조를 관리하는 클래스
    PlanningSceneInterface,  # 로봇 주변 환경(장애물 등)을 관리하는 클래스
    roscpp_initialize,  # MoveIt Commander 초기화 함수
    roscpp_shutdown  # MoveIt Commander 종료 함수
)
from geometry_msgs.msg import PoseStamped, Pose  # 위치와 자세를 표현하는 메시지 타입
from tf2_ros import Buffer, TransformListener  # TF 변환을 위한 클래스
from tf2_geometry_msgs import do_transform_pose  # TF 변환을 적용하는 함수
from std_msgs.msg import String, Float64  # 표준 메시지 타입
import tf.transformations as t  # TF 변환 관련 함수

# 전역 오류 상태 변수
motion_plan_error = 0

def transform_pose(target_frame, pose):
    """
    주어진 포즈를 target_frame 기준으로 변환하는 함수
    :param target_frame: 변환하고자 하는 목표 프레임
    :param pose: 변환할 포즈 (PoseStamped 메시지)
    :return: 변환된 포즈 (PoseStamped 메시지) 또는 None
    """
    try:
        transform = tf_buffer.lookup_transform(
            target_frame, 
            pose.header.frame_id, 
            rospy.Time(0), 
            rospy.Duration(2.0)
        )
        transformed_pose = do_transform_pose(pose, transform)
        return transformed_pose
    except Exception as e:
        rospy.logwarn(f"Transform failed: {e}")
        return None

def torque_on():
    """그리퍼의 토크를 활성화하는 함수"""
    # 그리퍼 제어를 위한 퍼블리셔 생성
    pub = rospy.Publisher('/robotis/sync_write_item', SyncWriteItem, queue_size=10)
    rospy.sleep(1)  # 퍼블리셔가 연결될 시간을 대기

    # SyncWriteItem 메시지 생성 및 설정
    msg = SyncWriteItem()
    msg.item_name = 'torque_enable'
    msg.joint_name = ['gripper']
    msg.value = [1]  # 1: 토크 활성화

    rospy.loginfo("Turning torque on for gripper")
    pub.publish(msg)  # 메시지 발행
    rospy.sleep(1)  # 명령이 실행될 시간을 대기

def gripper_open():
    """그리퍼를 여는 함수"""
    # 그리퍼 제어를 위한 퍼블리셔 생성
    pub = rospy.Publisher('/robotis/direct/sync_write_item', SyncWriteItem, queue_size=10)
    rospy.sleep(1)  # 퍼블리셔가 연결될 시간을 대기

    # SyncWriteItem 메시지 생성 및 설정
    msg = SyncWriteItem()
    msg.item_name = 'goal_position'
    msg.joint_name = ['gripper']
    msg.value = [0]  # 0: 그리퍼 열기 위치

    rospy.loginfo("Opening gripper")
    pub.publish(msg)  # 메시지 발행
    rospy.sleep(1)  # 명령이 실행될 시간을 대기

def gripper_close():
    """그리퍼를 닫는 함수"""
    # 그리퍼 제어를 위한 퍼블리셔 생성
    pub = rospy.Publisher('/robotis/direct/sync_write_item', SyncWriteItem, queue_size=10)
    rospy.sleep(1)  # 퍼블리셔가 연결될 시간을 대기

    # SyncWriteItem 메시지 생성 및 설정
    msg = SyncWriteItem()
    msg.item_name = 'goal_position'
    msg.joint_name = ['gripper']
    msg.value = [740]  # 740: 그리퍼 닫기 위치 (값은 그리퍼 모델에 따라 다름)

    rospy.loginfo("Closing gripper")
    pub.publish(msg)  # 메시지 발행
    rospy.sleep(1)  # 명령이 실행될 시간을 대기

def gripper_grasp():
    """그리퍼를 특정 위치로 닫는 함수 (grasping 동작)"""
    # 그리퍼 제어를 위한 퍼블리셔 생성
    pub = rospy.Publisher('/robotis/direct/sync_write_item', SyncWriteItem, queue_size=10)
    rospy.sleep(1)  # 퍼블리셔가 연결될 시간을 대기

    # SyncWriteItem 메시지 생성 및 설정
    msg = SyncWriteItem()
    msg.item_name = 'goal_position'
    msg.joint_name = ['gripper']
    msg.value = [300]  # 300: 그리퍼 닫기 위치 (grasping 용)

    rospy.loginfo("Closing gripper")
    pub.publish(msg)  # 메시지 발행
    rospy.sleep(1)  # 명령이 실행될 시간을 대기

def move_ee(Px, Py, Pz, Rx, Ry, Rz, Rw):
    """
    엔드 이펙터를 목표 위치와 자세로 이동시키는 함수
    :param Px: 목표 위치 X
    :param Py: 목표 위치 Y
    :param Pz: 목표 위치 Z
    :param Rx: 목표 자세 회전 쿼터니언 X
    :param Ry: 목표 자세 회전 쿼터니언 Y
    :param Rz: 목표 자세 회전 쿼터니언 Z
    :param Rw: 목표 자세 회전 쿼터니언 W
    """
    global motion_plan_error

    # 목표 포즈 생성
    pose_target = Pose()
    pose_target.position.x = Px
    pose_target.position.y = Py
    pose_target.position.z = Pz
    pose_target.orientation.x = Rx
    pose_target.orientation.y = Ry
    pose_target.orientation.z = Rz
    pose_target.orientation.w = Rw

    move_group.set_pose_target(pose_target)  # MoveGroup에 목표 포즈 설정
    success = move_group.go(wait=True)  # 계획 및 실행

    if not success:
        rospy.logwarn("No motion plan found.")
        motion_plan_error = 1
        rospy.sleep(0.1)
        motion_plan_error = 0
    else:
        rospy.loginfo("Plan executed successfully!")

def move_ee_with_current_orientation(Px, Py, Pz):
    """
    현재 그리퍼의 자세를 유지하면서 목표 위치로 이동하는 함수
    :param Px: 목표 위치 X
    :param Py: 목표 위치 Y
    :param Pz: 목표 위치 Z
    """
    global motion_plan_error
    current_pose = move_group.get_current_pose().pose  # 현재 포즈 가져오기

    # 현재 자세의 쿼터니언 추출
    Rx = current_pose.orientation.x
    Ry = current_pose.orientation.y
    Rz = current_pose.orientation.z
    Rw = current_pose.orientation.w

    # 목표 위치와 현재 자세로 새로운 목표 포즈 생성
    pose_target = Pose()
    pose_target.position.x = Px
    pose_target.position.y = Py
    pose_target.position.z = Pz
    pose_target.orientation.x = Rx
    pose_target.orientation.y = Ry
    pose_target.orientation.z = Rz
    pose_target.orientation.w = Rw

    move_group.set_pose_target(pose_target)  # MoveGroup에 목표 포즈 설정
    success = move_group.go(wait=True)  # 계획 및 실행

    if not success:
        rospy.logwarn("No motion plan found.")
        motion_plan_error = 1
        rospy.sleep(0.1)
        motion_plan_error = 0
    else:
        rospy.loginfo("Plan executed successfully!")

def move_Joint(q1, q2, q3, q4, q5, q6):
    """
    특정 관절 각도로 이동시키는 함수
    :param q1: 첫 번째 관절 목표 각도
    :param q2: 두 번째 관절 목표 각도
    :param q3: 세 번째 관절 목표 각도
    :param q4: 네 번째 관절 목표 각도
    :param q5: 다섯 번째 관절 목표 각도
    :param q6: 여섯 번째 관절 목표 각도
    """
    joint_goal = move_group.get_current_joint_values()  # 현재 관절 상태 가져오기
    joint_goal_list = [q1, q2, q3, q4, q5, q6]  # 목표 관절 각도 리스트

    if len(joint_goal) != len(joint_goal_list):
        rospy.logerr("Joint goal list size mismatch.")
        return

    for i in range(6):
        joint_goal[i] = joint_goal_list[i]  # 각 관절 목표 각도로 설정

    move_group.go(joint_goal, wait=True)  # 계획 및 실행

def set_default_pose():
    """로봇을 기본 자세로 이동시키는 함수"""
    rospy.loginfo("Moving to default pose...")
    move_Joint(3.073, -1.438, 1.688, -1.843, -1.592, 3.1415)
    rospy.loginfo("Default pose reached.")

def set_home_pose():
    """로봇을 홈 자세로 이동시키는 함수"""
    rospy.loginfo("Moving to home pose...")
    move_Joint(0, -1.57, 1.57, -1.57, -1.57, 0)
    rospy.loginfo("Home pose reached.")

def set_updown_pose():
    """로봇을 업다운 자세로 이동시키는 함수"""
    rospy.loginfo("Moving to updown pose...")
    move_Joint(1.5783, -1.573, 0.522, -1.900, -1.429, 3.141)
    rospy.loginfo("Updown pose reached.")    

def set_number_pose():
    """로봇을 숫자 자세로 이동시키는 함수"""
    rospy.loginfo("Moving to number pose...")
    move_Joint(-1.590, -1.438, -0.916, -0.802, 1.590, 0)
    rospy.loginfo("Number pose reached.")    

def way_point1():
    """바구니 이동 1"""
    rospy.loginfo("Moving to number pose...")
    move_Joint(1.59, -2.861, 1.688, -1.840, -1.594, 3.143)
    rospy.loginfo("Number pose reached.") 

def way_point2():
    """바구니 이동 2"""
    rospy.loginfo("Moving to number pose...")
    move_Joint(0.045, -2.861, 2.097, -1.840, -1.594, 3.142)
    rospy.loginfo("Number pose reached.") 


def move_to_above_object(x, y, z):
    """
    물체 위쪽으로 이동하는 함수
    :param x: 물체의 X 좌표
    :param y: 물체의 Y 좌표
    :param z: 물체의 Z 좌표
    """
    rospy.loginfo("Transforming position to ur3_base_link...")

    # odom 좌표계를 기준으로 물체의 위치를 정의 (물체 위쪽으로 이동)
    # 지금은 base_link 좌표계를 기준으로 물체의 위치를 정의
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = 'ur3_base_link'
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.pose.position.x = x 
    pose_stamped.pose.position.y = y
    pose_stamped.pose.position.z = z + 0.2  # 물체 위쪽으로 0.2m 이동
    pose_stamped.pose.orientation.x = 1
    pose_stamped.pose.orientation.y = 0
    pose_stamped.pose.orientation.z = 0
    pose_stamped.pose.orientation.w = 0

    # odom 좌표계를 기준으로 base_link 좌표계로 변환 -> 테스트 시 필요 없음
    # transformed_pose = transform_pose('base_link', pose_stamped)
    # if not transformed_pose:
    #     rospy.logerr("Failed to transform pose to ur3_base_link.")
    #     return

    # rospy.loginfo(f"Moving above object: {transformed_pose.pose.position.x}, "
    #               f"{transformed_pose.pose.position.y}, {transformed_pose.pose.position.z}")

    # 엔드 이펙터를 목표 위치로 이동
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
    """
    물체 앞쪽으로 이동하는 함수
    :param x: 물체의 X 좌표
    :param y: 물체의 Y 좌표
    :param z: 물체의 Z 좌표
    """
    rospy.loginfo("Transforming position to ur3_base_link...")

    # odom 좌표계를 기준으로 물체의 앞쪽으로 이동 (X축 -0.2m)
    # 지금은 base_link 기준 좌표로 이동
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = 'ur3_base_link'
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.pose.position.x = x   
    pose_stamped.pose.position.y = y -0.2
    pose_stamped.pose.position.z = z 
    pose_stamped.pose.orientation.x = 0
    pose_stamped.pose.orientation.y = 0
    pose_stamped.pose.orientation.z = 0
    pose_stamped.pose.orientation.w = 0

    # odom 좌표계를 기준으로 base_link 좌표계로 변환
    # 테스트 시에는 필요 없음
    # transformed_pose = transform_pose('base_link', pose_stamped)
    # if not transformed_pose:
    #     rospy.logerr("Failed to transform pose to ur3_base_link.")
    #     return

    # rospy.loginfo(f"Moving front of object: {transformed_pose.pose.position.x}, "
    #               f"{transformed_pose.pose.position.y}, {transformed_pose.pose.position.z}")

    # 현재 자세를 유지하면서 엔드 이펙터를 목표 위치로 이동
    move_ee_with_current_orientation(
        pose_stamped.pose.position.x,
        pose_stamped.pose.position.y,
        pose_stamped.pose.position.z
    )

def grasp_object(x, y, z):
    """
    물체를 잡는 일련의 동작을 수행하는 함수
    :param x: 물체의 X 좌표
    :param y: 물체의 Y 좌표
    :param z: 물체의 Z 좌표
    """
    rospy.loginfo("Starting grasp sequence...")

    # Step 1: 기본 자세로 이동
    set_default_pose()
    time.sleep(10)  # 이동 시간 대기

    # Step 2: 물체 위쪽으로 이동
    move_to_above_object(x, y, z)
    time.sleep(10)  # 이동 시간 대기

    # Step 3: 물체를 잡기 위한 자세로 이동
    rospy.loginfo("Transforming position to grasp object...")
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = 'ur3_base_link' # base_link 기준으로
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.pose.position.x = x
    pose_stamped.pose.position.y = y 
    pose_stamped.pose.position.z = z + 0.05
    pose_stamped.pose.orientation.x = 1
    pose_stamped.pose.orientation.y = 0
    pose_stamped.pose.orientation.z = 0
    pose_stamped.pose.orientation.w = 0

    # odom 좌표계를 기준으로 base_link 좌표계로 변환
    # 테스트 시 필요 없음
    # transformed_pose = transform_pose('base_link', pose_stamped)
    # if not transformed_pose:
    #     rospy.logerr("Failed to transform pose to ur3_base_link.")
    #     return

    # 엔드 이펙터를 목표 자세로 이동
    move_ee(
        pose_stamped.pose.position.x,
        pose_stamped.pose.position.y,
        pose_stamped.pose.position.z,
        pose_stamped.pose.orientation.x,
        pose_stamped.pose.orientation.y,
        pose_stamped.pose.orientation.z,
        pose_stamped.pose.orientation.w
    )

    time.sleep(10)  # 이동 시간 대기

    # Step 4: 그리퍼를 닫아 물체를 잡음
    gripper_grasp()
    time.sleep(10)  # 그리퍼 동작 시간 대기

    # Step 5: 바구니 자세로 이동
    way_point1()
    time.sleep(5)  # 이동 시간 대기

    way_point2()
    time.sleep(5)  # 이동 시간 대기

    # Step 6: 그리퍼를 열어 물체를 놓음
    gripper_open()
    time.sleep(10)  # 그리퍼 동작 시간 대기

    # Step 7: 기본 자세로 이동
    set_home_pose()
    time.sleep(5)  # 이동 시간 대기

def press_updown(x, y, z):
    """
    업다운 버튼을 누르는 일련의 동작을 수행하는 함수
    :param x: 버튼의 X 좌표
    :param y: 버튼의 Y 좌표
    :param z: 버튼의 Z 좌표
    """
    rospy.loginfo("Starting updown sequence...")
    gripper_close()  # 그리퍼 닫기
    time.sleep(5)  # 그리퍼 동작 시간 대기

    set_number_pose()  # 숫자 자세로 이동
    time.sleep(10)  # 이동 시간 대기

    move_to_front_object(x, y, z)  # 버튼 앞쪽으로 이동
    time.sleep(10)  # 이동 시간 대기

    # odom 좌표계를 기준으로 버튼 위치 정의
    # 지금은 base_link 좌표계 기준으로
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = 'ur3_base_link'
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.pose.position.x = x
    pose_stamped.pose.position.y = y
    pose_stamped.pose.position.z = z
    pose_stamped.pose.orientation.x = 0
    pose_stamped.pose.orientation.y = 0
    pose_stamped.pose.orientation.z = 0
    pose_stamped.pose.orientation.w = 0

    # odom 좌표계를 기준으로 base_link 좌표계로 변환
    # 테스트 시 필요 없음
    # transformed_pose = transform_pose('base_link', pose_stamped)
    # if not transformed_pose:
    #     rospy.logerr("Failed to transform pose to ur3_base_link.")
    #     return

    rospy.loginfo("Pressing updown button...")
    # 현재 자세를 유지하면서 엔드 이펙터를 목표 위치로 이동
    move_ee_with_current_orientation(
        pose_stamped.pose.position.x,
        pose_stamped.pose.position.y,
        pose_stamped.pose.position.z,
    )

def press_number(x, y, z):
    """
    숫자 버튼을 누르는 일련의 동작을 수행하는 함수
    :param x: 버튼의 X 좌표
    :param y: 버튼의 Y 좌표
    :param z: 버튼의 Z 좌표
    """
    rospy.loginfo("Starting number sequence...")
    gripper_close()  # 그리퍼 닫기
    time.sleep(5)  # 그리퍼 동작 시간 대기

    set_number_pose()  # 숫자 자세로 이동
    time.sleep(10)  # 이동 시간 대기

    move_to_front_object(x, y, z)  # 버튼 앞쪽으로 이동
    time.sleep(10)  # 이동 시간 대기

    # odom 좌표계를 기준으로 버튼 위치 정의
    # 지금은 base_link 기준으로
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = 'ur3_base_link'
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.pose.position.x = x
    pose_stamped.pose.position.y = y - 0.1
    pose_stamped.pose.position.z = z
    pose_stamped.pose.orientation.x = 0
    pose_stamped.pose.orientation.y = 0
    pose_stamped.pose.orientation.z = 0
    pose_stamped.pose.orientation.w = 0

    # odom 좌표계를 기준으로 base_link 좌표계로 변환
    # transformed_pose = transform_pose('base_link', pose_stamped)
    # if not transformed_pose:
    #     rospy.logerr("Failed to transform pose to ur3_base_link.")
    #     return

    rospy.loginfo("Pressing number button...")
    # 현재 자세를 유지하면서 엔드 이펙터를 목표 위치로 이동
    move_ee_with_current_orientation(
        pose_stamped.pose.position.x,
        pose_stamped.pose.position.y,
        pose_stamped.pose.position.z,
    )

# ======================================================
# ActionExecutor 클래스 (grasp, press_number, press_updown 기능)
# ======================================================
class ActionExecutor:
    def __init__(self, node):
        """
        ActionExecutor 클래스의 초기화 함수
        :param node: YoloDepthAndWait 클래스의 인스턴스 (카메라와 물체 좌표 검출)
        """
        self.node = node

    def grasp_object(self):
        """
        물체를 잡는 동작을 실행
        """
        if self.node.object_found and self.node.latest_xyz is not None:
            x, y, z = self.node.latest_xyz
            grasp_object(x, y, z)
            return True
        else:
            rospy.logwarn("No object found yet for grasping.")
            return False

    def press_updown(self):
        """
        업다운 버튼을 누르는 동작을 실행
        """
        if self.node.object_found and self.node.latest_xyz is not None:
            x, y, z = self.node.latest_xyz
            press_updown(x, y, z)
            return True
        else:
            rospy.logwarn("No object found yet for press_updown.")
            return False

    def press_number(self):
        """
        숫자 버튼을 누르는 동작을 실행
        """
        if self.node.object_found and self.node.latest_xyz is not None:
            x, y, z = self.node.latest_xyz
            press_number(x, y, z)
            return True
        else:
            rospy.logwarn("No object found yet for press_number.")
            return False

# ======================================================
# YoloDepthAndWait 클래스 (카메라 루프 및 좌표 변환 처리)
# ======================================================
class YoloDepthAndWait:
    def __init__(self, mode):
        """
        YoloDepthAndWait 클래스의 초기화 함수
        - Realsense 카메라 설정
        - YOLO 모델 로드
        - 프레임 ID 설정
        - 객체 검출 상태 초기화
        """
        # Realsense 파이프라인 설정
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 깊이 스트림 설정
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 컬러 스트림 설정
        self.pipeline.start(config)  # 카메라 시작
        self.align = rs.align(rs.stream.color)  # 컬러 스트림에 맞춰 깊이 스트림 정렬
        self.mode = mode

        # YOLO 모델 로드 (사용자 환경에 맞춰 경로 수정 필요)
        self.model = YOLO('/home/husky/Downloads/best.pt')

        # 프레임 ID 설정
        self.camera_link = "camera_link"  # 카메라 프레임
       #self.odom_link = "odom"  # 오도메트리 프레임
        self.base_link = "ur3_base_link" # 테스트 base_link 기준
        # 객체 검출 상태 초기화
        self.object_found = False       # 객체 발견 여부
        self.latest_xyz = None     # 최신 객체의 base_link 좌표

    def transform_camera_to_base_link(self, X, Y, Z):
        """
        camera_link 좌표계의 (X, Y, Z)를 base_link 좌표계로 변환하는 함수
        :param X: 카메라 좌표계 X
        :param Y: 카메라 좌표계 Y
        :param Z: 카메라 좌표계 Z
        :return: base_link 좌표계의 (x, y, z) 또는 None
        """
        global tf_buffer
        try:
            # odom 프레임 기준으로 camera_link 프레임으로의 변환 정보 가져오기
            transform = tf_buffer.lookup_transform(
                self.base_link,
                self.camera_link,
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            trans = transform.transform.translation  # 변환의 평행 이동
            rot   = transform.transform.rotation  # 변환의 회전
            trans_vec = [trans.x, trans.y, trans.z]
            quat      = [rot.x, rot.y, rot.z, rot.w]

            # 쿼터니언을 행렬로 변환
            mat = t.quaternion_matrix(quat)
            mat[0:3, 3] = trans_vec  # 평행 이동 추가

            pt_cam  = np.array([X, Y, Z, 1.0])  # 카메라 좌표계 포인트
            pt_base_link = mat @ pt_cam  # 변환 행렬을 사용하여 base_link 좌표계로 변환
            return pt_base_link[:3]  # (x, y, z) 반환
        except Exception as e:
            rospy.logwarn(f"transform_camera_to_base_link error: {e}")
            return None


    def run(self):
        """
        카메라 프레임을 지속적으로 처리하며, QR 코드와 YOLO로 객체를 인식.
        Enter 키(ASCII=13)가 눌리면, 가장 최근 인식된 객체를 잡는 동작을 수행.
        """
        rate = rospy.Rate(15)  # 루프 주기 설정 (15Hz)

        while not rospy.is_shutdown():
            frames = self.pipeline.wait_for_frames()  # 프레임 대기
            aligned_frames = self.align.process(frames)  # 프레임 정렬
            depth_frame = aligned_frames.get_depth_frame()  # 깊이 프레임 가져오기
            color_frame = aligned_frames.get_color_frame()  # 컬러 프레임 가져오기
            if not depth_frame or not color_frame:
                continue  # 프레임이 없으면 다음 루프로

            frame = np.asanyarray(color_frame.get_data())  # 프레임 데이터를 NumPy 배열로 변환
            intr = depth_frame.profile.as_video_stream_profile().get_intrinsics()  # 카메라 내부 파라미터 가져오기

            # 1) QR 코드 탐지
            qr_codes = decode(frame)  # 프레임에서 QR 코드 디코딩
            for qr in qr_codes:
                x, y, w, h = qr.rect  # QR 코드의 위치와 크기
                cx = x + w//2  # QR 코드의 중심 X
                cy = y + h//2  # QR 코드의 중심 Y
                depth_val = depth_frame.get_distance(cx, cy)  # 중심 지점의 깊이 값
                X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], depth_val)  # 픽셀 좌표를 3D 좌표로 변환
                base_pt = self.transform_camera_to_base_link(X, Y, Z)  # 카메라 좌표계를 base_link 좌표계로 변환
                if base_pt is not None:
                    txt = qr.data.decode('utf-8')  # QR 코드 데이터 디코딩
                    rospy.loginfo(
                        f"QR Detected: {txt}, base=({base_pt[0]:.3f},{base_pt[1]:.3f},{base_pt[2]:.3f})"
                    )
                    # 객체 발견 상태 업데이트
                    self.object_found = True
                    self.latest_xyz = base_pt

                # 디스플레이에 QR 코드 표시
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)  # QR 코드 사각형 그리기
                cv2.putText(
                    frame,
                    f"{qr.data.decode('utf-8')} {depth_val:.2f}m",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,255,0), 1
                )  # QR 코드 텍스트 표시

            # 2) YOLO 탐지
            results = self.model(frame, stream=True, conf = 0.8, half=True)  # YOLO 모델로 객체 검출
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
                    cx = (x1 + x2)//2  # 바운딩 박스의 중심 X
                    cy = (y1 + y2)//2  # 바운딩 박스의 중심 Y
                    depth_val = depth_frame.get_distance(cx, cy)  # 중심 지점의 깊이 값

                    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], depth_val)  # 픽셀 좌표를 3D 좌표로 변환
                    base_pt = self.transform_camera_to_base_link(X, Y, Z)  # 카메라 좌표계를 base_link 좌표계로 변환
                    if base_pt is not None:
                        label = self.model.names[int(box.cls)]  # 객체 라벨
                        conf  = float(box.conf.item())  # 신뢰도
                        rospy.loginfo(
                            f"YOLO Detected: {label}({conf:.2f}), "
                            f"base=({base_pt[0]:.3f},{base_pt[1]:.3f},{base_pt[2]:.3f})"
                        )
                        # 객체 발견 상태 업데이트
                        self.object_found = True
                        self.latest_xyz = base_pt

                        # 디스플레이에 YOLO 객체 표시
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)  # 객체 사각형 그리기
                        cv2.putText(
                            frame,
                            f"{label} {conf:.2f} {depth_val:.2f}m",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255,0,0), 1
                        )  # 객체 텍스트 표시

            # 3) 시각화
            cv2.imshow("Realsense YOLO/QR detection", frame)  # OpenCV 창에 프레임 표시
            key = cv2.waitKey(1) & 0xFF  # 키 입력 대기
            # ESC 키(ASCII=27)로 종료
            if key == 27:
                break
            
            if self.mode == "grasp":
                if self.object_found and self.latest_xyz is not None:
                    x_o, y_o, z_o = self.latest_xyz
                    grasp_object(x_o, y_o, z_o)
                    self.object_found = False
                else:
                    rospy.logwarn("No object found yet.")
            else:
                if self.object_found and self.latest_xyz is not None:
                    x_o, y_o, z_o = self.latest_xyz
                    print(x_o, y_o, z_o)
                    press_number(x_o, y_o, z_o)
                    self.object_found = False
                else:
                    rospy.logwarn("No EV button found yet.")

            rate.sleep()  # 루프 주기 유지

# ======================================================
# 메인 함수
# ======================================================
if __name__ == "__main__":
    try:
        rospy.init_node("yolo_main_node", anonymous=True)  # ROS 노드 초기화
        # rospy.init_node('gripper_control_node', anonymous=True)
        torque_on()  # 그리퍼 토크 활성화
        roscpp_initialize(sys.argv)  # MoveIt Commander 초기화

        global tf_buffer, tf_listener, error_pub, move_group
        tf_buffer = Buffer()  # TF 버퍼 생성
        tf_listener = TransformListener(tf_buffer)  # TF 리스너 생성

        error_pub = rospy.Publisher('/error_topic', String, queue_size=10)  # 오류 메시지 퍼블리셔 생성

        robot = RobotCommander()  # 로봇 상태 정보 가져오기
        scene = PlanningSceneInterface()  # 플래닝 씬 인터페이스 초기화
        #move_group = MoveGroupCommander("manipulator")  # MoveGroupCommander 초기화 (그룹 이름: manipulator)
        group_name = "manipulator"
        move_group = MoveGroupCommander(group_name)
        set_number_pose()
        # FIXED_FRAME = 'base_link'
        # YoloDepthAndWait 객체 생성 (카메라 루프를 위한 객체)
        node = YoloDepthAndWait(mode="press_button")

        # ActionExecutor 객체 생성 (grasp, press_number, press_updown 기능 제공)
        action_executor = ActionExecutor(node)

        # 카메라 루프를 별도 스레드에서 실행 (물체 좌표를 지속적으로 업데이트)
        camera_thread = threading.Thread(target=node.run)
        camera_thread.start()

        rospy.loginfo("Waiting for object detection...")
        # 단순 예시로, 물체가 검출될 때까지 대기
        while not rospy.is_shutdown() and not node.object_found:
            rospy.sleep(0.1)

        rospy.loginfo("Object detected, executing grasp sequence...")
        action_executor.press_number()  # 객체를 잡는 시퀀스 실행

        # grasp가 완료된 후 종료 (필요에 따라 다른 행동 실행 가능)
        rospy.loginfo("Grasp sequence finished.")

    except rospy.ROSInterruptException:
        pass
    finally:
        roscpp_shutdown()  # MoveIt Commander 종료
