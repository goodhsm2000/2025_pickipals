#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import rospy
import pyrealsense2 as rs  # Intel Realsense
import numpy as np
import cv2
import time
import threading
from pyzbar.pyzbar import decode
from ultralytics import YOLO

from robotis_controller_msgs.msg import SyncWriteItem

# === ROS / MoveIt 관련 ===
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
from std_msgs.msg import String, Float64
import tf.transformations as t


def transform_pose(target_frame, pose):
    """포즈를 target_frame 기준으로 변환"""
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
    pub = rospy.Publisher('/robotis/sync_write_item', SyncWriteItem, queue_size=10)
    rospy.sleep(1)

    msg = SyncWriteItem()
    msg.item_name = 'torque_enable'
    msg.joint_name = ['gripper']
    msg.value = [1]  # 1: 토크 활성화

    rospy.loginfo("Turning torque on for gripper")
    pub.publish(msg)
    rospy.sleep(1)

def gripper_open():
    pub = rospy.Publisher('/robotis/direct/sync_write_item', SyncWriteItem, queue_size=10)
    rospy.sleep(1)

    msg = SyncWriteItem()
    msg.item_name = 'goal_position'
    msg.joint_name = ['gripper']
    msg.value = [0]  # 0: 그리퍼 열기

    rospy.loginfo("Opening gripper")
    pub.publish(msg)
    rospy.sleep(1)

def gripper_close():
    pub = rospy.Publisher('/robotis/direct/sync_write_item', SyncWriteItem, queue_size=10)
    rospy.sleep(1)

    msg = SyncWriteItem()
    msg.item_name = 'goal_position'
    msg.joint_name = ['gripper']
    msg.value = [740]  # 740: 그리퍼 닫기 위치

    rospy.loginfo("Closing gripper")
    pub.publish(msg)
    rospy.sleep(3)

def gripper_grasp():
    """그리퍼를 특정 위치(300)로 닫기 -> grasp"""
    pub = rospy.Publisher('/robotis/direct/sync_write_item', SyncWriteItem, queue_size=10)
    rospy.sleep(1)

    msg = SyncWriteItem()
    msg.item_name = 'goal_position'
    msg.joint_name = ['gripper']
    msg.value = [300]

    rospy.loginfo("Closing gripper for grasp")
    pub.publish(msg)
    rospy.sleep(3)


def move_ee(Px, Py, Pz, Rx, Ry, Rz, Rw, max_retries=3):
    """
    엔드 이펙터를 목표 Pose로 이동, 실패 시 재시도
    """
    pose_target = Pose()
    pose_target.position.x = Px
    pose_target.position.y = Py
    pose_target.position.z = Pz
    pose_target.orientation.x = Rx
    pose_target.orientation.y = Ry
    pose_target.orientation.z = Rz
    pose_target.orientation.w = Rw

    for attempt in range(max_retries):
        move_group.set_pose_target(pose_target)
        success = move_group.go(wait=True)

        if success:
            rospy.loginfo("move_ee: Plan executed successfully!")
            return True
        else:
            rospy.logwarn(f"move_ee: No motion plan found. Retrying... {attempt+1}/{max_retries}")
            rospy.sleep(1.0)

    rospy.logerr("move_ee: Motion plan failed after all retry attempts.")
    return False


def move_ee_with_current_orientation(Px, Py, Pz, max_retries=3):
    """
    현재 orientation 유지, X/Y/Z만 변경
    """
    current_pose = move_group.get_current_pose().pose
    Rx = current_pose.orientation.x
    Ry = current_pose.orientation.y
    Rz = current_pose.orientation.z
    Rw = current_pose.orientation.w

    pose_target = Pose()
    pose_target.position.x = Px
    pose_target.position.y = Py
    pose_target.position.z = Pz
    pose_target.orientation.x = Rx
    pose_target.orientation.y = Ry
    pose_target.orientation.z = Rz
    pose_target.orientation.w = Rw

    for attempt in range(max_retries):
        move_group.set_pose_target(pose_target)
        success = move_group.go(wait=True)
        if success:
            rospy.loginfo("move_ee_with_current_orientation: Plan executed successfully!")
            return True
        else:
            rospy.logwarn(f"move_ee_with_current_orientation: No motion plan found. Retrying... {attempt+1}/{max_retries}")
            rospy.sleep(1.0)

    rospy.logerr("move_ee_with_current_orientation: Motion plan failed after all retry attempts.")
    return False


def move_Joint(q1, q2, q3, q4, q5, q6, max_retries=3):
    """
    특정 관절 각도로 이동, 실패 시 재시도
    """
    joint_goal_list = [q1, q2, q3, q4, q5, q6]

    for attempt in range(max_retries):
        joint_goal = move_group.get_current_joint_values()
        if len(joint_goal) != 6:
            rospy.logerr("move_Joint: Joint goal size mismatch.")
            return False

        for i in range(6):
            joint_goal[i] = joint_goal_list[i]

        success = move_group.go(joint_goal, wait=True)
        if success:
            rospy.loginfo("move_Joint: Joint move executed successfully!")
            return True
        else:
            rospy.logwarn(f"move_Joint: No motion plan found. Retrying... {attempt+1}/{max_retries}")
            rospy.sleep(1.0)

    rospy.logerr("move_Joint: Joint motion failed after all retry attempts.")
    return False


def execute_cartesian_path_with_retry(waypoints, eef_step=0.002, jump_threshold=0.0, max_retries=3):
    """
    Cartesian Path 생성 및 실행, 실패 시 재시도
    """
    for attempt in range(max_retries):
        # avoid_collisions를 명시적 키워드로
        plan, fraction = move_group.compute_cartesian_path(
            waypoints,
            eef_step,
            avoid_collisions=True
        )

        if fraction < 0.95:
            rospy.logwarn(f"execute_cartesian_path_with_retry: fraction={fraction:.2f} <0.95. Retry {attempt+1}/{max_retries}")
            rospy.sleep(1.0)
            continue

        exec_success = move_group.execute(plan, wait=True)
        if exec_success:
            rospy.loginfo("execute_cartesian_path_with_retry: Cartesian path executed successfully!")
            return True
        else:
            rospy.logwarn(f"execute_cartesian_path_with_retry: Execution failed. Retry {attempt+1}/{max_retries}")
            rospy.sleep(1.0)

    rospy.logerr("execute_cartesian_path_with_retry: Failed after all retries.")
    return False


def set_default_pose():
    rospy.loginfo("Moving to default pose...")
    move_Joint(3.073, -1.438, 1.688, -1.843, -1.592, 3.1415)
    rospy.loginfo("Default pose reached.")

def set_origin_pose():
    rospy.loginfo("Moving to origin pose...")
    move_Joint(-4.7985707e-05, -1.5707724, -4.7985707e-05, -1.5707963, 5.9921127e-05, -5.9906636e-05)
    rospy.loginfo("Origin pose reached.")

def set_number_pose():
    rospy.loginfo("Moving to number pose...")
    move_Joint(1.570, -1.570, 0.0, -1.570, 0.0, 0.0)
    rospy.loginfo("Number pose reached.")

def knock_point1():
    rospy.loginfo("Moving to knock point1...")
    move_Joint(0, -2.05, 0.0, -0.396, 1.570, 0.0)
    rospy.loginfo("knock point1 reached.")

def knock_point2():
    rospy.loginfo("Moving to knock point2...")
    move_Joint(0, -2.05, 0.0, -0.854, 1.570, 0.0)
    rospy.loginfo("knock point2 reached.")

def way_point1():
    rospy.loginfo("Moving to waypoint1...")
    move_Joint(1.59, -2.861, 1.688, -1.840, -1.594, 3.14)
    rospy.loginfo("waypoint1 reached.")

def way_point2():
    rospy.loginfo("Moving to waypoint2...")
    move_Joint(0.045, -2.861, 2.097, -1.840, -1.594, 3.14)
    rospy.loginfo("waypoint2 reached.")


def move_to_above_object(x, y, z):
    """
    물체 위쪽으로 이동
    """
    rospy.loginfo("Move_to_above_object...")
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = 'base_link'
    pose_stamped.pose.position.x = x
    pose_stamped.pose.position.y = y
    pose_stamped.pose.position.z = z + 0.25
    pose_stamped.pose.orientation.x = 1
    pose_stamped.pose.orientation.y = 0
    pose_stamped.pose.orientation.z = 0
    pose_stamped.pose.orientation.w = 0

    success = move_ee(
        pose_stamped.pose.position.x,
        pose_stamped.pose.position.y,
        pose_stamped.pose.position.z,
        pose_stamped.pose.orientation.x,
        pose_stamped.pose.orientation.y,
        pose_stamped.pose.orientation.z,
        pose_stamped.pose.orientation.w
    )
    return success


def move_to_front_object(x, y, z):
    """
    버튼 앞쪽으로 이동
    """
    rospy.loginfo("Move_to_front_object...")
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = 'base_link'
    pose_stamped.pose.position.x = x - 0.22
    pose_stamped.pose.position.y = y
    pose_stamped.pose.position.z = z

    success = move_ee_with_current_orientation(
        pose_stamped.pose.position.x,
        pose_stamped.pose.position.y,
        pose_stamped.pose.position.z
    )
    return success


def grasp_object(x, y, z):
    """
    물체 잡는 동작
    """
    rospy.loginfo("Starting grasp sequence...")

    # Step 1) 물체 위쪽으로 이동
    ok = move_to_above_object(x, y, z)
    if not ok:
        return False
    time.sleep(1)

    # Step 2) 조금 더 내려감
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = 'base_link'
    pose_stamped.pose.position.x = x
    pose_stamped.pose.position.y = y
    pose_stamped.pose.position.z = z + 0.13
    pose_stamped.pose.orientation.x = 1
    pose_stamped.pose.orientation.y = 0
    pose_stamped.pose.orientation.z = 0
    pose_stamped.pose.orientation.w = 0

    ok2 = move_ee(
        pose_stamped.pose.position.x,
        pose_stamped.pose.position.y,
        pose_stamped.pose.position.z,
        pose_stamped.pose.orientation.x,
        pose_stamped.pose.orientation.y,
        pose_stamped.pose.orientation.z,
        pose_stamped.pose.orientation.w
    )
    if not ok2:
        return False

    time.sleep(1)
    gripper_grasp()
    time.sleep(1)

    way_point1()
    time.sleep(1)
    way_point2()
    time.sleep(1)

    gripper_open()
    time.sleep(1)

    set_origin_pose()
    return True


def press_number(x, y, z):
    """
    버튼 누르기 동작:
    1) 버튼 앞 이동
    2) 카트esian 경로 (2cm 전진->복귀)
    3) origin pose
    """
    rospy.loginfo("Pressing number button...")

    ok = move_to_front_object(x, y, z)
    if not ok:
        return False
    time.sleep(1)

    current_pose = move_group.get_current_pose().pose
    waypoints = []

    # (1) 시작점
    waypoints.append(current_pose)

    # (2) 2cm 전진
    press_pose = Pose()
    press_pose.position.x = current_pose.position.x + 0.02
    press_pose.position.y = current_pose.position.y
    press_pose.position.z = current_pose.position.z
    press_pose.orientation = current_pose.orientation
    waypoints.append(press_pose)

    # (3) 원위치로 복귀
    waypoints.append(current_pose)

    success = execute_cartesian_path_with_retry(waypoints, eef_step=0.002, jump_threshold=0.0, max_retries=3)
    if not success:
        return False

    set_origin_pose()
    time.sleep(1)
    return True


class ArmGripper(object):
    def __init__(self):
        torque_on()
        roscpp_initialize(sys.argv)

        gripper_close()
        time.sleep(1)
        gripper_open()
        time.sleep(1)

        global tf_buffer, tf_listener, error_pub, move_group
        tf_buffer = Buffer()
        tf_listener = TransformListener(tf_buffer)

        error_pub = rospy.Publisher('/error_topic', String, queue_size=10)

        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()

        group_name = "ur3_manipulator"
        move_group = MoveGroupCommander(group_name)
        move_group.set_end_effector_link("tool0")

        print("Planning frame:", move_group.get_planning_frame())
        print("Current pose:", move_group.get_current_pose().pose)

        # Realsense 파이프라인
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # YOLO 모델
        self.model = YOLO('/home/husky/Downloads/best.pt')

        self.camera_link = "camera_link"
        self.base_link   = "base_link"

        self.object_found = False
        self.latest_xyz = None

    def __call__(self, mode, target_num, target_floor):
        rate = rospy.Rate(15)

        self.object_found = False

        if mode == "pickup":
            rospy.loginfo("pickup mode...")
            gripper_open()
            time.sleep(1)
            set_default_pose()
            time.sleep(1)

        elif mode == "knock":
            rospy.loginfo("knock mode...")
            gripper_close()
            time.sleep(1)
            knock_point1()
            time.sleep(1)
            knock_point2()
            time.sleep(1)
            knock_point1()
            time.sleep(1)
            knock_point2()
            return True

        else:  # 엘리베이터 버튼 누르기
            rospy.loginfo("elevator mode...")
            gripper_close()
            time.sleep(1)
            set_number_pose()
            time.sleep(1)

        while not rospy.is_shutdown() and not self.object_found:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            intr = depth_frame.profile.as_video_stream_profile().get_intrinsics()

            # 1) QR 코드 탐지
            qr_codes = decode(frame)
            for qr in qr_codes:
                x, y, w, h = qr.rect
                cx = x + w//2
                cy = y + h//2
                depth_val = depth_frame.get_distance(cx, cy)
                if depth_val == 0:
                    continue
                X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], depth_val)
                base_pt = self.transform_camera_to_base_link(X, Y, Z)
                if base_pt is not None:
                    txt = qr.data.decode('utf-8')
                    rospy.loginfo(
                        f"QR Detected: {txt}, base=({base_pt[0]:.3f},{base_pt[1]:.3f},{base_pt[2]:.3f})"
                    )
                    # pickup 모드에서 특정 QR
                    if mode == "pickup" and txt == "TEL:01093512078":
                        self.object_found = True
                        self.latest_xyz = base_pt

                decoded_text = qr.data.decode('utf-8')

                # 만약 디코딩된 값이 "01059585328" 이라면, "TEL:01059585328"으로 표시
                if decoded_text == "01059585328":
                    display_text = f"TEL:{decoded_text}"
                else:
                    display_text = decoded_text

                # 사각형 그리기 (기존 코드와 동일)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

                # 텍스트 표시 부분
                cv2.putText(
                    frame,
                    f"{display_text} {depth_val:.2f}m",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    1
                )

            # 2) YOLO 탐지
            results = self.model(frame, stream=True, conf=0.8, half=True)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2)//2
                    cy = (y1 + y2)//2
                    depth_val = depth_frame.get_distance(cx, cy)
                    if depth_val == 0:
                        continue

                    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], depth_val)
                    base_pt = self.transform_camera_to_base_link(X, Y, Z)
                    if base_pt is not None:
                        label = self.model.names[int(box.cls)]
                        conf  = float(box.conf.item())
                        rospy.loginfo(
                            f"YOLO Detected: {label}({conf:.2f}), base=({base_pt[0]:.3f},{base_pt[1]:.3f},{base_pt[2]:.3f})"
                        )

                        # elevator 모드에서 목표 라벨이면
                        if mode != "pickup" and label == target_floor:
                            self.object_found = True
                            self.latest_xyz = base_pt

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
                        cv2.putText(
                            frame,
                            f"{label} {conf:.2f} {depth_val:.2f}m",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255,0,0), 1
                        )

            cv2.imshow("Realsense YOLO/QR detection", frame)
            key = cv2.waitKey(3) & 0xFF
            if key == 27:  # ESC
                break

            # 만약 이미 목표를 찾았다면(예: self.object_found = True)
            if self.object_found and self.latest_xyz is not None:
                x_o, y_o, z_o = self.latest_xyz
                rospy.loginfo(f"Target acquired at: {x_o:.3f}, {y_o:.3f}, {z_o:.3f}")

                # 모드에 따라 동작 시도
                if mode == "pickup":
                    # pickup 모드
                    is_success = grasp_object(x_o + 0.01, y_o - 0.04, z_o)
                else:
                    # elevator 모드
                    is_success = press_number(x_o + 0.015, y_o + 0.03, z_o - 0.015)

                if is_success:
                    # 동작 성공이면 종료
                    rospy.loginfo("Action success! Exiting.")
                    return True
                else:
                    # 동작 실패면 다시 카메라로 좌표를 재탐색해야 하므로
                    rospy.logwarn("Action failed -> reacquire target!")
                    self.object_found = False  # 다시 while문에서 탐지하도록
                    self.latest_xyz = None
                    if mode == "pickup":
                        set_default_pose()
                        return False
                    else:
                        set_number_pose()
                        return False

            rate.sleep()

        return True

    def transform_camera_to_base_link(self, X, Y, Z):
        """
        camera_link (X, Y, Z)를 base_link로 변환
        """
        global tf_buffer
        try:
            transform = tf_buffer.lookup_transform(
                self.base_link,
                self.camera_link,
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            trans = transform.transform.translation
            rot   = transform.transform.rotation
            trans_vec = [trans.x, trans.y, trans.z]
            quat      = [rot.x, rot.y, rot.z, rot.w]

            mat = t.quaternion_matrix(quat)
            mat[0:3, 3] = trans_vec

            pt_cam  = np.array([X, Y, Z, 1.0])
            pt_base_link = mat @ pt_cam
            return pt_base_link[:3]
        except Exception as e:
            rospy.logwarn(f"transform_camera_to_base_link error: {e}")
            return None
