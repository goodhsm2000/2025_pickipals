#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import rospy
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import threading 
from pyzbar.pyzbar import decode
from ultralytics import YOLO

# === ROS / MoveIt 관련 임포트 ===
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

# 전역 오류 상태 변수
motion_plan_error = 0

def transform_pose(target_frame, pose):
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
        motion_plan_error = 1
        rospy.sleep(0.1)
        motion_plan_error = 0
    else:
        rospy.loginfo("Plan executed successfully!")

def move_ee_with_current_orientation(Px, Py, Pz):
    """
    현재 그리퍼의 orientation(자세)을 유지하면서 목표 위치로 이동합니다.
    """
    global motion_plan_error    
    current_pose = move_group.get_current_pose().pose

    # 현재 자세
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
        motion_plan_error = 1
        rospy.sleep(0.1)
        motion_plan_error = 0
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
    rospy.loginfo("Moving to updown pose...")
    move_Joint(1.5783, -1.573, 0.522, -1.900, -1.429, 3.141)
    rospy.loginfo("Updown pose reached.")    

def set_number_pose():
    rospy.loginfo("Moving to number pose...")
    move_Joint(-0.052, -1.167, -1.243, -0.719, 1.594, 0.038)
    rospy.loginfo("Number pose reached.")    

def move_to_above_object(x, y, z):
    rospy.loginfo("Transforming position to ur3_base_link...")

    # Transform odom 좌표를 base_link 좌표로 변환
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = 'odom'
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.pose.position.x = x
    pose_stamped.pose.position.y = y
    pose_stamped.pose.position.z = z + 0.2  # 물체 위쪽
    pose_stamped.pose.orientation.x = 1
    pose_stamped.pose.orientation.y = 0
    pose_stamped.pose.orientation.z = 0
    pose_stamped.pose.orientation.w = 0

    transformed_pose = transform_pose('base_link', pose_stamped)
    if not transformed_pose:
        rospy.logerr("Failed to transform pose to ur3_base_link.")
        return

    rospy.loginfo(f"Moving above object: {transformed_pose.pose.position.x}, "
                  f"{transformed_pose.pose.position.y}, {transformed_pose.pose.position.z}")

    move_ee(
        transformed_pose.pose.position.x,
        transformed_pose.pose.position.y,
        transformed_pose.pose.position.z,
        transformed_pose.pose.orientation.x,
        transformed_pose.pose.orientation.y,
        transformed_pose.pose.orientation.z,
        transformed_pose.pose.orientation.w
    )

def move_to_front_object(x, y, z):
    rospy.loginfo("Transforming position to ur3_base_link...")

    # Transform odom 좌표를 base_link 좌표로 변환 (물체 정면)
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

    rospy.loginfo(f"Moving front of object: {transformed_pose.pose.position.x}, "
                  f"{transformed_pose.pose.position.y}, {transformed_pose.pose.position.z}")

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
    rospy.loginfo("Grasping the object...")
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
    rospy.loginfo("Transforming position to grasp object...")
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

    # Step 4: Grasp
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

def press_updown(x, y, z):
    rospy.loginfo("Starting updown sequence...")
    publish_position_close()
    time.sleep(5)

    set_number_pose()
    time.sleep(10)

    move_to_front_object(x, y, z)
    time.sleep(10)

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

    rospy.loginfo("Pressing updown button...")
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

    rospy.loginfo("Pressing number button...")
    move_ee_with_current_orientation(
        transformed_pose.pose.position.x,
        transformed_pose.pose.position.y,
        transformed_pose.pose.position.z,
    )

# ======================================================
# ActionExecutor 클래스 (grasp, press_number, press_updown 기능)
# ======================================================
class ActionExecutor:
    def __init__(self, node):
        """
        node: YoloDepthAndWait 클래스의 인스턴스 (카메라와 물체 좌표 검출)
        """
        self.node = node

    def grasp_object(self):
        if self.node.object_found and self.node.latest_odom_xyz is not None:
            x, y, z = self.node.latest_odom_xyz
            grasp_object(x, y, z)
            return True
        else:
            rospy.logwarn("No object found yet for grasping.")
            return False

    def press_updown(self):
        if self.node.object_found and self.node.latest_odom_xyz is not None:
            x, y, z = self.node.latest_odom_xyz
            press_updown(x, y, z)
            return True
        else:
            rospy.logwarn("No object found yet for press_updown.")
            return False

    def press_number(self):
        if self.node.object_found and self.node.latest_odom_xyz is not None:
            x, y, z = self.node.latest_odom_xyz
            press_number(x, y, z)
            return True
        else:
            rospy.logwarn("No object found yet for press_number.")
            return False

# ======================================================
# YoloDepthAndWait 클래스 (카메라 루프 및 좌표 변환 처리)
# ======================================================
# ----------------------------
#  YOLO + Realsense 노드
# ----------------------------
class YoloDepthAndWait:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # YOLO 로드 (사용자 환경에 맞춰 경로 수정)
        self.model = YOLO('/home/yujin/Downloads/content/runs/detect/train/weights/best.pt')

        # frame id
        self.camera_link = "camera_link"
        self.odom_link = "odom"

        self.object_found = False       # 한 번 물체를 찾았으면 True
        self.latest_odom_xyz = None     # 인식된 물체의 odom 좌표

    def run(self):
        """
        계속해서 카메라 프레임을 보고, QR/YOLO로 물체 인식.
        Enter 키(ASCII=13)가 눌리면, grasp_object(...)를 호출.
        """
        rate = rospy.Rate(15)

        while not rospy.is_shutdown():
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
                X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], depth_val)
                odom_pt = self.transform_camera_to_odom(X, Y, Z)
                if odom_pt is not None:
                    txt = qr.data.decode('utf-8')
                    rospy.loginfo(
                        f"QR Detected: {txt}, Odom=({odom_pt[0]:.3f},{odom_pt[1]:.3f},{odom_pt[2]:.3f})"
                    )
                    # 갱신
                    self.object_found = True
                    self.latest_odom_xyz = odom_pt

                # 디스플레이 표시
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(
                    frame,
                    f"{qr.data.decode('utf-8')} {depth_val:.2f}m",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,255,0), 1
                )

            # 2) YOLO 탐지
            results = self.model(frame, stream=True)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2)//2
                    cy = (y1 + y2)//2
                    depth_val = depth_frame.get_distance(cx, cy)

                    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], depth_val)
                    odom_pt = self.transform_camera_to_odom(X, Y, Z)
                    if odom_pt is not None:
                        label = self.model.names[int(box.cls)]
                        conf  = float(box.conf.item())
                        rospy.loginfo(
                            f"YOLO Detected: {label}({conf:.2f}), "
                            f"Odom=({odom_pt[0]:.3f},{odom_pt[1]:.3f},{odom_pt[2]:.3f})"
                        )
                        self.object_found = True
                        self.latest_odom_xyz = odom_pt

                    # 디스플레이 표기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
                    cv2.putText(
                        frame,
                        f"{label} {conf:.2f} {depth_val:.2f}m",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255,0,0), 1
                    )

            # 3) 시각화
            cv2.imshow("Realsense YOLO/QR detection", frame)
            key = cv2.waitKey(1) & 0xFF
            # ESC 등으로 종료
            if key == 27:
                break

            # 4) 사용자가 Enter 키(ASCII=13) 누르면, 가장 최근 인식된 물체를 잡기
            # if key == 13:  # Enter
            if self.object_found and self.latest_odom_xyz is not None:
                x_o, y_o, z_o = self.latest_odom_xyz
                grasp_object(x_o, y_o, z_o)
            else:
                rospy.logwarn("No object found yet.")

            rate.sleep()

        # 종료
        self.pipeline.stop()
        cv2.destroyAllWindows()

    def transform_camera_to_odom(self, X, Y, Z):
        """
        camera_link 좌표(X, Y, Z)를 odom 좌표로 변환 -> (x, y, z)
        """
        global tf_buffer
        try:
            transform = tf_buffer.lookup_transform(
                self.odom_link,
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
            pt_odom = mat @ pt_cam
            return pt_odom[:3]
        except Exception as e:
            rospy.logwarn(f"transform_camera_to_odom error: {e}")
            return None

# ======================================================
# 메인 함수
# ======================================================
if __name__ == "__main__":
    try:
        rospy.init_node("yolo_main_node", anonymous=True)
        roscpp_initialize(sys.argv)

        global tf_buffer, tf_listener, error_pub, move_group
        tf_buffer = Buffer()
        tf_listener = TransformListener(tf_buffer)

        error_pub = rospy.Publisher('/error_topic', String, queue_size=10)

        robot = RobotCommander()
        scene = PlanningSceneInterface()
        move_group = MoveGroupCommander("ur3_manipulator")

        # YoloDepthAndWait 객체 생성 (카메라 루프를 위한 객체)
        node = YoloDepthAndWait()

        # ActionExecutor 객체 생성 (grasp, press_number, press_updown 기능 제공)
        action_executor = ActionExecutor(node)

        # 카메라 루프를 별도 스레드에서 실행 (물체 좌표를 지속적으로 업데이트)
        camera_thread = threading.Thread(target=node.run_camera_loop)
        camera_thread.start()

        rospy.loginfo("Waiting for object detection...")
        # 단순 예시로, 물체가 검출될 때까지 대기
        while not rospy.is_shutdown() and not node.object_found:
            rospy.sleep(0.1)

        rospy.loginfo("Object detected, executing grasp sequence...")
        action_executor.grasp_object()

        # grasp가 완료된 후 종료 (필요에 따라 다른 행동 실행 가능)
        rospy.loginfo("Grasp sequence finished.")

    except rospy.ROSInterruptException:
        pass
    finally:
        roscpp_shutdown()
