#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import rospy
import pyrealsense2 as rs
import numpy as np
import cv2
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


# ----------------------------
# MoveIt 및 로봇 팔 제어 함수
# ----------------------------
def transform_pose(target_frame, pose_stamped):
    """
    pose_stamped(어떤 frame_id) → target_frame 으로 TF 변환
    """
    global tf_buffer
    try:
        transform = tf_buffer.lookup_transform(
            target_frame,
            pose_stamped.header.frame_id,
            rospy.Time(0),
            rospy.Duration(2.0)
        )
        transformed_pose = do_transform_pose(pose_stamped, transform)
        return transformed_pose
    except Exception as e:
        rospy.logwarn(f"Transform failed: {e}")
        return None

def move_ee(px, py, pz, rx, ry, rz, rw):
    global move_group, error_pub
    pose_target = Pose()
    pose_target.position.x = px
    pose_target.position.y = py
    pose_target.position.z = pz
    pose_target.orientation.x = rx
    pose_target.orientation.y = ry
    pose_target.orientation.z = rz
    pose_target.orientation.w = rw

    move_group.set_pose_target(pose_target)
    success = move_group.go(wait=True)
    if not success:
        rospy.logwarn("No motion plan found.")
        error_message = "Error: Motion plan failed."
        rospy.loginfo(error_message)
        error_pub.publish(error_message)
    else:
        rospy.loginfo("Plan executed successfully!")

def move_Joint(q1, q2, q3, q4, q5, q6):
    global move_group
    joint_goal = move_group.get_current_joint_values()
    joint_goal_list = [q1, q2, q3, q4, q5, q6]
    if len(joint_goal) != len(joint_goal_list):
        rospy.logerr("Joint goal list size mismatch.")
        return
    for i in range(6):
        joint_goal[i] = joint_goal_list[i]
    move_group.go(joint_goal, wait=True)

def publish_position_open():
    pub = rospy.Publisher('/rh_p12_rn_position/command', Float64, queue_size=10)
    rospy.sleep(0.5)
    rospy.loginfo("Opening the gripper (data=0.0)")
    msg = Float64()
    msg.data = 0.0
    pub.publish(msg)
    rospy.sleep(0.5)
    rospy.loginfo("Gripper open command published once.")

def publish_position_close():
    pub = rospy.Publisher('/rh_p12_rn_position/command', Float64, queue_size=10)
    rospy.sleep(0.5)
    rospy.loginfo("Closing the gripper (data=0.7)")
    msg = Float64()
    msg.data = 0.7
    pub.publish(msg)
    rospy.sleep(0.5)
    rospy.loginfo("Gripper close command published once.")

def set_default_pose():
    rospy.loginfo("Moving to default pose for vertical grasp...")
    move_Joint(1.5783, -1.977, 1.651, -1.843, -1.592, 0.034)
    rospy.loginfo("Default pose reached.")

def set_home_pose():
    rospy.loginfo("Moving to home pose...")
    move_Joint(-1.57, -2.276, 1.924, -1.57, -1.57, 0)
    rospy.loginfo("Home pose reached.")

def move_to_above_object(x_odom, y_odom, z_odom):
    """
    odom 좌표에서 물체 위치를 받아 -> base_link 변환 후 -> 물체 위 0.2m로 이동
    """
    rospy.loginfo("[move_to_above_object] Called.")
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = 'odom'
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.pose.position.x = x_odom
    pose_stamped.pose.position.y = y_odom
    pose_stamped.pose.position.z = z_odom + 0.2
    pose_stamped.pose.orientation.x = 1
    pose_stamped.pose.orientation.y = 0
    pose_stamped.pose.orientation.z = 0
    pose_stamped.pose.orientation.w = 0

    transformed_pose = transform_pose('base_link', pose_stamped)
    if not transformed_pose:
        rospy.logerr("Failed to transform pose to ur3_base_link.")
        return

    px = transformed_pose.pose.position.x
    py = transformed_pose.pose.position.y
    pz = transformed_pose.pose.position.z
    ox = transformed_pose.pose.orientation.x
    oy = transformed_pose.pose.orientation.y
    oz = transformed_pose.pose.orientation.z
    ow = transformed_pose.pose.orientation.w

    rospy.loginfo("Moving above object => base_link: %.3f, %.3f, %.3f" % (px, py, pz))
    move_ee(px, py, pz, ox, oy, oz, ow)

def grasp_object(x_odom, y_odom, z_odom):
    """
    odom 좌표 (x, y, z)에서 물체를 잡는 시퀀스
    (각 단계마다 input("Press Enter")로 대기)
    """
    rospy.loginfo("[grasp_object] Starting grasp sequence...")

    # Step 1: Default Pose
    input("Press Enter to move to default pose...")
    set_default_pose()

    # Step 2: Move to position above object
    input("Press Enter to move above the object...")
    move_to_above_object(x_odom, y_odom, z_odom)

    # Step 3: Move to grasp position
    input("Press Enter to move to grasp position...")
    rospy.loginfo("Transforming position to ur3_base_link...")

    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = 'odom'
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.pose.position.x = x_odom
    pose_stamped.pose.position.y = y_odom
    pose_stamped.pose.position.z = z_odom
    pose_stamped.pose.orientation.x = 1
    pose_stamped.pose.orientation.y = 0
    pose_stamped.pose.orientation.z = 0
    pose_stamped.pose.orientation.w = 0

    transformed_pose = transform_pose('base_link', pose_stamped)
    if not transformed_pose:
        rospy.logerr("Failed to transform pose to ur3_base_link.")
        return

    px = transformed_pose.pose.position.x
    py = transformed_pose.pose.position.y
    pz = transformed_pose.pose.position.z
    ox = transformed_pose.pose.orientation.x
    oy = transformed_pose.pose.orientation.y
    oz = transformed_pose.pose.orientation.z
    ow = transformed_pose.pose.orientation.w

    move_ee(px, py, pz, ox, oy, oz, ow)

    # Step 4: Close gripper
    input("Press Enter to close the gripper...")
    publish_position_close()

    # Step 5: Return to home pose
    input("Press Enter to return to home pose...")
    set_home_pose()

    # Step 6: Open gripper
    input("Press Enter to open the gripper and release the object...")
    publish_position_open()

    # Step 7: Default pose
    input("Press Enter to return to default pose...")
    set_default_pose()

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


if __name__ == "__main__":
    try:
        # 전역 변수에 값을 할당하기 위해 global 선언
        global tf_buffer
        global tf_listener
        global error_pub
        global move_group

        # ROS 노드 및 MoveIt 초기화
        rospy.init_node("yolo_wait_pick_node", anonymous=True)
        roscpp_initialize(sys.argv)

        tf_buffer = Buffer()
        tf_listener = TransformListener(tf_buffer)

        error_pub = rospy.Publisher('/error_topic', String, queue_size=10)

        robot = RobotCommander()
        scene = PlanningSceneInterface()
        group_name = "ur3_manipulator"  # 로봇 환경에 맞게 수정
        move_group = MoveGroupCommander(group_name)

        # 메인 클래스 실행
        node = YoloDepthAndWait()
        node.run()

    except rospy.ROSInterruptException:
        pass
    finally:
        roscpp_shutdown()