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

# === 새로 추가: Marker 시각화 ===
from visualization_msgs.msg import Marker

##################################################
# 전역 함수들 (MoveIt 제어, Pose 변환 등)
##################################################

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
        msg = "Error: Motion plan failed."
        rospy.loginfo(msg)
        error_pub.publish(msg)
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
    rospy.loginfo("Opening gripper -> 0.0")
    pub.publish(0.0)
    rospy.sleep(0.5)


def publish_position_close():
    pub = rospy.Publisher('/rh_p12_rn_position/command', Float64, queue_size=10)
    rospy.sleep(0.5)
    rospy.loginfo("Closing gripper -> 0.7")
    pub.publish(0.7)
    rospy.sleep(0.5)


def set_default_pose():
    rospy.loginfo("Moving to default pose...")
    move_Joint(1.5783, -1.977, 1.651, -1.843, -1.592, 0.034)
    rospy.loginfo("Default pose reached.")


def set_home_pose():
    rospy.loginfo("Moving to home pose...")
    move_Joint(-1.57, -2.276, 1.924, -1.57, -1.57, 0)
    rospy.loginfo("Home pose reached.")


def move_to_above_object(x_odom, y_odom, z_odom):
    rospy.loginfo("[move_to_above_object] Called.")
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = 'odom'
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.pose.position.x = x_odom
    pose_stamped.pose.position.y = y_odom
    pose_stamped.pose.position.z = z_odom + 0.2
    pose_stamped.pose.orientation.w = 1.0  # 나머지 x,y,z=0

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
    rospy.loginfo("Moving above object => base_link: %.3f,%.3f,%.3f" % (px, py, pz))

    move_ee(px, py, pz, ox, oy, oz, ow)


def grasp_object(x_odom, y_odom, z_odom):
    rospy.loginfo("[grasp_object] Start grasp sequence...")

    set_default_pose()
    move_to_above_object(x_odom, y_odom, z_odom)

    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = 'odom'
    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.pose.position.x = x_odom
    pose_stamped.pose.position.y = y_odom
    pose_stamped.pose.position.z = z_odom
    pose_stamped.pose.orientation.w = 1.0

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

    publish_position_close()
    set_home_pose()
    publish_position_open()
    set_default_pose()


##################################################
# YoloDepthAndWait
##################################################

class YoloDepthAndWait:
    def __init__(self):
        rospy.loginfo("[YoloDepthAndWait] init...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # YOLO 모델
        self.model = YOLO('/home/yujin/Downloads/content/runs/detect/train/weights/best.pt')

        self.camera_link = "camera_link"
        self.odom_link   = "odom"

        # PoseStamped 퍼블리셔
        self.pose_pub = rospy.Publisher("/detected_object_pose", PoseStamped, queue_size=10)
        
        # Marker 퍼블리셔 (Arrow 시각화)
        self.marker_pub = rospy.Publisher("/detected_object_marker", Marker, queue_size=10)

        # 처음 인식된 물체만 사용하기 위한 플래그
        self.object_finalized = False
        self.final_xyz = None  # (x, y, z) in odom

        self.object_found = False
        self.latest_odom_xyz = None

    def run(self):
        rate = rospy.Rate(15)
        while not rospy.is_shutdown():
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # 만약 이미 첫 물체 좌표를 확보했다면, 마커만 계속 퍼블리시하고, 인식 로직은 건너뜀
            if self.object_finalized:
                self.publish_marker_once()
                rate.sleep()
                continue

            frame = np.asanyarray(color_frame.get_data())
            intr  = depth_frame.profile.as_video_stream_profile().get_intrinsics()

            # (A) QR 코드
            qr_codes = decode(frame)
            for qr in qr_codes:
                x, y, w, h = qr.rect
                cx = x + w//2
                cy = y + h//2
                depth_val = depth_frame.get_distance(cx, cy)

                X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], depth_val)
                odom_pt = self.transform_camera_to_odom(X, Y, Z)
                if odom_pt is not None and not self.object_finalized:
                    txt = qr.data.decode('utf-8')
                    rospy.loginfo(f"[QR] {txt}, Odom=({odom_pt[0]:.3f},{odom_pt[1]:.3f},{odom_pt[2]:.3f})")
                    self.object_found = True
                    self.latest_odom_xyz = odom_pt

                    # PoseStamped로 퍼블리시
                    self.publish_detected_pose(odom_pt)

                    # 첫 물체 좌표를 final_xyz에 저장, finalized
                    self.final_xyz = odom_pt
                    self.object_finalized = True

                # 시각화
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame,
                            f"{qr.data.decode('utf-8')} {depth_val:.2f}m",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # (B) YOLO 탐지 (아직 object_finalized 아니면 수행)
            if not self.object_finalized:
                results = self.model(frame, stream=True)
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = (x1 + x2)//2
                        cy = (y1 + y2)//2
                        depth_val = depth_frame.get_distance(cx, cy)

                        label = self.model.names[int(box.cls)]
                        conf  = float(box.conf.item())

                        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
                        cv2.putText(
                            frame,
                            f"{label} {conf:.2f} {depth_val:.2f}m",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1
                        )

                        # odom 변환
                        X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], depth_val)
                        odom_pt = self.transform_camera_to_odom(X, Y, Z)
                        if odom_pt is not None and not self.object_finalized:
                            rospy.loginfo(
                                f"[YOLO] {label}({conf:.2f}), Odom=({odom_pt[0]:.3f},{odom_pt[1]:.3f},{odom_pt[2]:.3f})"
                            )
                            self.object_found = True
                            self.latest_odom_xyz = odom_pt

                            # PoseStamped 퍼블리시
                            self.publish_detected_pose(odom_pt)

                            # 첫 물체 좌표 저장 + finalized
                            self.final_xyz = odom_pt
                            self.object_finalized = True

            cv2.imshow("Realsense YOLO/QR detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

            # 기존 grasp_object() 부분은 제거하거나 필요 시 사용
            # 만약 첫 물체 잡기를 원한다면, self.object_finalized된 후 grasp_object 호출 가능
            # 여기서는 marker만 표시하므로 주석 처리
            # if self.object_found and self.latest_odom_xyz is not None:
            #     x_o, y_o, z_o = self.latest_odom_xyz
            #     grasp_object(x_o, y_o, z_o)
            # else:
            #     rospy.logwarn("No object found yet.")

            rate.sleep()

        self.pipeline.stop()
        cv2.destroyAllWindows()

    def transform_camera_to_odom(self, X, Y, Z):
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

    def publish_detected_pose(self, odom_pt):
        """
        odom_pt(x,y,z)를 PoseStamped로 변환해
        /detected_object_pose 토픽에 퍼블리시 (선택적).
        """
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "odom"
        pose_msg.pose.position.x = odom_pt[0]
        pose_msg.pose.position.y = odom_pt[1]
        pose_msg.pose.position.z = odom_pt[2]
        pose_msg.pose.orientation.w = 1.0

        self.pose_pub.publish(pose_msg)
        rospy.loginfo("Publish /detected_object_pose: (%.3f,%.3f,%.3f)" % (odom_pt[0], odom_pt[1], odom_pt[2]))

    def publish_marker_once(self):
        """
        final_xyz(odom 좌표)에 대한 Arrow 마커를 계속 퍼블리시.
        frame_id = odom
        """
        if self.final_xyz is None:
            return  # 아무것도 안 함

        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "detected_object"
        marker.id = 1
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # position (화살표 기준)
        marker.pose.position.x = self.final_xyz[0]
        marker.pose.position.y = self.final_xyz[1]
        marker.pose.position.z = self.final_xyz[2]
        # 화살표는 방향 지정 (ex: odom Z축을 위로, X축을 앞으로)
        # 간단히 회전 없이 w=1.0 (정방향)
        marker.pose.orientation.w = 1.0

        # 크기 (scale.x=길이, scale.y=굵기, scale.z=굵기)
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        # 색 (RGBA)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # 퍼블리시
        self.marker_pub.publish(marker)


##################################################
# 메인 실행
##################################################

if __name__ == "__main__":
    try:
        rospy.init_node("yolo_wait_pick_node", anonymous=True)
        roscpp_initialize(sys.argv)

        global tf_buffer, tf_listener, error_pub, move_group
        tf_buffer = Buffer()
        tf_listener = TransformListener(tf_buffer)

        error_pub = rospy.Publisher('/error_topic', String, queue_size=10)

        robot = RobotCommander()
        scene = PlanningSceneInterface()
        move_group = MoveGroupCommander("ur3_manipulator")

        node = YoloDepthAndWait()
        node.run()

    except rospy.ROSInterruptException:
        pass
    finally:
        roscpp_shutdown()
