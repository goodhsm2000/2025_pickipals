#!/usr/bin/env python3

import rospy
import tf2_ros
import tf
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

class DepthToWorld:
    def __init__(self):
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.pub = rospy.Publisher('/world_coordinates', PointStamped, queue_size=10)
        
        rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        
    def depth_callback(self, msg):
        # Depth 이미지를 OpenCV 형식으로 변환
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        
        # 특정 픽셀의 Depth 데이터 가져오기 (예: 중앙 픽셀)
        u, v = depth_image.shape[1] // 2, depth_image.shape[0] // 2
        depth = depth_image[v, u]
        
        if depth == 0 or np.isnan(depth) or np.isinf(depth):
            rospy.logwarn("Invalid depth at center pixel")
            return
        
        # 카메라 내부 파라미터 (정확한 값으로 수정 필요)
        fx, fy, cx, cy = 570.34, 570.34, 319.5, 239.5
        
        # 카메라 좌표로 변환
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth

        rospy.loginfo(f"Camera coordinates: X={X}, Y={Y}, Z={Z}")

        try:
            # 카메라 좌표계에서 odom 좌표계로 변환 정보 가져오기
            trans = self.tf_buffer.lookup_transform("odom", "camera_link", rospy.Time(0), rospy.Duration(1.0))

            # 변환 정보 추출
            translation = trans.transform.translation
            rotation = trans.transform.rotation

            # 변환 행렬 생성
            trans_vec = [translation.x, translation.y, translation.z]
            quat = [rotation.x, rotation.y, rotation.z, rotation.w]
            rotation_matrix = tf.transformations.quaternion_matrix(quat)
            transformation_matrix = rotation_matrix.copy()
            transformation_matrix[0:3, 3] = trans_vec

            # 카메라 좌표의 점 (동차 좌표계)
            point_camera = np.array([X, Y, Z, 1])

            # 월드 좌표로 변환
            point_world = np.dot(transformation_matrix, point_camera)

            X_w, Y_w, Z_w = point_world[:3]

            rospy.loginfo(f"World coordinates: X={X_w}, Y={Y_w}, Z={Z_w}")

            # 변환된 월드 좌표를 퍼블리시
            point_world_msg = PointStamped()
            point_world_msg.header.frame_id = "odom"
            point_world_msg.header.stamp = rospy.Time.now()
            point_world_msg.point.x = X_w
            point_world_msg.point.y = Y_w
            point_world_msg.point.z = Z_w

            self.pub.publish(point_world_msg)

        except tf2_ros.TransformException as e:
            rospy.logerr(f"Transform failed: {e}")

if __name__ == '__main__':
    rospy.init_node('depth_to_world')
    DepthToWorld()
    rospy.spin()
