#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import UInt16, Int16, UInt16MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


import time
import cv2
import numpy as np

import onnx
import UltraFaceDetector.vision.utils.box_utils_numpy as box_utils
from onnxruntime import InferenceSession

# 얼굴 검출 관련 설정
face_min_width = 20
face_min_height = 20
threshold = 0.7

# 회전 제어 관련 설정
current_angular_velocity = 0.2

class FaceDetectorAndTurn:
    def __init__(self):
        rospy.init_node('face_detection_and_turn')

        # Publisher for velocity commands
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)
        # /scan 구독자 생성
        self.scan_subscriber = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        # Publisher for processed images and face detection flag
        self.image_pub = rospy.Publisher('/facedetection/rgb_image', Image, queue_size=10)
        self.face_detection_pub = rospy.Publisher('/facedetection/detect_face', UInt16, queue_size=10)
        
        # Initialize Twist message for velocity
        self.twist = Twist()
        self.face_detected = False
        # 이동 플래그 설정
        self.moving = False

        # Set up rate for continuous publishing
        self.rate = rospy.Rate(10)  # 10 Hz

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Load ONNX model for face detection
        label_path = "/home/turtle/catkin_ws/src/turtlebot_control/scripts/UltraFaceDetector/models/voc-model-labels.txt"
        onnx_path = "/home/turtle/catkin_ws/src/turtlebot_control/scripts/UltraFaceDetector/models/onnx/version-RFB-320.onnx"
        self.class_names = [name.strip() for name in open(label_path).readlines()]
        self.ort_session = InferenceSession(onnx_path)
        self.input_name = self.ort_session.get_inputs()[0].name

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)  # set Width
        self.cap.set(4, 480)  # set Height

    def predict(self, width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = box_utils.hard_nms(box_probs,
                                           iou_threshold=iou_threshold,
                                           top_k=top_k)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

    def face_detection_callback(self, msg):
        # Update the face detection flag based on the message data
        if msg.data == 1:
            self.face_detected = True
        else:
            self.face_detected = False

    def scan_callback(self, msg):
         if self.face_detected and not self.moving:
            # 250번째 요소의 거리 값을 가져옴
            self.distance = msg.ranges[250] - 0.1
            # 거리 값 확인 (예: 최소 거리 및 최대 거리 체크)
            if distance > 0.2 and distance < msg.range_max:
                rospy.loginfo(f"Moving forward by {distance} meters")

                # 선형 속도를 설정 (0.2 m/s)
                self.vel_msg.linear.x = 0.2
                self.vel_msg.angular.z = 0.0

                # 이동 시간 계산 (거리 / 속도)
                time_to_move = distance / self.vel_msg.linear.x
                
                # 현재 시간 기록
                start_time = rospy.Time.now().to_sec()
                
                # 이동 플래그 설정
                self.moving = True

                # 주어진 시간 동안 전진
                while rospy.Time.now().to_sec() - start_time < time_to_move:
                    self.velocity_publisher.publish(self.vel_msg)
                    rospy.sleep(0.1)
                
                # 정지 명령
                self.vel_msg.linear.x = 0.0
                self.velocity_publisher.publish(self.vel_msg)
                rospy.loginfo("TurtleBot has stopped")
                
                # 한 번만 이동하도록 구독자 종료
                self.scan_subscriber.unregister()
            
            else:
                rospy.logwarn(f"Invalid distance: {distance} meters")
            self.face_detected = False
        
    def run(self):
        # Subscriber for face detection flag
        rospy.Subscriber('/facedetection/detect_face', UInt16, self.face_detection_callback)

        while not rospy.is_shutdown():
            ret, orig_image = self.cap.read()
            if orig_image is None:
                rospy.loginfo("No image from camera")
                break

            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (320, 240))
            image_mean = np.array([127, 127, 127])
            image = (image - image_mean) / 128
            image = np.transpose(image, [2, 0, 1])
            image = np.expand_dims(image, axis=0)
            image = image.astype(np.float32)
            confidences, boxes = self.ort_session.run(None, {self.input_name: image})
            boxes, labels, probs = self.predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)

            if boxes.shape[0] != 0:
                face_cnt = 0
                if (boxes[face_cnt, 2] - boxes[face_cnt, 0]) > face_min_width and (boxes[face_cnt, 3] - boxes[face_cnt, 1]) > face_min_height and probs[face_cnt] >= 0.97:
                    box = boxes[face_cnt, :]
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    face_center_x = (box[0] + box[2]) / 2
                    img_width = orig_image.shape[1]
                    left_third = img_width / 4
                    right_third = 3 * img_width / 4

                    if left_third < face_center_x < right_third:
                        self.face_detected = True
                    else:
                        self.face_detected = False

                    label = "{}:{:.0f}%".format(self.class_names[labels[face_cnt]], probs[face_cnt] * 100)
                    box_size_text = "Width: {}, Height: {}".format(width, height)

                    cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
                    cv2.putText(orig_image, label, (box[0] - 40, box[3] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    cv2.putText(orig_image, box_size_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    self.face_detected = False
            else:
                self.face_detected = False
            img_pub = self.bridge.cv2_to_imgmsg(orig_image, encoding="passthrough")
            self.image_pub.publish(img_pub)
            self.face_detection_pub.publish(UInt16(1 if self.face_detected else 0))
            self.vel_pub.publish(self.twist)
            self.rate.sleep()
            if self.face_detected and not self.moving:
                self.scan_subscriber = rospy.Subscriber('/scan', LaserScan, self.scan_callback)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        controller = FaceDetectorAndTurn()
        controller.run()
    except rospy.ROSInterruptException:
        pass
