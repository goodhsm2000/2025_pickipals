#!/usr/bin/env python

import rospy
from std_msgs.msg import UInt16, Int16, UInt16MultiArray, Bool, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from scipy import signal

import onnx
import UltraFaceDetector.vision.utils.box_utils_numpy as box_utils
from caffe2.python.onnx import backend
import onnxruntime as ort

import time
import math
import message_filters
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry  # Odometry 메시지 임포트

# 초기 변수 설정
face_min_width = 50
face_min_height = 50
init_delay_proc_time = 10
during_proc_time = 60
sampling_rate = 30  # Camera FPS
array_size = 300
bridge = CvBridge()

# 기본 신호 처리 변수 설정
fs = 30  # 샘플링 주파수
cffs = fs / 2
low = 0.75
high = 4
nfft = 2048
b, a = signal.butter(4, [low / cffs, high / cffs], btype='bandpass')

# RGB 배열 초기화
redArray = np.zeros(array_size)
greenArray = np.zeros(array_size)
blueArray = np.zeros(array_size)
timeArray = np.zeros(array_size)
dataArray = np.zeros((3, array_size))

# 상태 머신을 위한 상태 변수
STATE_IDLE = 'IDLE'  # 대기 상태
STATE_ROTATING = 'ROTATING'
STATE_MOVING = 'MOVING'
STATE_MEASURING = 'MEASURING'
state = STATE_IDLE  # 초기 상태를 IDLE로 설정

# Odometry 관련 변수
current_position = None
start_position = None
target_distance = 0.0

# 얼굴 검출 및 거리 측정 함수
def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
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
                                      top_k=top_k,
                                      )
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

burst_face_detection_Flag = 0
start_face_flag = 0
firstDetectionFlag = 0

# 얼굴 감지 플래그 콜백 함수
def face_flag_callback(input_data):
    global start_face_flag, burst_face_detection_Flag, firstDetectionFlag
    start_face_flag = input_data.data
    burst_face_detection_Flag = 1
    firstDetectionFlag = 0
    rospy.loginfo("start_face_flag = %d" % (start_face_flag))

# 얼굴 중심의 깊이 추출 함수
def get_depth_at_face_center(depth_image, face_box):
    x_center = (face_box[0] + face_box[2]) // 2
    y_center = (face_box[1] + face_box[3]) // 2
    return depth_image[y_center, x_center] / 1000.0  # 깊이 값을 미터로 변환

# 모델 로드
label_path = "/home/turtle/catkin_ws/src/turtlebot_control/scripts/UltraFaceDetector/models/voc-model-labels.txt"
onnx_path = "/home/turtle/catkin_ws/src/turtlebot_control/scripts/UltraFaceDetector/models/onnx/version-RFB-320.onnx"
class_names = [name.strip() for name in open(label_path).readlines()]
predictor = onnx.load(onnx_path)
onnx.checker.check_model(predictor)
# onnx.helper.printable_graph(predictor.graph)  # 필요시 활성화
predictor = backend.prepare(predictor, device="CPU")
ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name

# ROS 퍼블리셔 및 서브스크라이버 설정
pub1 = rospy.Publisher('/facedetection/rgb_image', Image, queue_size=10)
pub2 = rospy.Publisher('/facedetection/detect_face', UInt16, queue_size=10)
pub3 = rospy.Publisher('/facedetection/ROI_image', Image, queue_size=10)
pub4 = rospy.Publisher('/facedetection/roi_position_x', Int16, queue_size=10)
pub5 = rospy.Publisher('/facedetection/roi_position_y', Int16, queue_size=10)
pub6 = rospy.Publisher('/facedetection/roi_size', UInt16, queue_size=10)
HR_finsh_pub = rospy.Publisher('/facedetection/HR_finsh_flag', UInt16, queue_size=10)
sub5 = rospy.Subscriber('/facedetection/HR_start_flag', UInt16, face_flag_callback)

# 로봇 제어를 위한 cmd_vel 퍼블리셔 수정
cmd_vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)

# Odometry 서브스크라이버 설정
def odom_callback(msg):
    global current_position
    current_position = msg.pose.pose.position

odom_sub = rospy.Subscriber('/odom', Odometry, odom_callback)

# 상태 퍼블리셔 설정
state_pub = rospy.Publisher('/astra_twist_ppgi/state', String, queue_size=10)

# 제어 토픽 서브스크라이버 콜백 함수
def control_callback(msg):
    global state
    if msg.data:
        if state == STATE_IDLE:
            state = STATE_ROTATING
            state_pub.publish("ROTATING")
            rospy.loginfo("Received start command. Switching to ROTATING state.")
    else:
        if state != STATE_IDLE:
            state = STATE_IDLE
            stop_robot()
            state_pub.publish("IDLE")
            rospy.loginfo("Received stop command. Switching to IDLE state.")

# 제어 토픽 서브스크라이버 설정
control_sub = rospy.Subscriber('/control_astra_twist_ppgi', Bool, control_callback)

# 로봇 회전 및 직진 제어 함수
def rotate_robot(speed):
    twist = Twist()
    twist.angular.z = speed
    cmd_vel_pub.publish(twist)

def move_robot(speed):
    twist = Twist()
    twist.linear.x = speed
    cmd_vel_pub.publish(twist)

def stop_robot():
    twist = Twist()
    cmd_vel_pub.publish(twist)

# PPGI 기반 심박수 계산 함수
def calculate_heart_rate():
    # POS 알고리즘에 따른 신호 처리
    try:
        CnInv = np.linalg.inv(np.diag(np.mean(dataArray, axis=1)))
    except np.linalg.LinAlgError:
        rospy.logwarn("Singular matrix encountered while calculating heart rate.")
        return 0.0
    Cn = np.dot(CnInv, dataArray)
    SS = np.array([[0, 1, -1], [-2, 1, 1]])
    AA = np.dot(SS, Cn)
    AA_std = np.std(AA[0, :]) / np.std(AA[1, :])
    Pn = np.array([1, AA_std])
    P = np.dot(Pn, AA)
    
    # 필터 적용
    filtered_PPG = signal.lfilter(b, a, P)
    cut_PPG = filtered_PPG[2 * fs:]
    
    # FFT를 이용하여 주파수 최대값을 찾고 BPM 계산
    fft_out = np.fft.fft(cut_PPG, n=nfft)
    f0 = np.fft.fftfreq(nfft, d=1 / fs)
    heart_rate_freq = f0[np.argmax(np.abs(fft_out[:nfft // 2]))]
    
    return heart_rate_freq * 60  # BPM 변환

# Astra Camera Image Callback
def image_callback(rgb_msg, depth_msg):
    try:
        orig_image = bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth_image = bridge.imgmsg_to_cv2(depth_msg, "32FC1")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
        return
    
    process_image(orig_image, depth_image)

# 얼굴 검출 및 PPGI 측정 함수
def process_image(orig_image, depth_image):
    global firstDetectionFlag, burst_face_detection_Flag, start_face_flag, state
    global dataArray, timeArray, redArray, greenArray, blueArray
    global start_position, target_distance
    
    threshold = 0.7
    face_detection_Flag = 0

    if state == STATE_IDLE:
        # IDLE 상태에서는 이미지 처리를 하지 않음
        try:
            img_pub = bridge.cv2_to_imgmsg(orig_image, encoding="bgr8")
            pub1.publish(img_pub)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error while publishing image: {0}".format(e))
        return

    # 이미지 처리
    image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (320, 240))
    image_normalized = (image_resized - np.array([127, 127, 127])) / 128
    image_transposed = np.transpose(image_normalized, [2, 0, 1])
    input_image = np.expand_dims(image_transposed, axis=0).astype(np.float32)

    confidences, boxes = ort_session.run(None, {input_name: input_image})
    boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, 0.7)
    if boxes.shape[0] != 0:
        face_detection_Flag = 1
        face_cnt = 0
        if (boxes[face_cnt, 2] - boxes[face_cnt, 0]) > face_min_width and \
           (boxes[face_cnt, 3] - boxes[face_cnt, 1]) > face_min_height and \
           probs[face_cnt] >= 0.97:
            firstDetectionFlag = 1
            face_detection_Flag = 1
            box = boxes[face_cnt, :]
            depth_value = get_depth_at_face_center(depth_image, box)  # 얼굴 중심의 깊이 값
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            label = "{}:{:.0f}% Depth: {:.2f}m".format(class_names[labels[face_cnt]], probs[face_cnt]*100, depth_value)
            cv2.putText(orig_image, label, (box[0] - 40, box[3] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            
            # 화면 가로 크기 계산
            image_width = orig_image.shape[1]
            left_third = image_width / 3
            right_third = 2 * image_width / 3

            # 얼굴 박스의 가로 중심 계산
            face_center_x = (box[0] + box[2]) / 2

            # 얼굴 박스가 중앙 3분의 1 구간 내에 있는지 확인
            if left_third <= face_center_x <= right_third:
                if state == STATE_ROTATING:
                    # 회전을 멈추고 이동 준비
                    stop_robot()
                    state = STATE_MOVING
                    state_pub.publish("MOVING")
                    rospy.loginfo("Face detected in central third. Stopping rotation.")
                    
                    # 이동을 위한 변수 설정
                    if current_position is not None and depth_value > 0.0:
                        start_position = (current_position.x, current_position.y)
                        target_distance = depth_value
                        rospy.loginfo("Starting to move forward {:.2f} meters.".format(target_distance))
                    else:
                        rospy.logwarn("Invalid depth_value or current_position not available.")
            else:
                # 얼굴 박스가 중앙 3분의 1 구간 밖에 있을 경우, 회전을 계속함
                if state != STATE_ROTATING:
                    state = STATE_ROTATING
                    state_pub.publish("ROTATING")
                    rospy.loginfo("Face detected but not in central third. Starting to rotate.")
            
            # 얼굴이 중앙 3분의 1 구간 내에 있는 경우 PPG 신호 업데이트
            if left_third <= face_center_x <= right_third:
                # 얼굴 영역(ROI) 추출 및 PPG 신호 업데이트
                roi_box = orig_image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                roi_element_cnt = (int(box[3]) - int(box[1])) * (int(box[2]) - int(box[0]))
                if roi_element_cnt > 0:
                    red_mean = np.sum(roi_box[:, :, 2]) / roi_element_cnt
                    green_mean = np.sum(roi_box[:, :, 1]) / roi_element_cnt
                    blue_mean = np.sum(roi_box[:, :, 0]) / roi_element_cnt

                    redArray[:-1] = redArray[1:]
                    greenArray[:-1] = greenArray[1:]
                    blueArray[:-1] = blueArray[1:]
                    timeArray[:-1] = timeArray[1:]

                    redArray[-1] = red_mean
                    greenArray[-1] = green_mean
                    blueArray[-1] = blue_mean
                    timeArray[-1] = time.time()

                    # BPM 계산
                    dataArray[0, :] = redArray
                    dataArray[1, :] = greenArray
                    dataArray[2, :] = blueArray
                    heart_rate_bpm = calculate_heart_rate()

                    # BPM 표시
                    cv2.putText(orig_image, "BPM: {:.1f}".format(heart_rate_bpm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # 얼굴이 검출되지 않았을 때
            if state == STATE_ROTATING:
                # 이미 회전 중인 상태라면 계속 회전
                pass
            elif state != STATE_MOVING and state != STATE_MEASURING:
                # 회전하지 않고 얼굴을 못 찾은 경우 회전 시작
                state = STATE_ROTATING
                state_pub.publish("ROTATING")
                rospy.loginfo("No face detected. Starting to rotate.")
    
    # 이미지 퍼블리시
    try:
        img_pub = bridge.cv2_to_imgmsg(orig_image, encoding="bgr8")
        pub1.publish(img_pub)
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error while publishing image: {0}".format(e))

# cmd_vel 타이머 콜백 함수 정의
def cmd_vel_timer_callback(event):
    global state, start_position, target_distance, current_position

    if state == STATE_ROTATING:
        rotate_robot(0.2)  # 로봇을 회전시킴
        rospy.loginfo("Rotating...")
    elif state == STATE_MOVING:
        if current_position is None or start_position is None:
            rospy.logwarn("Current position or start position is not available.")
            return
        
        # 현재 위치에서 이동한 거리 계산
        dx = current_position.x - start_position[0]
        dy = current_position.y - start_position[1]
        distance_moved = math.sqrt(dx**2 + dy**2)
        
        rospy.loginfo("Distance moved: {:.2f}/{:.2f} meters.".format(distance_moved, target_distance))
        
        if distance_moved < target_distance-0.5:
            move_robot(0.2)  # 로봇을 직진시킴 (속도는 필요에 따라 조정)
            rospy.loginfo("Moving forward...")
        else:
            stop_robot()
            state = STATE_MEASURING
            state_pub.publish("MEASURING")
            rospy.loginfo("Reached target distance. Stopping and starting to measure heart rate.")
            # 심박수 측정을 위해 데이터 배열 초기화
            global redArray, greenArray, blueArray, timeArray, dataArray
            redArray = np.zeros(array_size)
            greenArray = np.zeros(array_size)
            blueArray = np.zeros(array_size)
            timeArray = np.zeros(array_size)
            dataArray = np.zeros((3, array_size))
    elif state == STATE_MEASURING:
        # 심박수 측정 로직을 여기에 추가
        rospy.loginfo("Measuring heart rate...")
        
        # 심박수 측정 완료 후 IDLE 상태로 전환
        state = STATE_IDLE
        state_pub.publish("IDLE")
        rospy.loginfo("Heart rate measured. Switching to IDLE state.")

if __name__ == '__main__':
    rospy.init_node('astra_face_detection')

    # 카메라 이미지 및 깊이 데이터 구독 설정
    rgb_sub = message_filters.Subscriber("/camera/rgb/image_raw", Image)
    depth_sub = message_filters.Subscriber("/camera/depth/image_raw", Image)
    ats = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.1)
    ats.registerCallback(image_callback)

    # cmd_vel 타이머 설정 (0.1초마다 cmd_vel 메시지를 퍼블리시)
    rospy.Timer(rospy.Duration(0.1), cmd_vel_timer_callback)

    rospy.spin()
