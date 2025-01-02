#!/usr/bin/env python3

import rospy
from std_msgs.msg import UInt16, Int16, UInt16MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from scipy import signal
import time
import math
import onnx
import UltraFaceDetector.vision.utils.box_utils_numpy as box_utils
from caffe2.python.onnx import backend
import onnxruntime as ort
import message_filters

# Initial Variables
face_min_width = 50
face_min_height = 50
init_delay_proc_time = 10
during_proc_time = 60
sampling_rate = 30  # Camera FPS
array_size = 300
bridge = CvBridge()

# 기본 변수 설정
fs = 30  # 샘플링 주파수
cffs = fs / 2
low = 0.75
high = 4
nfft = 2048
b, a = signal.butter(4, [low / cffs, high / cffs], btype='bandpass')

# RGB 배열 초기화
array_size = 300
redArray = np.zeros(array_size)
greenArray = np.zeros(array_size)
blueArray = np.zeros(array_size)
timeArray = np.zeros(array_size)
dataArray = np.zeros((3, array_size))


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
# Define face_flag_callback first
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

# Load Model
label_path = "/home/turtle/catkin_ws/src/turtlebot_control/scripts/UltraFaceDetector/models/voc-model-labels.txt"
onnx_path = "/home/turtle/catkin_ws/src/turtlebot_control/scripts/UltraFaceDetector/models/onnx/version-RFB-320.onnx"
class_names = [name.strip() for name in open(label_path).readlines()]
predictor = onnx.load(onnx_path)
onnx.checker.check_model(predictor)
onnx.helper.printable_graph(predictor.graph)
predictor = backend.prepare(predictor, device="CPU")
ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name

# ROS Setup
bridge = CvBridge()
pub1 = rospy.Publisher('/facedetection/rgb_image', Image, queue_size=10)
pub2 = rospy.Publisher('/facedetection/detect_face', UInt16, queue_size=10)
pub3 = rospy.Publisher('/facedetection/ROI_image', Image, queue_size=10)
pub4 = rospy.Publisher('/facedetection/roi_position_x', Int16, queue_size=10)
pub5 = rospy.Publisher('/facedetection/roi_position_y', Int16, queue_size=10)
pub6 = rospy.Publisher('/facedetection/roi_size', UInt16, queue_size=10)
HR_finsh_pub = rospy.Publisher('/facedetection/HR_finsh_flag', UInt16, queue_size=10)
sub5 = rospy.Subscriber('/facedetection/HR_start_flag', UInt16, face_flag_callback)

# PPGI 기반 심박수 계산 함수
def calculate_heart_rate():
    # POS 알고리즘에 따른 신호 처리
    CnInv = np.linalg.inv(np.diag(np.mean(dataArray, axis=1)))
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

# 얼굴 검출 및 PPGI 측정
def process_image(orig_image, depth_image):
    global firstDetectionFlag, burst_face_detection_Flag, start_face_flag
    threshold = 0.7
    face_detection_Flag = 0
    
    global dataArray, timeArray, redArray, greenArray, blueArray
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
        if (boxes[face_cnt, 2] - boxes[face_cnt, 0]) > face_min_width and (boxes[face_cnt, 3] - boxes[face_cnt, 1]) > face_min_height and probs[face_cnt] >= 0.97:
            firstDetectionFlag = 1
            face_detection_Flag = 1
            box = boxes[face_cnt, :]
            depth_value = get_depth_at_face_center(depth_image, box)  # Get depth at face center
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            label = f"{class_names[labels[face_cnt]]}:{(probs[face_cnt]*100):3.0f}% Depth: {depth_value:.2f}m"
            cv2.putText(orig_image, label, (box[0] - 40, box[3] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            
            # 얼굴 영역(ROI) 추출
            roi_box = orig_image[box[1]:box[3], box[0]:box[2]]
            # PPG 신호 배열 업데이트
            roi_element_cnt = (box[3] - box[1]) * (box[2] - box[0])
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
            cv2.putText(orig_image, f"BPM: {heart_rate_bpm:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            face_detection_Flag = 0            
    
    # 이미지 전송
    img_pub = bridge.cv2_to_imgmsg(orig_image, encoding="bgr8")
    pub1.publish(img_pub)

if __name__ == '__main__':
    rospy.init_node('astra_face_detection')
    rgb_sub = message_filters.Subscriber("/camera/rgb/image_raw", Image)
    depth_sub = message_filters.Subscriber("/camera/depth/image_raw", Image)
    ats = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.1)
    ats.registerCallback(image_callback)
    rospy.spin()
