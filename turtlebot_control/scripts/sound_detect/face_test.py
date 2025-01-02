#! /usr/bin/env python3
# import roslib; roslib.load_manifest('laser_values')

import rospy #ros protocal
from std_msgs.msg import UInt16, Int16, UInt16MultiArray
from sensor_msgs.msg import Image #sensor data type justify
from cv_bridge import CvBridge, CvBridgeError #openCV image data to ros data

import time
import cv2 # openCV
import numpy as np # array calculation 

from scipy import signal
import math #signal calculation

# 얼굴검출 관련 라이브러리
import onnx #open Neural Network Exchange
import UltraFaceDetector.vision.utils.box_utils_numpy as box_utils
from caffe2.python.onnx import backend
import onnxruntime as ort

# =================================초기 설정변수 start ============================
# 얼굴 최소 너비
face_min_width = 50
# 얼굴 최소 높이
face_min_height = 50
#초기 몇초뒤 bpm계산을 할지에 대한 지연시간(sec)
init_delay_proc_time = 10   

during_proc_time = 60
# =================================초기 설정변수 end ============================

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

def face_flag_callback(input_data) :
    global start_face_flag, burst_face_detection_Flag, firstDetectionFlag
    start_face_flag = input_data.data
    
    # 플래그들 초기화
    burst_face_detection_Flag = 1
    firstDetectionFlag = 0
    rospy.loginfo("start_face_flag = %d"%(start_face_flag))
    
def web_cam():
    pub1 = rospy.Publisher('/facedetection/rgb_image', Image)
    pub2 = rospy.Publisher('/facedetection/detect_face', UInt16)
    pub3 = rospy.Publisher('/facedetection/ROI_image', Image)
    pub7 = rospy.Publisher('/facedetection/test', UInt16MultiArray)
    #이미지의 크기 및 중심점과의 거리
    pub4 = rospy.Publisher('/facedetection/roi_position_x', Int16)
    pub5 = rospy.Publisher('/facedetection/roi_position_y', Int16)
    pub6 = rospy.Publisher('/facedetection/roi_size', UInt16)
    HR_finsh_pub = rospy.Publisher('/facedetection/HR_finsh_flag', UInt16)

    sub5 = rospy.Subscriber('/facedetection/HR_start_flag', UInt16, face_flag_callback)

    rospy.init_node('rgb_image')
    Bridge=CvBridge()



    label_path = "/home/turtle/catkin_ws/src/turtlebot_control/scripts/UltraFaceDetector/models/voc-model-labels.txt"
    onnx_path = "/home/turtle/catkin_ws/src/turtlebot_control/scripts/UltraFaceDetector/models/onnx/version-RFB-320.onnx"
    class_names = [name.strip() for name in open(label_path).readlines()]
    predictor = onnx.load(onnx_path)
    onnx.checker.check_model(predictor)
    onnx.helper.printable_graph(predictor.graph)
    predictor = backend.prepare(predictor, device="CPU")  # default CPU

    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    
    # cap = cv2.VideoCapture("/home/linzai/Videos/video/16_6.MP4")  # capture from camera
    cap = cv2.VideoCapture(0)  # capture from camera
    cap.set(3,640) # set Width
    cap.set(4,480) # set Height
    threshold = 0.7
    
    sum = 0
    #프레임계산용
    pixStartTime = time.time()
    startTime = time.time()
    nowTime = time.time()
    prevTime = 0
    proc_time = 0
    proc_start_time = time.time()
    frameCnt = 0
    preFrameCnt = 0
    
    stable_time_value = 0
    stable_flag = 0
    
    while True:
        ret, orig_image = cap.read()
        if orig_image is None:
            print("no img")
            break
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 240))
        # image = cv2.resize(image, (640, 480))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        # confidences, boxes = predictor.run(image)
        time_time = time.time()
        confidences, boxes = ort_session.run(None, {input_name: image})
        # print("cost time:{}".format(time.time() - time_time))
        boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
        

        # 얼굴 검출됨
        if boxes.shape[0] != 0 :
            face_cnt=0
            face_detection_Flag = 0
            # 얼굴 여러개일때 예외처리 없음 현재 0번에 검출된 얼굴만을 이용함
            # 얼굴이 가로세로 20픽셀보다 작은경우 무시 및 얼굴검출 스코어가 0.97이상일때만 검출
            if (boxes[face_cnt,2]-boxes[face_cnt,0]) > face_min_width and (boxes[face_cnt,3]-boxes[face_cnt,1]) > face_min_height and probs[face_cnt] >= 0.97 :
                firstDetectionFlag = 1
                face_detection_Flag = 1
            
                box = boxes[face_cnt, :]
                width = box[2] - box[0]
                height = box[3] - box[1]
                label = f"{class_names[labels[face_cnt]]}:{(probs[face_cnt]*100):3.0f}%"
                box_size_text = f"({width}, {height})"

                cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

                roi_box = orig_image[box[1]:box[3],box[0]:box[2]]
                
                cv2.putText(orig_image, label,
                            (box[0]-40, box[3] + 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,  # font scale
                            (255, 0, 255),
                            2)  # line type
                
                #font
                cv2.putText(orig_image, box_size_text,
                            (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,  # font scale
                            (0, 255, 0),  # color (green)
                            2)  # line type
        # 얼굴 검출하지 못한경우의 예외처리
        else :
            #얼굴을 못찾은 경우
            face_detection_Flag = 0
            box = [0, 0, 0, 0]
            
        img_pub = Bridge.cv2_to_imgmsg(orig_image,encoding="passthrough")
        pub1.publish(img_pub)
        
        
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    while not rospy.is_shutdown():
        try :
            web_cam()
            
        except rospy.ROSInterruptException: pass
