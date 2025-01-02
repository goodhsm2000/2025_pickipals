#! /usr/bin/env python3
# import roslib; roslib.load_manifest('laser_values')

import rospy
from std_msgs.msg import UInt16, Int16, UInt16MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import time
import cv2
import numpy as np

from scipy import signal
import math

# 얼굴검출 관련 라이브러리
import onnx
import UltraFaceDetector.vision.utils.box_utils_numpy as box_utils
from caffe2.python.onnx import backend
import onnxruntime as ort

# =================================초기 설정변수 start ============================
# 얼굴 최소 너비
face_min_width = 80
# 얼굴 최소 높이
face_min_height = 110
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

    global start_face_flag, burst_face_detection_Flag, firstDetectionFlag

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



    label_path = "/home/turtle/catkin_ws/src/user_projects/src/UltraFaceDetector/models/voc-model-labels.txt"
    onnx_path = "/home/turtle/catkin_ws/src/user_projects/src/UltraFaceDetector/models/onnx/version-RFB-320.onnx"
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

    # 필터 생성 및 변수 선언
    fs = 30
    cffs = fs/2
    low = 0.75
    high = 4
    nfft = 2048
    proc_flag = 0
    b, a = signal.butter(4, [low/cffs, high/cffs], btype='bandpass')
    preFrameCnt = 30
    # 그래프 생성용
    graph_freq_cut = np.fix((nfft/2)/(fs/2)*4)
    graph_size = 640
    graph_x = np.arange(graph_size)
    graph_y = np.zeros(graph_size)

    # 플래그 변수
    
    
    array_size = 300
    freq_max_bpm = 0
    face_detection_Flag = 0
    during_proc_cnt = during_proc_time
    unstable_flag = 0
    # bpm알고리즘 변수선언
    dataArray = np.zeros((3, array_size))
    redArray = np.zeros(array_size)
    greenArray = np.zeros(array_size)
    blueArray = np.zeros(array_size)
    timeArray = np.zeros(array_size)

    box = [0, 0, 0, 0]
    
    unstable_cnt = 0


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
                label = f"{class_names[labels[face_cnt]]}:{(probs[face_cnt]*100):3.0f}%"

                cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

                roi_box = orig_image[box[1]:box[3],box[0]:box[2]]
                
                cv2.putText(orig_image, label,
                            (box[0]-40, box[3] + 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,  # font scale
                            (255, 0, 255),
                            2)  # line type
        # 얼굴 검출하지 못한경우의 예외처리
        else :
            #얼굴을 못찾은 경우
            face_detection_Flag = 0
            box = [0, 0, 0, 0]

            # if firstDetectionFlag == 0 :
            #     firstDetectionFlag = 0
            #     print('얼굴인식대기중')
            # else :
                
            #     cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            #     roi_box = orig_image[box[1]:box[3],box[0]:box[2]]

        # ROI 신호처리
        if face_detection_Flag == 1 and start_face_flag == 1 :

            roi_element_cnt = (box[3]- box[1]) * (box[2] - box[0])
            get_ch1 = roi_box[:, :, 0]
            Blue_mean = np.sum(get_ch1) / roi_element_cnt
            # Blue_mean=np.mean(get_ch1)
            get_ch2 = roi_box[:, :, 1]
            Green_mean = np.sum(get_ch2) / roi_element_cnt
            # Green_mean=np.mean(get_ch2)
            get_ch3 = roi_box[:, :, 2]
            Red_mean = np.sum(get_ch3) / roi_element_cnt
            # Red_mean=np.mean(get_ch3)
            
            #RGB 배열 쉬프트
            for index in range(array_size-1,0,-1) :
                redArray[index] = redArray[index-1] 
                greenArray[index] = greenArray[index-1] 
                blueArray[index] = blueArray[index-1] 
                timeArray[index] = timeArray[index-1] 
            redArray[0] = Red_mean
            greenArray[0] = Green_mean
            blueArray[0] = Blue_mean
            timeArray[0] = nowTime - pixStartTime
            dataArray[0, :] = redArray
            dataArray[1, :] = greenArray
            dataArray[2, :] = blueArray

            # 배열전체가 0일경우 오류가 나기 때문에 계산 안하도록 초기화
            if np.sum(redArray)==0 :
                proc_flag = 0
            
            if proc_flag == 1 : # 1초마다 실행
                #POS 알고리즘

                CnInv = np.linalg.inv(np.diag(np.mean(dataArray,axis=1)))
                Cn = np.dot(CnInv,dataArray)
                SS = np.array([[0, 1, -1],[-2, 1, 1]])
                AA = np.dot(SS,Cn)
                AA_std = np.std(AA[0,:])/np.std(AA[1,:])
                Pn = np.array([1, AA_std])
                P = np.dot(Pn, AA)
                #디지털 필터 (Butter 4차 0.75~4 BPF)
                filterPPG = signal.lfilter(b, a, P) 
                cutPPG = filterPPG[2*fs:np.size(filterPPG)]
                #FFT변환
                fft_out = np.fft.fft(cutPPG, nfft)
                fft_abs = np.abs(fft_out)
                fft_nor = (fft_abs-np.min(fft_abs))/np.max((fft_abs-np.min(fft_abs)))

                #주파수 축으로 변환
                y = fft_nor[range(math.trunc(nfft/2))]
                k = np.arange(nfft)
                f0 = k*fs/nfft
                f0 = f0[range(math.trunc(nfft/2))]
                #주파수의 최대값 검출
                peaks, _ = signal.find_peaks(y,distance=10,)
                peaks_amp = y[peaks]
                Peaks_max_idx = peaks[np.argmax(peaks_amp)]
                #BPM계산
                freq_max_bpm = f0[Peaks_max_idx]*60
                #그래프 입히기
                y_height = y[range(math.trunc(graph_freq_cut))]*360
                graph_y = signal.resample(y_height, graph_size)

                graph_y = -(graph_y-360)

                #플래그 초기화
                proc_flag = 0


            

            # 불안정
            if stable_time_value >= 9 :
                str = "BPM : Not measurable"
                cv2.putText(orig_image, str, (0, 420), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                str = "Measure count : Looking for a face"
                cv2.putText(orig_image, str, (0, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # 측정10초 대기                
            elif stable_time_value < 9 and stable_time_value>0 :
                str = "BPM : Wait.."
                cv2.putText(orig_image, str, (0, 420), cv2.FONT_HERSHEY_SIMPLEX, 1, (70, 200, 60), 2)
                str = "Measure count : %d" % stable_time_value
                cv2.putText(orig_image, str, (0, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (70, 200, 60), 2)

            else :
                str = "BPM : %0.1f" % freq_max_bpm
                cv2.putText(orig_image, str, (0, 420), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                str = "Time : %2d (/%2d)" % (60-(during_proc_cnt+10), during_proc_time)
                cv2.putText(orig_image, str, (0, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                pts = np.vstack((graph_x,graph_y)).astype(np.int32).T
                cv2.polylines(orig_image, [pts], isClosed=False, color=(255,0,0), thickness=2)

            if burst_face_detection_Flag == 1 :
                stable_time_value = init_delay_proc_time
                pub2.publish(face_detection_Flag)
                burst_face_detection_Flag = 0

            if nowTime - startTime >= 1 :
                proc_time = nowTime-proc_start_time
                if proc_time >= init_delay_proc_time :
                    proc_flag = 1
                stable_time_value = stable_time_value - 1
                during_proc_cnt = during_proc_cnt - 1

        # 안정도 판별
        if firstDetectionFlag ==1 and face_detection_Flag == 0 and start_face_flag == 1 :
            unstable_cnt = unstable_cnt+1
            stable_time_value = init_delay_proc_time
            proc_start_time = time.time()
            during_proc_cnt = during_proc_time

        # 얼굴이 불안정하게 검출되거나 60초 동안 측정이 끝나면 심박수 계산 종료 
        if unstable_cnt >=200 or (during_proc_cnt+10) <= 0 :
            unstable_cnt = 0
            during_proc_cnt = during_proc_time
            # firstDetectionFlag = 0
            start_face_flag = 0
            stable_flag = 0
            # unstable_flag = unstable_flag+1
            HR_finsh_flag = 1
            HR_finsh_pub.publish(HR_finsh_flag)

        #FPS계산
        if nowTime - startTime >= 1 :
            preFrameCnt = frameCnt
            frameCnt = 0
            startTime = nowTime
            nowTime = time.time()
            
            
            # print(preFrameCnt)
        else :
            nowTime = time.time()
            frameCnt += 1


        # if nowTime - startTime >= 1 :
        #     preFrameCnt = frameCnt
        #     frameCnt = 0
        #     startTime = nowTime
        #     nowTime = time.time()
        #     proc_time = nowTime-proc_start_time
        #     stable_time_value = stable_time_value - 1
        #     during_proc_cnt = during_proc_cnt - 1
        #     if proc_time >= init_delay_proc_time :
        #         proc_flag = 1
        #     # print(preFrameCnt)
        # else :
        #     nowTime = time.time()
        #     frameCnt += 1


        str = "FPS : %0.1f" % preFrameCnt
        cv2.putText(orig_image, str, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        str = "Dedug : %d, %d, %d" % (start_face_flag, unstable_cnt, stable_time_value)
        cv2.putText(orig_image, str, (0, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # str = "BPM : %0.1f" % freq_max_bpm
        # cv2.putText(orig_image, str, (0, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # pts = np.vstack((graph_x,graph_y)).astype(np.int32).T
        # cv2.polylines(orig_image, [pts], isClosed=False, color=(255,0,0), thickness=2)

        
        # cv2.imshow('V1', orig_image)

        img_pub = Bridge.cv2_to_imgmsg(orig_image,encoding="passthrough")
        pub1.publish(img_pub)
        # pub2.publish(face_detection_Flag)
        
        # test_array = UInt16MultiArray()
        # test_array.data= [face_detection_Flag, box[2]-box[0], box[3]-box[1]]
        
        # pub7.publish(test_array)

        # rospy.loginfo(img_pub)
        # rospy.loginfo(face_detection_Flag)


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    while not rospy.is_shutdown():
        try :
            web_cam()
            
        except rospy.ROSInterruptException: pass


