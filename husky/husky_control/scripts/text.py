#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import cv2
import pytesseract

class OCRNode:
    def __init__(self):
        # CvBridge 초기화
        self.bridge = CvBridge()
        
        # 이미지 구독
        self.image_sub = rospy.Subscriber(
            "/camera/color/image_raw",  # subscribe할 토픽 이름
            Image,
            self.image_callback,
            queue_size=1
        )
        
        rospy.loginfo("OCRNode initialized. Waiting for images...")

    def image_callback(self, msg):
        try:
            # ROS Image를 OpenCV BGR 포맷으로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr("CV Bridge Error: {}".format(e))
            return

        # OCR 처리
        text = self.perform_ocr(cv_image)
        
        if text.strip():
            rospy.loginfo("추출된 텍스트: {}".format(text))
        else:
            rospy.loginfo("인식된 텍스트가 없습니다.")

    def perform_ocr(self, image):
        """
        OpenCV 이미지를 입력받아 Tesseract를 이용해 텍스트 추출
        """
        # 그레이스케일 변환 (가독성 향상을 위해)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # (선택) 이진화나 블러 등 전처리
        gray = cv2.medianBlur(gray, 3)
        # gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]
        

        cv2.imshow('test', gray)
        cv2.waitKey(1)
        # pytesseract로 이미지에서 텍스트 추출
        # 만약 한국어를 사용하려면 lang='kor' 등 언어 옵션 추가 (언어 데이터 설치 필요)
        # text = pytesseract.image_to_string(gray, lang='eng')  # 영어
        text = pytesseract.image_to_string(gray, config='--psm 6')  # 기본(영어)
        
        return text

def main():
    rospy.init_node("ocr_node", anonymous=True)
    node = OCRNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down OCRNode.")

if __name__ == "__main__":
    main()
