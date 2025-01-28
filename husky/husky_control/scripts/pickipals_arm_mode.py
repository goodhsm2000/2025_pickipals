#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import cv2
import rospy
import tkinter as tk
from tkinter import Entry, Label, Button

import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, PoseWithCovarianceStamped
from actionlib_msgs.msg import GoalID
from std_msgs.msg import String

import numpy as np
from collections import deque

from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan

from MoveBase import MoveBase
from MapChange import MapChange
from IsDoorOpen import IsDoorOpen
from ArmGripper import ArmGripper

from dataclasses import dataclass

@dataclass
class Task:
    name: str
    is_completed: bool = False
    description: str = ""

class PickAndPlaceRobot(object):

    def __init__(self, hz=10):
        # ROS 관련 초기화
        self.rate = rospy.Rate(hz)
        rospy.on_shutdown(self.poweroff)

        # MoveBase Action 서버
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.move_base.wait_for_server(rospy.Duration(1))

        # 목표 취소 퍼블리셔
        self.cancel_pub = rospy.Publisher("/move_base/cancel", GoalID, queue_size=1)
        # 물체 인식 퍼블리셔
        self.object_pub = rospy.Publisher('/target_object', String, queue_size=10)

        # 기능별 클래스 (필요 시 사용)
        self.driving = MoveBase()
        self.map_change = MapChange()
        self.isdooropen = IsDoorOpen()
        self.armgripper = ArmGripper()

        # 층수/물체 번호
        self.floor_num = 0
        self.target_num = 1  # 초기값(1번)
        self.obj = ""  # 현재 선택된 물체("1번" 또는 "2번")

        # GUI 관련
        self.window = tk.Tk()
        self.window.title("Start/Stop GUI")
        self.window.geometry("600x900+600+0")

        # 층수 입력 위젯 (예: 2~5층으로 가정)
        floor_label = Label(self.window, text="목적지 층수:", width=15, height=3)
        floor_label.grid(row=0, column=0, columnspan=2, pady=10, sticky="e")
        self.floor_num_entry = Entry(self.window, width=10)
        self.floor_num_entry.grid(row=0, column=2, pady=10, sticky="w")

        # 층 선택 버튼(2,3,4,5)
        self.create_floor_button("2층", 2)
        self.create_floor_button("3층", 3)
        self.create_floor_button("4층", 4)
        self.create_floor_button("5층", 5)

        # 물체 선택 위젯
        obj_label = Label(self.window, text="대상 물체:", width=15, height=3)
        obj_label.grid(row=3, column=0, columnspan=2, pady=10, sticky="e")

        self.obj_num_entry = Entry(self.window, width=10)
        self.obj_num_entry.grid(row=3, column=2, pady=10, sticky="w")

        # 물체 버튼 (1번, 2번)
        self.create_object_button("1번", 1)
        self.create_object_button("2번", 2)

        # 운행(Start/Stop) 버튼
        drive_label = Label(self.window, text="Driving Mode", font=("Arial", 15), height=4)
        drive_label.grid(row=8, column=0, columnspan=4, pady=(50, 10), sticky="nsew")

        start_button = Button(self.window, text="Start", width=20, height=5, command=self.start_button_clicked)
        start_button.grid(row=9, column=0, columnspan=2, sticky="nsew", padx=5, pady=10)

        stop_button = Button(self.window, text="Stop", width=20, height=5, command=self.stop_button_clicked)
        stop_button.grid(row=9, column=2, columnspan=2, sticky="nsew", padx=5, pady=10)

        # 추가 기능 버튼 (Pickup Mode, Click Button Mode)
        feature_label = Label(self.window, text="Arm Mode", font=("Arial", 15), height=4)
        feature_label.grid(row=10, column=0, columnspan=4, pady=(50, 10), sticky="nsew")

        pickup_mode_button = Button(self.window, text="Pickup Mode", width=15, height=5, command=self.pickup_mode)
        pickup_mode_button.grid(row=11, column=0, columnspan=2, sticky="nsew", padx=5, pady=10)

        clickbutton_mode_button = Button(self.window, text="Click Button Mode", width=15, height=5, command=self.clickbutton_mode)
        clickbutton_mode_button.grid(row=11, column=2, columnspan=2, sticky="nsew", padx=5, pady=10)

        # 동작 플래그
        self.START = False

        # 주기적으로 처리할 함수 (예: ROS spin과 함께 GUI도 동작)
        rospy.Timer(rospy.Duration(1.0 / hz), self.run)

    def create_floor_button(self, text, floor_num):
        """층수 버튼 생성 함수"""
        button = Button(self.window, text=text, width=15, height=2,
                        command=lambda: self.manual_button_clicked(floor_num, is_floor=True))
        button.grid(row=2, column=floor_num-2, padx=5, pady=5, sticky="ew")

    def create_object_button(self, text, column_index):
        """물체 선택 버튼 생성 함수"""
        button = Button(self.window, text=text, width=15, height=2,
                        command=lambda: self.manual_button_clicked(text, is_floor=False))
        button.grid(row=4, column=column_index, padx=5, pady=5, sticky="ew")

    def manual_button_clicked(self, target, is_floor):
        """층수/물체 버튼 클릭 시 호출되는 함수"""
        if is_floor:
            # 층수 갱신
            self.floor_num_entry.delete(0, tk.END)
            self.floor_num_entry.insert(0, str(target))
            self.floor_num = self.floor_num_entry.get()
            print(self.floor_num)
            rospy.loginfo(f"선택된 층수: {self.floor_num}")
        else:
            # 물체 갱신 ("1번", "2번")
            self.obj = target
            self.obj_num_entry.delete(0, tk.END)
            self.obj_num_entry.insert(0, self.obj)
            if target == "1번":
                self.target_num = 1
            elif target == "2번":
                self.target_num = 2
            # 필요하다면 토픽 퍼블리시(옵션)
            self.publish_object(self.obj)
            rospy.loginfo(f"선택된 물체: {self.obj}")

    def publish_object(self, obj_name):
        """선택된 물체를 퍼블리시할 때 사용(옵션)"""
        if obj_name:
            self.object_pub.publish(obj_name)
            rospy.loginfo(f"퍼블리시된 대상 물체: {obj_name}")
        else:
            rospy.logwarn("대상 물체가 선택되지 않았습니다.")

    def start_button_clicked(self):
        """Start 버튼"""
        self.START = True
        rospy.loginfo("시작 버튼 클릭")

    def stop_button_clicked(self):
        """Stop 버튼"""
        self.START = False
        self.move_base_shutdown()
        rospy.loginfo("정지 버튼 클릭")

    def pickup_mode(self):
        """Pickup 모드 버튼"""
        if not self.obj:
            rospy.logwarn("Pickup 모드 실행 전, 물체(1번/2번)를 선택하세요.")
            return
        # ArmGripper 노드를 호출하여 pickup 기능 수행
        result = self.armgripper("pickup", self.target_num, self.floor_num)
        if result:
            rospy.loginfo("Pickup 완료!")
        else:
            rospy.logwarn("Pickup 실패!")

    def clickbutton_mode(self):
        """Click Button 모드 버튼"""
        # ArmGripper 노드를 호출하여 clickbutton 기능 수행
        result = self.armgripper("clickbutton", self.target_num, self.floor_num)
        if result:
            rospy.loginfo("Click Button 완료!")
        else:
            rospy.logwarn("Click Button 실패!")

    def move_base_shutdown(self):
        """move_base 목표를 취소"""
        rospy.loginfo("Stopping the Goal...")
        cancel_msg = GoalID()
        self.cancel_pub.publish(cancel_msg)
        rospy.loginfo("cancel_goal 퍼블리시 완료")

    def poweroff(self):
        """노드 종료 시 정리"""
        rospy.loginfo("poweroff")
        self.move_base_shutdown()
        self.window.destroy()

    def run(self, event):
        """
        주기적으로 실행될 함수.
        여기서는 START가 True인 경우에 특정 로직을 수행하도록 할 수 있지만,
        현재는 Pickup/Clickbutton만 버튼으로 제어하도록 했으므로
        필요하면 추가 동작을 이 곳에서 구현하면 됨.
        """
        if self.START:
            # 필요하면 자율 주행, 맵 변경 등 추가 로직 작성
            pass

    def start_gui(self):
        """Tkinter GUI 실행 함수"""
        self.window.mainloop()  # GUI 이벤트 루프
        rospy.spin()            # ROS 이벤트 루프

if __name__ == "__main__":
    rospy.init_node('pickibot')
    pickibot = PickAndPlaceRobot()
    try:
        pickibot.start_gui()
    except rospy.ROSInterruptException:
        pickibot.poweroff()
        rospy.loginfo("Shutting down")
    finally:
        print("Driving Done")
