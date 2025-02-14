#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import cv2
import rospy
import tkinter as tk
from tkinter import Entry, Label, Button, Text

import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, PoseWithCovarianceStamped
from actionlib_msgs.msg import GoalID
from std_msgs.msg import String

import numpy as np
from collections import deque
import time
import random

from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan

from MoveBase import MoveBase
from MapChange import MapChange
from MapChangev2 import MapChangev2
from IsDoorOpen import IsDoorOpen
from ArmGripper import ArmGripper

from dataclasses import dataclass

@dataclass
class Task:
    name: str
    is_completed: bool = False
    description: str = ""
    retry_count: int = 0
    time_taken: float = 0.0  # ★ 각 태스크에서 소요된 시간을 누적할 필드

# -------------------------
# 미리 정의된 위치/포즈
# -------------------------
INITIAL_POSE : dict = {'x' : 32.44, 'y' : -10.02,'qx' : 0.0 ,'qy' : 0.0 ,'qz' : -0.744,'qw' : 0.667}
MAP_CHANGE_POSE : dict = {'x' : 0.112 , 'y' : 0.076,'qx' : 0.0 ,'qy' : 0.0 ,'qz' : 0.035,'qw' : 1.0}
PICKUP_DESTINATION : dict = {'x' : 3.423 , 'y' : -2.405,'qx' : 0.0 ,'qy' : 0.0 ,'qz' : -0.715,'qw' : 0.6985}
ELEVATOR_BUTTON_DESTINATION : dict = {'x' : 41.68 , 'y' : -23.18,'qx' : 0.0 ,'qy' : 0.0 ,'qz' : 0,'qw' : 1}
FRONT_ELEVATOR_DESTINATION : dict = {'x' : -8.592 , 'y' : 31.41, 'qx' : 0.0 ,'qy' : 0.0 ,'qz' : 0.696,'qw' : 0.717}
ELEVATOR_DESTINATION : dict = {'x' : -8.54 , 'y' : 33.57,'qx' : 0.0 ,'qy' : 0.0 ,'qz' : 0,'qw' : 1.0}
ELEVATOR_BUTTON_POSE : dict = {'x' : -8.377 , 'y' : 33.288,'qx' : 0.0 ,'qy' : 0.0 ,'qz' : -0.032,'qw' : 1.0}
HEADING_POSE : dict = {'x' : -8.54 , 'y' : 33.62,'qx' : 0.0 ,'qy' : 0.0 ,'qz' : -0.641,'qw' : 0.767}
# DELIVER_DESTINATION : dict = {'x' : 35.347 , 'y' : 13.657,'qx' : 0.0 ,'qy' : 0.0 ,'qz' : -0.660,'qw' : 0.750}
DELIVER_DESTINATION : dict = {'x' : 24.053 , 'y' : 10.938,'qx' : 0.0 ,'qy' : 0.0 ,'qz' : 0.698,'qw' : 0.715}

class PickAndPlaceRobot(object):

    def __init__(self, hz = 100):
        self.rate = rospy.Rate(hz)

        self.tasks = [
            Task(name="move_to_pickup", description="픽업 장소로 이동"),
            Task(name="pickup", description="목표 물건 픽업"),
            Task(name="move_to_front_elevator", description="엘리베이터 앞으로 이동"),
            Task(name="isdooropen", description="엘리베이터 문 열렸는지 확인"),
            Task(name="move_to_elevator", description="엘리베이터로 이동"),
            Task(name="clickbutton", description="엘리베이터 버튼 클릭"),
            Task(name="heading_straight", description="정면 바라보기"),
            Task(name="change_map", description="맵 변경"),
            Task(name="initialize_pose_after_map_change", description="맵 변경 후 Pose 초기화"),
            Task(name="isdooropen", description="엘리베이터 문 열렸는지 확인"),
            Task(name="istargetfloor", description="목표 층수가 맞는지 확인"),
            Task(name="move_to_deliver", description="배달 장소로 이동"),
            Task(name="knock", description="문 두드리기")  # 마지막 단계
        ]

        # move_base
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.move_base.wait_for_server(rospy.Duration(1))
        self.cancel_pub = rospy.Publisher("/move_base/cancel", GoalID, queue_size=1)
        self.object_pub = rospy.Publisher('/target_object', String, queue_size=10)

        # 하위 클래스들
        self.driving = MoveBase()
        self.map_change = MapChange()
        self.map_changev2 = MapChangev2()
        self.isdooropen = IsDoorOpen()
        self.armgripper = ArmGripper()

        self.START = False
        self.floor_num = 0
        self.cur_task_level = 0

        # ----------------------
        # Tkinter GUI 초기화
        # ----------------------
        self.window = tk.Tk()
        self.window.title("Start/Stop GUI")
        self.window.geometry("600x900+600+0")

        # 층수 입력 라벨 및 Entry
        floor_label = Label(self.window, text="목적지 층수:", width=15, height=3)
        floor_label.grid(row=0, column=0, columnspan=2, pady=10, sticky="e")

        self.floor_num = Entry(self.window, width=10)
        self.floor_num.grid(row=0, column=2, pady=10, sticky="w")

        # 각 층 버튼
        self.create_floor_button("2층", 2)
        self.create_floor_button("3층", 3)
        self.create_floor_button("4층", 4)
        self.create_floor_button("5층", 5)

        # 물건 라벨 및 Entry
        obj_label = Label(self.window, text="대상 물체:", width=15, height=3)
        obj_label.grid(row=3, column=0, columnspan=2, pady=10, sticky="e")

        self.obj_num = Entry(self.window, width=10)
        self.obj_num.grid(row=3, column=2, pady=10, sticky="w")

        self.create_object_button("택배 1", 1)
        self.create_object_button("택배 2", 2)

        drive_label = Label(self.window, text="Driving Mode", font=("Arial", 15), height=4)
        drive_label.grid(row=8, column=0, columnspan=4, pady=(100, 10), sticky="nsew")

        start_button = Button(self.window, text="Start", width=20, height=15, command=self.start_button_clicked)
        start_button.grid(row=9, column=0, columnspan=2, sticky="nsew", padx=5, pady=10)

        stop_button = Button(self.window, text="Stop", width=20, height=15, command=self.stop_button_clicked)
        stop_button.grid(row=9, column=2, columnspan=2, sticky="nsew", padx=5, pady=10)

        # 상태 표시용 Text 위젯 (미션의 시작/성공/실패 표시)
        self.status_text = Text(self.window, width=60, height=10)
        self.status_text.grid(row=15, column=0, columnspan=4, padx=10, pady=10)

        # door judge param
        self.min_ang = -45
        self.max_ang = 45 
        self.threshold = 1.5
        self.target_num = 1

        rospy.on_shutdown(self.poweroff)

    def create_floor_button(self, text, floor_num):
        button = Button(self.window, text=text, width=15, height=3, 
                        command=lambda: self.manual_button_clicked(floor_num, is_floor=True))
        button.grid(row=2, column=floor_num-2, padx=5, pady=5, sticky="ew")

    def create_object_button(self, text, obj_num):
        button = Button(self.window, text=text, width=15, height=3, 
                        command=lambda: self.manual_button_clicked(text, is_floor=False))
        button.grid(row=4, column=obj_num, padx=5, pady=5, sticky="ew")
        
    def publish_object(self, obj_name):
        if obj_name:
            self.object_pub.publish(obj_name)
            rospy.loginfo(f"퍼블리시된 대상 물체: {obj_name}")
        else:
            rospy.logwarn("대상 물체가 선택되지 않았습니다.")

    def manual_button_clicked(self, target, is_floor):
        """층/물체 버튼 클릭 시 Entry 업데이트 & ROS Publish"""
        if is_floor:
            self.floor = target
            self.floor_num.delete(0, tk.END)
            self.floor_num.insert(0, str(self.floor))
            self.floor_num = self.floor_num.get()  # str
            print(f"선택된 층수: {self.floor}")
        else:
            self.obj = target
            self.obj_num.delete(0, tk.END)
            self.obj_num.insert(0, self.obj)
            self.publish_object(self.obj)
            print(f"선택된 물체: {self.obj}")

    def start_button_clicked(self):
        """Start 버튼 콜백"""
        self.START = True

    def stop_button_clicked(self):
        """Stop 버튼 콜백"""
        self.START = False
        self.move_base_shutdown()

    def update_mission_status(self, task_name, status_str):
        """
        미션 상태를 Text 위젯에 출력
        status_str에는 '시작', '성공', '실패' 등
        """
        self.status_text.insert(tk.END, f"[{rospy.Time.now().to_sec():.2f}] Task '{task_name}': {status_str}\n")
        self.status_text.see(tk.END)

    def area_visualize(self):
        """
        목적지 주변을 시각화(예: rviz에서 표시)하고 싶다면
        PoseWithCovarianceStamped 등을 퍼블리시
        """
        init_pose= rospy.Publisher("pickup_dest", PoseWithCovarianceStamped, queue_size=10)
        init_msg = PoseWithCovarianceStamped()
        init_msg.header.stamp = rospy.Time.now()
        init_msg.header.frame_id = "map"
        init_msg.pose.pose.position.x = PICKUP_DESTINATION['x']
        init_msg.pose.pose.position.y = PICKUP_DESTINATION['y']
        init_msg.pose.pose.orientation.w = 1.0
        init_msg.pose.covariance = [0.0 for _ in range(36)]
        init_msg.pose.covariance[0] = 0.25
        init_msg.pose.covariance[7] = 0.25
        init_pose.publish(init_msg)

        init_pose2= rospy.Publisher("elevator_dest", PoseWithCovarianceStamped, queue_size=10)
        init_msg2 = PoseWithCovarianceStamped()
        init_msg2.header.stamp = rospy.Time.now()
        init_msg2.header.frame_id = "map"
        init_msg2.pose.pose.position.x = ELEVATOR_DESTINATION['x']
        init_msg2.pose.pose.position.y = ELEVATOR_DESTINATION['y']
        init_msg2.pose.pose.orientation.w = 1.0
        init_msg2.pose.covariance = [0.0 for _ in range(36)]
        init_msg2.pose.covariance[0] = 0.25
        init_msg2.pose.covariance[7] = 0.25
        init_pose2.publish(init_msg2)

    def re_initialize_pose(self, ep):
        """
        로봇의 초기 위치를 (ep)에 맞게 재설정
        """
        print("adjust_pose!")
        init_pose= rospy.Publisher("initialpose", PoseWithCovarianceStamped, queue_size=10)
        init_msg = PoseWithCovarianceStamped()
        init_msg.header.stamp = rospy.Time.now()
        init_msg.header.frame_id = "map"
        init_msg.pose.pose.position.x = ep['x']
        init_msg.pose.pose.position.y = ep['y']
        init_msg.pose.pose.orientation.x = ep['qx']
        init_msg.pose.pose.orientation.y = ep['qy']
        init_msg.pose.pose.orientation.z = ep['qz']
        init_msg.pose.pose.orientation.w = ep['qw']
        init_msg.pose.covariance = [0.0 for _ in range(36)]
        init_msg.pose.covariance[0] = 0.25
        init_msg.pose.covariance[7] = 0.25
        init_msg.pose.covariance[-1] = 0.06853892326654787

        for _ in range(5):
            init_pose.publish(init_msg)
            rospy.sleep(0.1)

        time.sleep(15) 
        self.clear_costmap()

    def clear_costmap(self):
        try:
            clear_costmap_service = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
            clear_costmap_service()
        except rospy.ServiceException as e:
            rospy.logerr(f"clear_costmap 실패: {e}")

    def poweroff(self):
        print("poweroff")
        self.move_base_shutdown()
        self.window.destroy()

    def move_base_shutdown(self):
        rospy.loginfo("Stopping the Goal...")
        cancel_msg = GoalID()
        self.cancel_pub.publish(cancel_msg)
        print("cancel_goal")

    def run_task(self, event):
        """
        주기적으로(1/50초) 호출되어 현재 task를 확인,
        START 버튼이 눌린 상태라면 미션을 순차적으로 진행
        """
        rospy.loginfo_once("MAIN NODE START!")

        # 맨 처음에만 목적지 시각화
        self.area_visualize()

        if self.cur_task_level == len(self.tasks):
            print("All tasks are done.")
            self.print_summary()
            self.window.destroy()
            return

        if self.START:
            current_task = self.tasks[self.cur_task_level]
            task_name = current_task.name

            # --- 미션 시작 표시 (아직 수행 안 했으면)
            self.update_mission_status(task_name, "시작")

            start_time = time.time()
            is_completed = False

            # ==== task별 로직 수행 ====
            if task_name == "move_to_pickup":
                is_completed = self.driving(PICKUP_DESTINATION, self.move_base)

            elif task_name == "pickup":
                # ArmGripper pickup 모드
                is_completed = self.armgripper("pickup", self.target_num, self.floor_num)

            elif task_name == "move_to_front_elevator":
                is_completed = self.driving(FRONT_ELEVATOR_DESTINATION, self.move_base)

            elif task_name == "isdooropen":
                is_completed = self.isdooropen(self.min_ang, self.max_ang, self.threshold)

            elif task_name == "move_to_elevator":
                is_completed = self.driving(ELEVATOR_DESTINATION, self.move_base)

            elif task_name == "clickbutton":
                is_completed = self.armgripper("clickbutton", self.target_num, self.floor_num)

            elif task_name == "heading_straight":
                is_completed = self.driving(HEADING_POSE, self.move_base)

            elif task_name == "change_map":
                is_completed = self.map_changev2(self.floor_num)

            elif task_name == "initialize_pose_after_map_change":
                self.re_initialize_pose(MAP_CHANGE_POSE)
                is_completed = True
            
            elif task_name == "istargetfloor":
                # 단순 테스트용(실제 로직 대체)
                sleep_time = random.uniform(0.2, 1.0)  # 0.2초에서 1.0초 사이의 랜덤 값
                time.sleep(sleep_time)
                is_completed = True

            elif task_name == "move_to_deliver":
                is_completed = self.driving(DELIVER_DESTINATION, self.move_base)

            elif task_name == "knock":
                # ArmGripper knock 모드
                is_completed = self.armgripper("knock", self.target_num, self.floor_num)

            else:
                rospy.logwarn(f"Unknown Task: {task_name}")
                is_completed = True

            # ===== 미션 결과 처리 =====
            end_time = time.time()
            elapsed_time = end_time - start_time

            # ★ 시간을 누적해서 더해주는 부분 (성공/실패 상관없이 시도 시간 기록)
            current_task.time_taken += elapsed_time

            if is_completed:
                self.update_mission_status(task_name, "성공")
                self.cur_task_level += 1
            else:
                self.update_mission_status(task_name, "실패")
                current_task.retry_count += 1
                # 실패 시 cur_task_level을 증가시키지 않음 → 다음 주기 때 재시도

    def print_summary(self):
        total_time = 0
        for task in self.tasks:
            total_time += task.time_taken
            print(f"Task '{task.name}': {task.time_taken:.4f} seconds, Retries: {task.retry_count}")
        print(f"\nTotal time taken for all tasks: {total_time:.4f} seconds")

    def start_gui(self):
        """
        메인 GUI 루프
        """
        rospy.Timer(rospy.Duration(1.0/50.0), self.run_task)
        self.window.mainloop()
        rospy.spin()

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
