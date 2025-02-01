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
# import tf
import subprocess
import time

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

INITIAL_POSE : dict = {'x' : 32.44, 'y' : -10.02,'qx' : 0.0 ,'qy' : 0.0 ,'qz' : -0.744,'qw' : 0.667}

MAP_CHANGE_POSE : dict = {'x' : 0.112 , 'y' : 0.076,'qx' : 0.0 ,'qy' : 0.0 ,'qz' : 0.035,'qw' : 1.0}

PICKUP_DESTINATION : dict = {'x' : 3.423 , 'y' : -2.405,'qx' : 0.0 ,'qy' : 0.0 ,'qz' : -0.715,'qw' : 0.6985} # 완

ELEVATOR_BUTTON_DESTINATION : dict = {'x' : 41.68 , 'y' : -23.18,'qx' : 0.0 ,'qy' : 0.0 ,'qz' : 0,'qw' : 1}

FRONT_ELEVATOR_DESTINATION : dict = {'x' : -8.592 , 'y' : 31.41, 'qx' : 0.0 ,'qy' : 0.0 ,'qz' : 0.696,'qw' : 0.717} #완

ELEVATOR_DESTINATION : dict = {'x' : -8.54 , 'y' : 33.62,'qx' : 0.0 ,'qy' : 0.0 ,'qz' : 0,'qw' : 1.0}

ELEVATOR_BUTTON_POSE : dict = {'x' : -8.377 , 'y' : 33.288,'qx' : 0.0 ,'qy' : 0.0 ,'qz' : -0.032,'qw' : 1.0}

HEADING_POSE : dict = {'x' : -8.54 , 'y' : 33.62,'qx' : 0.0 ,'qy' : 0.0 ,'qz' : -0.641,'qw' : 0.767}

DELIVER_DESTINATION : dict = {'x' : 35.347 , 'y' : 13.657,'qx' : 0.0 ,'qy' : 0.0 ,'qz' : -0.660,'qw' : 0.750}

class PickAndPlaceRobot(object):

    def __init__(self, hz = 100):
        
        self.rate = rospy.Rate(hz)

        self.tasks = [
            # Task(name="initialize_pose", description="Pose 초기화"),
            # Task(name="move_to_pickup", description="픽업 장소로 이동"),
            # Task(name="pickup", description="목표 물건 픽업"),
            # Task(name="move_to_up_button_place", description="엘리베이터 버튼 장소"),
            # Task(name="move_to_front_elevator", description="엘리베이터 앞으로 이동"),
            # Task(name="up button click", description="엘리베이터 올라가는 버튼 클릭"),
            Task(name="isdooropen", description="엘리베이터 문 열렸는지 확인"),
            Task(name="move_to_elevator", description="엘리베이터로 이동"),
            # # Task(name="heading_button", description="버튼 쪽 바라보기"),
            Task(name="clickbutton", description="엘리베이터 버튼 클릭"),
            Task(name="heading_straight", description="정면 바라보기"),
            Task(name="change_map", description="맵 변경"),
            Task(name="initialize_pose_after_map_change", description="맵 변경 후 Pose 초기화"),
            Task(name="isdooropen", description="엘리베이터 문 열렸는지 확인"),
            # # Task(name="istargetfloor", description="목표 층수가 맞는지 확인"),
            Task(name="move_to_deliver", description="배달 장소로 이동"),
            Task(name="knock", description="문 두드리기")  # 마지막 단계
        ]

        # rospy.wait_for_service('/move_base/clear_costmaps')
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.move_base.wait_for_server(rospy.Duration(1))
        self.cancel_pub = rospy.Publisher("/move_base/cancel", GoalID, queue_size=1)
        self.object_pub = rospy.Publisher('/target_object', String, queue_size=10)

        self.driving = MoveBase()
        self.map_change = MapChange()
        self.map_changev2 = MapChangev2()
        self.isdooropen = IsDoorOpen()
        self.armgripper = ArmGripper()
        # self.istargetfloor = IsTargetFloor()

        self.START = False
        self.floor_num = 0
        self.cur_task_level = 0

        self.window = tk.Tk()
        self.window.title("Start/Stop GUI")
        self.window.geometry("600x900+600+0")

        # 층수 입력 Entry 위젯
        floor_label = Label(self.window, text="목적지 층수:", width=15, height=3)
        floor_label.grid(row=0, column=0, columnspan=2, pady=10, sticky="e")  # 오른쪽 정렬

        # 층수 입력 Entry 위젯 (Label 옆에 배치)
        self.floor_num = Entry(self.window, width=10)  # 너비를 짧게 설정
        self.floor_num.grid(row=0, column=2, pady=10, sticky="w")

        # 각 층을 선택하는 버튼
        self.create_floor_button("2층", 2)
        self.create_floor_button("3층", 3)
        self.create_floor_button("4층", 4)
        self.create_floor_button("5층", 5)

        # # 물건 입력 Entry 위젯
        obj_label = Label(self.window, text="대상 물체:", width=15, height=3)
        obj_label.grid(row=3, column=0, columnspan=2, pady=10, sticky="e")

        # 대상 물체 Entry 위젯
        self.obj_num = Entry(self.window, width=10)
        self.obj_num.grid(row=3, column=2, pady=10, sticky="w")
        
        self.create_object_button("택배 1", 1)
        self.create_object_button("택배 2", 2)


        drive_label = Label(self.window, text="Driving Mode", font=("Arial", 15), height=4)
        drive_label.grid(row=8, column=0, columnspan=4, pady=(100, 10), sticky="nsew")

        start_button = Button(self.window, text="Start", width=20, height=15, command=self.start_button_clicked)
        start_button.grid(row=9, column=0, columnspan=2, sticky="nsew", padx=5, pady=10)

        # Stop 버튼 (오른쪽 반)
        stop_button = Button(self.window, text="Stop", width=20, height=15, command=self.stop_button_clicked)
        stop_button.grid(row=9, column=2, columnspan=2, sticky="nsew", padx=5, pady=10)

        # elevator door judge param
        self.min_ang = -45
        self.max_ang = 45 
        self.threshold = 1.5
        self.target_num = 1
        
        # self.floor_num = "5"
        rospy.on_shutdown(self.poweroff)

    def create_floor_button(self, text, floor_num):
        """층 버튼"""
        button = Button(self.window, text=text, width=15, height=3, command=lambda: self.manual_button_clicked(floor_num, is_floor=True))
        button.grid(row=2, column=floor_num-2, padx=5, pady=5, sticky="ew")

    def create_object_button(self, text, obj_num):
        """물체 버튼"""
        button = Button(self.window, text=text, width=15, height=3, command=lambda: self.manual_button_clicked(text, is_floor=False))
        button.grid(row=4, column=obj_num, padx=5, pady=5, sticky="ew")
        
    def publish_object(self, obj_name):
      if obj_name:
          self.object_pub.publish(obj_name)
          rospy.loginfo(f"퍼블리시된 대상 물체: {obj_name}")
      else:
          rospy.logwarn("대상 물체가 선택되지 않았습니다.")

    def manual_button_clicked(self, target, is_floor):
        if is_floor:
            self.floor = target
            self.floor_num.delete(0, tk.END)  # 기존 입력값 삭제
            self.floor_num.insert(0, str(self.floor))  # 선택한 층수 입력란에 표시
            self.floor_num = self.floor_num.get()
            print(self.floor_num)
            print(f"선택된 층수: {self.floor}")

        else:
            self.obj = target
            self.obj_num.delete(0, tk.END)
            self.obj_num.insert(0, self.obj)
            # self.obj_num = self.obj_num.get()
            self.publish_object(self.obj)
        

            print(f"선택된 물체: {self.obj}")

    def start_button_clicked(self):
        self.START = True
    
    def stop_button_clicked(self):
        self.START = False
        self.move_base_shutdown()

    # def TimerCB(self,event):
    #     rospy.loginfo_once("MAIN NODE START!")

    #     if (self.START):
    #         # if 
    #         self.picki_control()

    def area_visualize(self):
        # covariance 0.25 = 0.5m radius

        init_pose= rospy.Publisher("pickup_dest", PoseWithCovarianceStamped, queue_size=10)
        init_msg = PoseWithCovarianceStamped()
        init_msg.header.stamp = rospy.Time.now()
        init_msg.header.frame_id = "map"
        init_msg.pose.pose.position.x = PICKUP_DESTINATION['x']
        init_msg.pose.pose.position.y = PICKUP_DESTINATION['y']
        init_msg.pose.pose.orientation.x = 0.0
        init_msg.pose.pose.orientation.y = 0.0
        init_msg.pose.pose.orientation.z = 0.0
        init_msg.pose.pose.orientation.w = 1.0
        init_msg.pose.covariance = [0.0 for _ in range(36)]
        init_msg.pose.covariance[0] = 0.25
        init_msg.pose.covariance[7] = 0.25
        init_msg.pose.covariance[-1] = 0.0

        init_pose.publish(init_msg)     

        init_pose= rospy.Publisher("elevator_dest", PoseWithCovarianceStamped, queue_size=10)
        init_msg = PoseWithCovarianceStamped()
        init_msg.header.stamp = rospy.Time.now()
        init_msg.header.frame_id = "map"
        init_msg.pose.pose.position.x = ELEVATOR_DESTINATION['x']
        init_msg.pose.pose.position.y = ELEVATOR_DESTINATION['y']
        init_msg.pose.pose.orientation.x = 0.0
        init_msg.pose.pose.orientation.y = 0.0
        init_msg.pose.pose.orientation.z = 0.0
        init_msg.pose.pose.orientation.w = 1.0
        init_msg.pose.covariance = [0.0 for _ in range(36)]
        init_msg.pose.covariance[0] = 0.25
        init_msg.pose.covariance[7] = 0.25
        init_msg.pose.covariance[-1] = 0.0
        init_pose.publish(init_msg)   
     
    def re_initialize_pose(self,ep):
        
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

        subprocess.Popen(['rosservice', 'call', '/move_base/clear_costmaps'])
        time.sleep(5)

    def poweroff(self):

        print("poweroff")
        self.move_base_shutdown()
        self.window.destroy()

    def move_base_shutdown(self):

        rospy.loginfo("Stopping the Goal...")
        # self.move_base.cancel_goal()
        cancel_msg = GoalID()
        self.cancel_pub.publish(cancel_msg)
        print("cancel_goal")


    # def get_tf(self):

    #     try:
    #         (trans,rot) = self.listener.lookupTransform('map', 'base_link', rospy.Time(0))
    #     except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
    #         return None
        
    #     return trans,rot

    def clear_costmap(self):
        try:
            clear_costmap_service = rospy.ServiceProxy('/move_base/clear_costmaps',Empty)
            clear_costmap_service()
        except rospy.ServiceException as e:
            rospy.logerr("",e)

    def run_task(self, event):
        rospy.loginfo_once("MAIN NODE START!")

        self.area_visualize()

        if self.cur_task_level == len(self.tasks):
            print("All tasks are done.")
            self.window.destroy()

        if self.START:
            
            # initialize
            is_completed = False
            
            # 각 작업을 처리하는 함수들
            if self.tasks[self.cur_task_level].name == "initialize_pose":
                self.re_initialize_pose(INITIAL_POSE)
                print(f"Task '{self.tasks[self.cur_task_level].name}' completed!")
                self.cur_task_level += 1

            # pickup 장소로 이동
            elif self.tasks[self.cur_task_level].name == "move_to_pickup":
                is_completed = self.driving(PICKUP_DESTINATION, self.move_base)
                if is_completed:
                    self.cur_task_level += 1

            # pickup 장소로 이동
            elif self.tasks[self.cur_task_level].name == "pickup":
                is_completed = self.armgripper("pickup", self.target_num, self.floor_num)
                if is_completed:
                    self.cur_task_level += 1

            # 엘리베이터 버튼 누르는 장소 이동
            # elif self.tasks[self.cur_task_level].name == "move_to_up_button_place":
            #     is_completed = self.driving(ELEVATOR_BUTTON_DESTINATION, self.move_base)
            #     if is_completed:
            #         print(f"Task '{self.tasks[self.cur_task_level].name}' completed!")
            #         self.cur_task_level += 1
            
            # elevator 앞으로 이동
            elif self.tasks[self.cur_task_level].name == "move_to_front_elevator":
                is_completed = self.driving(FRONT_ELEVATOR_DESTINATION, self.move_base)
                if is_completed:
                    print(f"Task '{self.tasks[self.cur_task_level].name}' completed!")
                    self.cur_task_level += 1
               
            # elevator 안으로 이동        
            elif self.tasks[self.cur_task_level].name == "move_to_elevator":
                is_completed = self.driving(ELEVATOR_DESTINATION, self.move_base)
                if is_completed:
                    print(f"Task '{self.tasks[self.cur_task_level].name}' completed!")
                    self.cur_task_level += 1

            # 버튼 쪽 바라보기
            elif self.tasks[self.cur_task_level].name == "heading_button":
                is_completed = self.driving(ELEVATOR_BUTTON_POSE, self.move_base)
                if is_completed:
                    print(f"Task '{self.tasks[self.cur_task_level].name}' completed!")
                    self.cur_task_level += 1


            elif self.tasks[self.cur_task_level].name == "clickbutton":
                is_completed = self.armgripper("clickbutton", self.target_num, self.floor_num)
                if is_completed:
                    self.cur_task_level += 1
            
            # 정면 바라보기
            elif self.tasks[self.cur_task_level].name == "heading_straight":
                is_completed = self.driving(HEADING_POSE, self.move_base)
                if is_completed:
                    print(f"Task '{self.tasks[self.cur_task_level].name}' completed!")
                    self.cur_task_level += 1

            # floor_num에 해당하는 층의 map으로 변경 
            # 추후 GUI를 이용해 기능 확장 가능 
            elif self.tasks[self.cur_task_level].name == "change_map":
                is_completed = self.map_changev2(self.floor_num)
                if is_completed:
                    print(f"Task '{self.tasks[self.cur_task_level].name}' completed!")
                    self.cur_task_level += 1
             
            elif self.tasks[self.cur_task_level].name == "initialize_pose_after_map_change":
                self.re_initialize_pose(MAP_CHANGE_POSE)
                print(f"Task '{self.tasks[self.cur_task_level].name}' completed!")
                self.cur_task_level += 1
                    
            elif self.tasks[self.cur_task_level].name == "isdooropen":
                is_completed = self.isdooropen(self.min_ang, self.max_ang, self.threshold)
                if is_completed:
                    print(f"Task '{self.tasks[self.cur_task_level].name}' completed!")
                    print("Current floor: 5F")
                    self.cur_task_level += 1

            elif self.tasks[self.cur_task_level].name == "istargetfloor":
                is_completed = self.istargetfloor(self.min_ang, self.max_ang, self.threshold)
                if is_completed:
                    print(f"Task '{self.tasks[self.cur_task_level].name}' completed!")
                    self.cur_task_level += 1

            elif self.tasks[self.cur_task_level].name == "move_to_deliver":
                is_completed = self.driving(DELIVER_DESTINATION, self.move_base)
                if is_completed:
                    print(f"Task '{self.tasks[self.cur_task_level].name}' completed!")
                    self.cur_task_level += 1

            elif self.tasks[self.cur_task_level].name == "knock":
                # 문 두드리기 동작 예시. ArmGripper에 knock 모드가 있다고 가정.
                is_completed = self.armgripper("knock", self.target_num, self.floor_num)
                if is_completed:
                    print(f"Task '{self.tasks[self.cur_task_level].name}' completed!")
                    print("Knock the door and finish.")
                    self.cur_task_level += 1        

            ## 문 두드리기 & 최종 물건 drop 좌표 전달 및 물체 pickup 수행 part
            # ----

            # ----
            # elif task.name == "knock_and_deliver":
            #     print("Knock the door and deliver")
            #     task.is_completed = True  # 완료 처리


    # def picki_control(self):
    #     # 목적지 좌표 시각화하기
    #     self.area_visualize()

        

    #     # 작업 수행
    #     for task in self.tasks:
    #         print(f"Starting task: {task.description}")
    #         while not task.is_completed:
    #             self.run_task(task)
    #         print(f"Task '{task.name}' completed!")

    #     print("All tasks are done.")
    #     self.window.destroy()

    def start_gui(self):
        # self.window.grid_rowconfigure(0, weight=1)
        # self.window.grid_rowconfigure(1, weight=1)
        # self.window.grid_rowconfigure(2, weight=1)
        # self.window.grid_columnconfigure(0, weight=1)
        # self.window.grid_columnconfigure(1, weight=1)
        # self.window.grid_columnconfigure(2, weight=1)
        # self.window.grid_columnconfigure(3, weight=1)
        # self.window.grid_columnconfigure(4, weight=1)
        rospy.Timer(rospy.Duration(1.0/ 50.0), self.run_task)
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