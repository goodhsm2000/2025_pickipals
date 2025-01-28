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

import numpy as np
from collections import deque
# import tf

from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan

from MoveBase import MoveBase
from MapChange import MapChange

from dataclasses import dataclass

@dataclass
class Task:
    name: str
    is_completed: bool = False
    description: str = ""


# test initial pose
    #   z: 0.041924330408994195
    #   w: 0.9991204233600605

# test target
# 0.543;0.089;0.000;1.0
# -0.898;-1.182;0.0;1.0


INITIAL_POSE : dict = {'x' : 0 , 'y' : 0,'qx' : 0.0 ,'qy' : 0.0 ,'qz' : 0.042,'qw' : 1}

MAP_CHANGE_POSE : dict = {'x' : 0 , 'y' : 0,'qx' : 0.0 ,'qy' : 0.0 ,'qz' : 0.042,'qw' : 1}

PICKUP_DESTINATION : dict = {'x' : -1 , 'y' : 2.0,'qx' : 0.0 ,'qy' : 0.0 ,'qz' : 0,'qw' : 1.0}

FRONT_ELEVATOR_DESTINATION : dict = {'x' : 11.6117628 , 'y' : 6.406402,'qx' : 0.0 ,'qy' : 0.0 ,'qz' : 0.479070,'qw' : 0.8777764781459757}

ELEVATOR_DESTINATION : dict = {'x' : 11.6117628 , 'y' : 6.406402,'qx' : 0.0 ,'qy' : 0.0 ,'qz' : 0.479070,'qw' : 0.8777764781459757}

DELIVER_DESTINATION : dict = {'x' : 3 , 'y' : -5,'qx' : 0.0 ,'qy' : 0.0 ,'qz' : 0.479070,'qw' : 1}

class PickAndPlaceRobot(object):

    def __init__(self, hz = 100):
        
        self.rate = rospy.Rate(hz)
        # self.tasks = [
        #     Task(name="initialize_pose", description="Pose 초기화"),
        #     Task(name="move_to_pickup", description="픽업 장소로 이동"),
        #     Task(name="move_to_elevator", description="엘리베이터로 이동"),
        #     Task(name="change_map", description="맵 변경"),
        #     Task(name="move_to_deliver", description="배달 장소로 이동"),
        #     Task(name="knock_and_deliver", description="문 두드리기 및 배달")
        # ]

        self.tasks = [
            # Task(name="initialize_pose", description="Pose 초기화"),
            Task(name="move_to_pickup", description="픽업 장소로 이동"),
            # Task(name="move_to_elevator", description="엘리베이터로 이동"),
            Task(name="change_map", description="맵 변경"),
            Task(name="initialize_pose_after_map_change", description="맵 변경 후 Pose 초기화"),
            Task(name="move_to_deliver", description="배달 장소로 이동")
            # Task(name="knock_and_deliver", description="문 두드리기 및 배달")
        ]

        # rospy.wait_for_service('/move_base/clear_costmaps')
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.move_base.wait_for_server(rospy.Duration(1))
        self.cancel_pub = rospy.Publisher("/move_base/cancel", GoalID, queue_size=1)

        self.driving = MoveBase()
        self.map_change = MapChange()

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
        
        self.create_object_button("생수", 1)
        self.create_object_button("과자", 2)


        drive_label = Label(self.window, text="Driving Mode", font=("Arial", 15), height=4)
        drive_label.grid(row=8, column=0, columnspan=4, pady=(100, 10), sticky="nsew")

        start_button = Button(self.window, text="Start", width=20, height=15, command=self.start_button_clicked)
        start_button.grid(row=9, column=0, columnspan=2, sticky="nsew", padx=5, pady=10)

        # Stop 버튼 (오른쪽 반)
        stop_button = Button(self.window, text="Stop", width=20, height=15, command=self.stop_button_clicked)
        stop_button.grid(row=9, column=2, columnspan=2, sticky="nsew", padx=5, pady=10)

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

    def manual_button_clicked(self, target, is_floor):
        if is_floor:
            self.floor = target
            self.floor_num.delete(0, tk.END)  # 기존 입력값 삭제
            self.floor_num.insert(0, str(self.floor))  # 선택한 층수 입력란에 표시
            self.floor_num = self.floor_num.get()
            print(self.floor_num)
            print(self.floor_num == "5")
            print(f"선택된 층수: {self.floor}")

        else:
            self.obj = target
            self.obj_num.delete(0, tk.END)
            self.obj_num.insert(0, self.obj)
            self.obj_num = self.obj_num.get()
        

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

            ## 물건 좌표 전달 및 물체 pickup 수행 part
            # # ----

            # # ----

            # elevator로 이동
            # elif task.name == "move_to_elevator":
            #     task.is_completed = self.driving(FRONT_ELEVATOR_DESTINATION, self.move_base)
            #     if task.is_completed:
            #         task.is_completed = self.driving(ELEVATOR_DESTINATION, self.move_base)

            # floor_num에 해당하는 층의 map으로 변경 
            # 추후 GUI를 이용해 기능 확장 가능 
            elif self.tasks[self.cur_task_level].name == "change_map":
                is_completed = self.map_change(self.floor_num)
                if is_completed:
                    self.cur_task_level += 1

            elif self.tasks[self.cur_task_level].name == "initialize_pose_after_map_change":
                self.re_initialize_pose(MAP_CHANGE_POSE)
                print(f"Task '{self.tasks[self.cur_task_level].name}' completed!")
                self.cur_task_level += 1

            elif self.tasks[self.cur_task_level].name == "move_to_deliver":
                is_completed = self.driving(DELIVER_DESTINATION, self.move_base)
                if is_completed:
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


