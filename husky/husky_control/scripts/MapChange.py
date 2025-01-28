#! /usr/bin/env python
# -*- coding:utf-8 -*-

import rospy
import time
import subprocess
from nav_msgs.srv import LoadMap
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped

class MapChange(object):
    def __init__(self):
        # 맵 파일 경로
        self.maps = {
            "2": "/home/husky/catkin_ws/src/husky/husky_navigation/maps/electric2.yaml",
            "3": "/home/husky/catkin_ws/src/husky/husky_navigation/maps/electric3.yaml",
            "4": "/home/husky/catkin_ws/src/husky/husky_navigation/maps/electric4.yaml",
            "5": "/home/husky/catkin_ws/src/husky/husky_navigation/maps/electric5_path.yaml"}
        
    def __call__(self, map_num):
        if map_num not in self.maps:
            raise ValueError(f"Invalid map number: {map_num}. Available modes: {list(self.maps.keys())}")
        
        map_file = self.maps[map_num]
        # path_map_file = "/home/husky/catkin_ws/src/husky_navigation/maps/woojungwo_1_path.yaml"
        
        try:
            subprocess.Popen(['rosnode', 'kill', '/map_server'])
            time.sleep(2)
            
            # # map_server 실행
            # subprocess.Popen(['rosrun', 'map_server', 'map_server', map_file, '_map:=/path_map'])    
            # subprocess.Popen(['rosrun', 'map_server', 'map_server', path_map_file, '__name:=map_server2', '/map:=/path_map'])   
            
            # rospy.loginfo("Waiting for /path_map topic to be published...")
            # rospy.wait_for_message('/path_map', OccupancyGrid)
            # rospy.loginfo("Path_Map successfully loaded.")  
            
            subprocess.Popen(['rosrun', 'map_server', 'map_server', map_file])           
            subprocess.Popen(['rosnode', 'kill', '/map_server2'])

            
            # /map이 재로드됐는지 확인
            # 안됐다면 amcl을 재시작했을 때 오류
            rospy.loginfo("Waiting for /map topic to be published...")
            rospy.wait_for_message('/map', OccupancyGrid)
            rospy.loginfo("Map successfully loaded.")   


            time.sleep(10) 

            subprocess.Popen(['rosservice', 'call', '/move_base/clear_costmaps'])
            
            
            # amcl node 초기화
            subprocess.Popen(['rosnode', 'kill', '/amcl'])
            time.sleep(2)
            subprocess.Popen(['rosrun', 'amcl', 'amcl'])             
            
            # amcl_pose 토픽 대기
            rospy.loginfo("Waiting for /amcl_pose topic...")
            rospy.wait_for_message('/amcl_pose', PoseWithCovarianceStamped)
            rospy.loginfo("/amcl_pose topic received. AMCL is initialized.")

            rospy.loginfo("Change map succeeded")
            return True

        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False

if __name__ == "__main__":
    rospy.init_node("map_change_node", anonymous=True)
    map_changer = MapChange()
    # 사용 예제
    map_num = "7"
    if map_changer(map_num):
        rospy.loginfo("Map change process completed successfully.")
    else:
        rospy.logerr("Map change process failed.")



# uint8 RESULT_SUCCESS=0
# uint8 RESULT_MAP_DOES_NOT_EXIST=1
# uint8 RESULT_INVALID_MAP_DATA=2
# uint8 RESULT_INVALID_MAP_METADATA=3
# uint8 RESULT_UNDEFINED_FAILURE=255
