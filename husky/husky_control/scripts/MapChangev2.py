#! /usr/bin/env python
# -*- coding:utf-8 -*-

import rospy
import subprocess
import time
from nav_msgs.srv import LoadMap

class MapChangev2(object):
    def __init__(self):
        # 서비스 클라이언트 초기화
        rospy.wait_for_service('change_map')
        self.map_client = rospy.ServiceProxy('change_map', LoadMap)
        
        # 맵 파일 경로
        self.maps = {
            "2": "/home/husky/catkin_ws/src/husky/husky_navigation/maps/electric2.yaml",
            "3": "/home/husky/catkin_ws/src/husky/husky_navigation/maps/electric3.yaml",
            "4": "/home/husky/catkin_ws/src/husky/husky_navigation/maps/electric4.yaml",
            "5": "/home/husky/catkin_ws/src/husky/husky_navigation/maps/electric5_path.yaml"}

    def __call__(self, map_num):
        if map_num not in self.maps:
            raise ValueError(f"Invalid map number: {map_num}. Available modes: {list(self.maps.keys())}")
        
        # subprocess.Popen(['rosnode', 'kill', '/map_server2'])

        map_file = self.maps[map_num]
        
        subprocess.Popen(['rosnode', 'kill', '/map_server'])
        time.sleep(3)

        try:
            resp = self.map_client(map_file)
            if resp.result == 0:
                rospy.loginfo("Change map succeeded")
                return True
            else:
                rospy.loginfo("Change map failed")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False


# uint8 RESULT_SUCCESS=0
# uint8 RESULT_MAP_DOES_NOT_EXIST=1
# uint8 RESULT_INVALID_MAP_DATA=2
# uint8 RESULT_INVALID_MAP_METADATA=3
# uint8 RESULT_UNDEFINED_FAILURE=255