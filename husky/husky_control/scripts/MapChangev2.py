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
            "5": "/home/husky/catkin_ws/src/husky/husky_navigation/maps/electric5_path.yaml"
        }
    
    def wait_until_node_killed(self, node_name, timeout=10):
        """
        지정한 노드가 kill 될 때까지 대기하는 함수.
        :param node_name: 확인할 노드 이름 (예: '/map_server')
        :param timeout: 최대 대기 시간(초)
        :return: 노드가 종료되면 True, 타임아웃 시 False
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # 현재 활성화된 노드 목록 가져오기
                output = subprocess.check_output(['rosnode', 'list']).decode('utf-8')
                if node_name not in output:
                    rospy.loginfo(f"노드 {node_name}가 종료되었습니다.")
                    return True
            except subprocess.CalledProcessError as e:
                rospy.logwarn(f"노드 목록 확인 중 에러 발생: {e}")
            time.sleep(0.5)  # 0.5초마다 체크
        rospy.logwarn(f"타임아웃: {node_name} 노드가 {timeout}초 이내에 종료되지 않았습니다.")
        return False

    def __call__(self, map_num):
        if map_num not in self.maps:
            raise ValueError(f"Invalid map number: {map_num}. Available modes: {list(self.maps.keys())}")
        
        map_file = self.maps[map_num]
        
        # 이후 map_client 호출
        try:
            resp = self.map_client(map_file)
            if resp.result == 0:
                rospy.loginfo("맵 변경 성공")

                # map_server 노드 종료 요청
                subprocess.Popen(['rosnode', 'kill', '/map_server2'])
                rospy.loginfo("map_server 종료 요청을 보냈습니다. 노드 종료 대기 중...")
                
                # map_server가 완전히 종료될 때까지 대기
                if not self.wait_until_node_killed('/map_server2'):
                    rospy.logwarn("map_server가 여전히 실행 중입니다. 이후 동작에 문제가 발생할 수 있습니다.")
                
                return True
            else:
                rospy.loginfo("맵 변경 실패")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"서비스 호출 실패: {e}")
            return False

# 참고: 아래는 이 클래스가 사용되는 예시입니다.
if __name__ == '__main__':
    rospy.init_node('map_change_node')
    changer = MapChangev2()
    # 예시: "3"번 맵으로 변경 시도
