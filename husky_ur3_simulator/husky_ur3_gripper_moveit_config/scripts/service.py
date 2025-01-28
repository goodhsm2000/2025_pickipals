#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import sys
from your_package.srv import PerformAction, PerformActionRequest

def manipulate_client(command):
    """
    command = 'grasp', 'press_updown', 'press_number' 등
    """
    rospy.wait_for_service('/perform_action')
    try:
        service_proxy = rospy.ServiceProxy('/perform_action', PerformAction)
        req = PerformActionRequest(command=command)
        resp = service_proxy(req)
        if resp.success:
            rospy.loginfo(f"[manipulate_client] '{command}' action success!")
        else:
            rospy.logwarn(f"[manipulate_client] '{command}' action failed or object not found.")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")


if __name__ == "__main__":
    rospy.init_node('manipulate_client_node', anonymous=True)

    if len(sys.argv) < 2:
        rospy.loginfo("Usage: manipulate_client.py <command>")
        rospy.loginfo(" e.g.: manipulate_client.py grasp")
        sys.exit(1)

    command_input = sys.argv[1]  # grasp / press_updown / press_number
    manipulate_client(command_input)


#터미널 명령어
#rosrun husky_ur3_gripper_moveit_config manipulate_client.py grasp
#rosrun husky_ur3_gripper_moveit_config manipulate_client.py press_updown
#rosrun husky_ur3_gripper_moveit_config manipulate_client.py press_number
