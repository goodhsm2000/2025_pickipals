#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped

def send_goal(x, y, w):
    rospy.init_node('send_navigation_goal', anonymous=True)
    goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz

    while goal_pub.get_num_connections() == 0:
        rospy.loginfo("Waiting for a subscriber to connect to /move_base_simple/goal")
        rate.sleep()

    goal = PoseStamped()
    goal.header.stamp = rospy.Time.now()
    goal.header.frame_id = "map"

    goal.pose.position.x = x
    goal.pose.position.y = y
    goal.pose.position.z = 0.0

    goal.pose.orientation.w = w
    goal.pose.orientation.x = 0.0
    goal.pose.orientation.y = 0.0
    goal.pose.orientation.z = 0.0

    rospy.loginfo("Publishing goal to /move_base_simple/goal: x=%f, y=%f, w=%f", x, y, w)
    goal_pub.publish(goal)
    rospy.loginfo("Goal published")

if __name__ == '__main__':
    try:
        x_goal = 1.89
        y_goal = 0.318
        w_goal = 0.00632
        send_goal(x_goal, y_goal, w_goal)
    except rospy.ROSInterruptException:
        pass
