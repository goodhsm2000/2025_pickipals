#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist

def move_turtlebot():
    # Initialize the ROS node
    rospy.init_node('move_turtlebot', anonymous=True)
    # Create a publisher for the velocity commands
    velocity_publisher = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)
    # Set the publishing rate
    rate = rospy.Rate(10)  # 10 Hz

    # Create a Twist message and set the linear and angular velocities
    vel_msg = Twist()
    vel_msg.linear.x = 0.0
    vel_msg.angular.z = 1.0

    while not rospy.is_shutdown():
        # Publish the velocity command
        velocity_publisher.publish(vel_msg)
        # Sleep to maintain the desired rate
        rate.sleep()

if __name__ == '__main__':
    try:
        move_turtlebot()
    except rospy.ROSInterruptException:
        pass
