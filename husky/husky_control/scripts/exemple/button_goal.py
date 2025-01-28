#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from kobuki_msgs.msg import Led, ButtonEvent

class ButtonGoalLEDControl:
    def __init__(self):
        rospy.init_node('button_goal_led_control')
        
        # Subscriber for button events
        self.button_sub = rospy.Subscriber('/mobile_base/events/button', ButtonEvent, self.button_callback)
        
        # Publisher for LED state (LED1 for TurtleBot2)
        self.led_pub = rospy.Publisher('/mobile_base/commands/led1', Led, queue_size=10)
        
        # Publisher for navigation goals
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        
        self.led_state = Led()  # Initialize LED message
        self.led_state.value = Led.BLACK  # Start with LED off

    def send_goal(self, x, y, w):
        rate = rospy.Rate(10)  # 10 Hz

        while self.goal_pub.get_num_connections() == 0:
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
        self.goal_pub.publish(goal)
        rospy.loginfo("Goal published")

    def button_callback(self, msg):
        if msg.state == ButtonEvent.RELEASED:
            if msg.button == ButtonEvent.Button0 or msg.button == ButtonEvent.Button1:
                rospy.loginfo("Button {} released! Turning LED off.".format(msg.button))
                self.led_state.value = Led.BLACK
        elif msg.state == ButtonEvent.PRESSED:  # ButtonEvent.PRESSED
            if msg.button == ButtonEvent.Button0:
                rospy.loginfo("Button 0 pressed! Turning LED red and sending goal.")
                self.led_state.value = Led.RED
                self.send_goal(1.86, 0.508, 0.00257)  # Replace with your desired goal coordinates and orientation
                rospy.sleep(10)
                 # Return to the initial position
                rospy.loginfo("Returning to the initial position")
                self.send_goal(-0.32,1.02, 0.408)  # Replace with your desired goal coordinates and orientation
            elif msg.button == ButtonEvent.Button1:
                rospy.loginfo("Button 1 pressed! Turning LED green and sending goal.")
                self.led_state.value = Led.GREEN
                self.send_goal(1.89, 0.318, 0.00632)  # Replace with your desired goal coordinates and orientation
        
        # Publish LED state
        self.led_pub.publish(self.led_state)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        controller = ButtonGoalLEDControl()
        controller.run()
    except rospy.ROSInterruptException:
        pass
