#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from kobuki_msgs.msg import Led, ButtonEvent

class ButtonLEDControl:
    def __init__(self):
        rospy.init_node('button_led_control')
        
        # Subscriber for button events
        self.button_sub = rospy.Subscriber('/mobile_base/events/button', ButtonEvent, self.button_callback)
        
        # Publisher for LED state (LED1 for TurtleBot2)
        self.led_pub = rospy.Publisher('/mobile_base/commands/led1', Led, queue_size=10)

        # Publisher for velocity commands
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)
        
        self.led_state = Led()  # Initialize LED message
        self.led_state.value = Led.BLACK  # Start with LED off
        
        self.current_angular_velocity = 0.0  # Initialize the current angular velocity
        self.twist = Twist()  # Initialize Twist message for velocity

        # Set up rate for continuous publishing
        self.rate = rospy.Rate(10)  # 10 Hz, adjust this as needed

    def button_callback(self, msg):
        if msg.state == ButtonEvent.PRESSED:
            if msg.button == ButtonEvent.Button0:
                self.current_angular_velocity -= 0.1
                rospy.loginfo("Button 0 pressed! Decreasing angular velocity to {:.1f} rad/s.".format(self.current_angular_velocity))
                self.led_state.value = Led.RED
            elif msg.button == ButtonEvent.Button1:
                self.current_angular_velocity += 0.1
                rospy.loginfo("Button 1 pressed! Increasing angular velocity to {:.1f} rad/s.".format(self.current_angular_velocity))
                self.led_state.value = Led.GREEN
            
            # Ensure LED status is updated based on button pressed
            self.led_pub.publish(self.led_state)
            
        elif msg.state == ButtonEvent.RELEASED:
            rospy.loginfo("Button {} released!".format(msg.button))
            # No change in LED or velocity when button is released

    def run(self):
        while not rospy.is_shutdown():
            # Publish the current velocity continuously
            self.twist.angular.z = self.current_angular_velocity
            self.vel_pub.publish(self.twist)
            #self.rate.sleep()

if __name__ == '__main__':
    try:
        controller = ButtonLEDControl()
        controller.run()
    except rospy.ROSInterruptException:
        pass