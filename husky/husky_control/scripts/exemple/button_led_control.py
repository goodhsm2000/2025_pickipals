#!/usr/bin/env python

import rospy
from kobuki_msgs.msg import Led, ButtonEvent

class ButtonLEDControl:
    def __init__(self):
        rospy.init_node('button_led_control')
        
        # Subscriber for button events
        self.button_sub = rospy.Subscriber('/mobile_base/events/button', ButtonEvent, self.button_callback)
        
        # Publisher for LED state (LED1 for TurtleBot2)
        self.led_pub = rospy.Publisher('/mobile_base/commands/led1', Led, queue_size=10)
        
        self.led_state = Led()  # Initialize LED message
        self.led_state.value = Led.BLACK  # Start with LED off
        
    def button_callback(self, msg):
        if msg.state == ButtonEvent.RELEASED:
          if msg.button==ButtonEvent.Button0 or msg.button==ButtonEvent.Button1:
            rospy.loginfo("Button {} released! Turning LED off.".format(msg.button))
            self.led_state.value = Led.BLACK
        elif msg.state == ButtonEvent.PRESSED:  # ButtonEvent.PRESSED
             if msg.button == ButtonEvent.Button0:
                rospy.loginfo("Button 0 pressed! Turning LED red.")
                self.led_state.value = Led.RED
             elif msg.button == ButtonEvent.Button1:
                rospy.loginfo("Button 1 pressed! Turning LED green.")
                self.led_state.value = Led.GREEN
        
        
        # Publish LED state
        self.led_pub.publish(self.led_state)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        controller = ButtonLEDControl()
        controller.run()
    except rospy.ROSInterruptException:
        pass
