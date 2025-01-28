#!/usr/bin/env python3
import rospy
from robotis_controller_msgs.msg import SyncWriteItem

def torque_on():
    """토크 활성화"""
    pub = rospy.Publisher('/robotis/sync_write_item', SyncWriteItem, queue_size=10)
    rospy.sleep(1)

    msg = SyncWriteItem()
    msg.item_name = 'torque_enable'
    msg.joint_name = ['gripper']
    msg.value = [1]  # 1: Torque On

    rospy.loginfo("Turning torque on for gripper")
    pub.publish(msg)
    rospy.sleep(1)

def gripper_open():
    """그리퍼 열기"""
    pub = rospy.Publisher('/robotis/direct/sync_write_item', SyncWriteItem, queue_size=10)
    rospy.sleep(1)

    msg = SyncWriteItem()
    msg.item_name = 'goal_position'
    msg.joint_name = ['gripper']
    msg.value = [0]  # Open position

    rospy.loginfo("Opening gripper")
    pub.publish(msg)
    rospy.sleep(1)

def gripper_close():
    """그리퍼 닫기"""
    pub = rospy.Publisher('/robotis/direct/sync_write_item', SyncWriteItem, queue_size=10)
    rospy.sleep(1)

    msg = SyncWriteItem()
    msg.item_name = 'goal_position'
    msg.joint_name = ['gripper']
    msg.value = [740]  # Close position

    rospy.loginfo("Closing gripper")
    pub.publish(msg)
    rospy.sleep(1)

def main():
    rospy.init_node('gripper_control_node', anonymous=True)

    # 1. 토크 활성화
    torque_on()

    while not rospy.is_shutdown():
        # 2. 그리퍼 열기
        gripper_open()
        rospy.sleep(2)

        # 3. 그리퍼 닫기
        gripper_close()
        rospy.sleep(2)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass