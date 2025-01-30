#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import rospy
import tf.transformations

def quaternion_to_euler(x, y, z, w):
    """
    x, y, z, w = 쿼터니언(Quaternion) 구성 요소
    return = (roll, pitch, yaw) in radians
    """
    # quaternion을 파이썬 튜플로 만듦
    quaternion = (x, y, z, w)
    # tf의 euler_from_quaternion 이용
    roll, pitch, yaw = tf.transformations.euler_from_quaternion(quaternion)
    return (roll, pitch, yaw)

if __name__ == "__main__":
    # 예시 쿼터니언
    qx, qy, qz, qw = 1.0, 0.0, 0.0, 0.0  # 단위 쿼터니언(회전 없음)

    roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)

    # 라디안 출력
    print("roll (rad): ", roll)
    print("pitch(rad): ", pitch)
    print("yaw  (rad): ", yaw)

    # 도(deg) 단위로 변환하여 출력
    roll_deg  = roll  * 180.0 / math.pi
    pitch_deg = pitch * 180.0 / math.pi
    yaw_deg   = yaw   * 180.0 / math.pi

    print("roll (deg): ", roll_deg)
    print("pitch(deg): ", pitch_deg)
    print("yaw  (deg): ", yaw_deg)
