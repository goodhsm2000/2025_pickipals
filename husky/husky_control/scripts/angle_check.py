#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import tf.transformations as t

def quaternion_to_euler_deg(x, y, z, w):
    """
    쿼터니언(x, y, z, w)을 입력받아
    (roll, pitch, yaw) [도(deg)] 형태로 반환한다.
    """
    roll_rad, pitch_rad, yaw_rad = t.euler_from_quaternion([x, y, z, w])
    roll_deg = math.degrees(roll_rad)
    pitch_deg = math.degrees(pitch_rad)
    yaw_deg = math.degrees(yaw_rad)
    return roll_deg, pitch_deg, yaw_deg

def euler_deg_to_quaternion(roll_deg, pitch_deg, yaw_deg):
    """
    오일러 각(roll, pitch, yaw) [도(deg)]를 입력받아
    쿼터니언(x, y, z, w)을 반환한다.
    """
    roll_rad = math.radians(roll_deg)
    pitch_rad = math.radians(pitch_deg)
    yaw_rad = math.radians(yaw_deg)
    return t.quaternion_from_euler(roll_rad, pitch_rad, yaw_rad)

if __name__ == '__main__':
    # 예시 1) Quaternion -> Euler(deg)
    print("=== 테스트 1: 쿼터니언 -> 오일러(deg) ===")
    qx, qy, qz, qw = 0.0, 0.712, -0.702, 0.0 # 대략 yaw 90도에 해당
    roll, pitch, yaw = quaternion_to_euler_deg(qx, qy, qz, qw)
    print(f"Quaternion: x={qx}, y={qy}, z={qz}, w={qw}")
    print(f"Roll:  {roll:.2f} deg, Pitch: {pitch:.2f} deg, Yaw: {yaw:.2f} deg")
    print()

    # 예시 2) Euler(deg) -> Quaternion
    print("=== 테스트 2: 오일러(deg) -> 쿼터니언 ===")
    roll_deg_test, pitch_deg_test, yaw_deg_test = (0.0, 0.0, -90.0)
    x, y, z, w = euler_deg_to_quaternion(roll_deg_test, pitch_deg_test, yaw_deg_test)
    print(f"Euler(deg): Roll={roll_deg_test}, Pitch={pitch_deg_test}, Yaw={yaw_deg_test}")
    print(f"Quaternion: x={x:.4f}, y={y:.4f}, z={z:.4f}, w={w:.4f}")
    print()

