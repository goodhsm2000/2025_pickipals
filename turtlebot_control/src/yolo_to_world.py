#!/usr/bin/env python3
import sys
import rospy
import torch
import cv2
import numpy as np
from pathlib import Path
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import tf  # Use tf instead of tf2_ros
from geometry_msgs.msg import PointStamped

# Add YOLOv5 to path
FILE = Path(__file__).resolve()
ROOT = Path('/home/turtle/yolov5').resolve()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
from utils.augmentations import letterbox

class ImageSubscriber:
    def __init__(self, rgb_topic='/camera/rgb/image_raw', depth_topic='/camera/depth/image_raw'):
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None

        # Subscribers
        rgb_sub = message_filters.Subscriber(rgb_topic, Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)

        # Synchronize topics
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.5)
        ts.registerCallback(self.callback)

    def callback(self, rgb_data, depth_data):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(rgb_data, 'bgr8')
            self.depth_image = self.bridge.imgmsg_to_cv2(depth_data, desired_encoding='passthrough')
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

class CameraInfoSubscriber:
    def __init__(self, camera_info_topic='/camera/rgb/camera_info'):
        self.intrinsic_matrix = None
        rospy.Subscriber(camera_info_topic, CameraInfo, self.callback)

    def callback(self, msg):
        self.intrinsic_matrix = np.array(msg.K).reshape(3, 3)

def compute_3d_point(u, v, depth_value, intrinsic_matrix):
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    X = (u - cx) * depth_value / fx
    Y = (v - cy) * depth_value / fy
    Z = depth_value

    point_3d = np.array([X, Y, Z], dtype=np.float32)
    return point_3d

def main():
    rospy.init_node('yolov5_ros_node', anonymous=True)
    device = select_device('')
    model = DetectMultiBackend('/home/turtle/yolov5/yolov5s.pt', device=device)
    stride, names = model.stride, model.names
    imgsz = (640, 640)

    image_subscriber = ImageSubscriber()
    camera_info_subscriber = CameraInfoSubscriber()
    rate = rospy.Rate(10)

    # Initialize tf listener
    tf_listener = tf.TransformListener()

    model.warmup(imgsz=(1, 3, *imgsz))

    while not rospy.is_shutdown():
        if (image_subscriber.rgb_image is not None and
            image_subscriber.depth_image is not None and
            camera_info_subscriber.intrinsic_matrix is not None):

            img0 = image_subscriber.rgb_image.copy()
            depth0 = image_subscriber.depth_image.copy()
            intrinsic_matrix = camera_info_subscriber.intrinsic_matrix

            # Preprocess
            img = letterbox(img0, imgsz, stride=stride, auto=False)[0]
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)

            img_tensor = torch.from_numpy(img).to(device)
            img_tensor = img_tensor.float() / 255.0  # normalize
            if img_tensor.ndimension() == 3:
                img_tensor = img_tensor.unsqueeze(0)

            # Inference
            with torch.no_grad():
                pred = model(img_tensor)

            # NMS
            pred = non_max_suppression(pred, 0.25, 0.45, max_det=1000)

            # Process detections
            annotator = Annotator(img0, line_width=2, example=str(names))
            for det in pred:
                if len(det):
                    det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)
                        label = f'{names[c]} {conf:.2f}'

                        # Bounding box coordinates
                        x1, y1, x2, y2 = map(int, xyxy)
                        center_u = int((x1 + x2) / 2)
                        center_v = int((y1 + y2) / 2)

                        # Ensure coordinates are within image bounds
                        center_u = np.clip(center_u, 0, depth0.shape[1] - 1)
                        center_v = np.clip(center_v, 0, depth0.shape[0] - 1)

                        # Retrieve depth value
                        depth_value = depth0[center_v, center_u]

                        # Handle invalid depth
                        if depth_value == 0 or np.isnan(depth_value) or np.isinf(depth_value):
                            depth_text = 'Depth: N/A'
                            display_text = f'{label} {depth_text}'
                            annotator.box_label(xyxy, display_text, color=colors(c, True))
                            continue

                        # Compute 3D point in camera frame
                        point_3d = compute_3d_point(center_u, center_v, depth_value, intrinsic_matrix)

                        rospy.loginfo(f"Camera coordinates: X={point_3d[0]}, Y={point_3d[1]}, Z={point_3d[2]}")

                        try:
                            # Wait for the transform to be available
                            tf_listener.waitForTransform('odom', 'camera_link', rospy.Time(0), rospy.Duration(1.0))

                            # Get transform from camera_link to odom
                            (trans, rot) = tf_listener.lookupTransform('odom', 'camera_link', rospy.Time(0))

                            # Create transformation matrix
                            rotation_matrix = tf.transformations.quaternion_matrix(rot)
                            transformation_matrix = rotation_matrix.copy()
                            transformation_matrix[0:3, 3] = trans

                            # Convert point to homogeneous coordinates
                            point_camera_hom = np.array([point_3d[0], point_3d[1], point_3d[2], 1])

                            # Transform point to odom frame
                            point_world = np.dot(transformation_matrix, point_camera_hom)
                            X_w, Y_w, Z_w = point_world[:3]

                            rospy.loginfo(f"World coordinates: X={X_w}, Y={Y_w}, Z={Z_w}")

                            # Display the odom coordinates
                            odom_text = f'Odom: ({X_w:.2f}, {Y_w:.2f}, {Z_w:.2f}) m'
                            display_text = f'{label}\n{odom_text}'

                            # Draw bounding box and label
                            annotator.box_label(xyxy, display_text, color=colors(c, True))

                        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                            rospy.logerr(f"Transform failed: {e}")
                            depth_text = 'Depth: N/A'
                            display_text = f'{label} {depth_text}'
                            annotator.box_label(xyxy, display_text, color=colors(c, True))
                            continue

            img0 = annotator.result()

            # Display
            cv2.imshow('YOLOv5 ROS', img0)
            if cv2.waitKey(1) == ord('q'):
                break

        else:
            rospy.logwarn('Waiting for images and camera info...')
        rate.sleep()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
