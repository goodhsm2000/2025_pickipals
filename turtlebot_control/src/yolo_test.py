#!/usr/bin/env python3
import sys
import rospy
import torch
import cv2
import numpy as np
from pathlib import Path
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters

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
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.callback)

    def callback(self, rgb_data, depth_data):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(rgb_data, 'bgr8')
            self.depth_image = self.bridge.imgmsg_to_cv2(depth_data, desired_encoding='32FC1')
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

def main():
    rospy.init_node('yolov5_ros_node', anonymous=True)
    device = select_device('')
    model = DetectMultiBackend('/home/turtle/yolov5/yolov5s.pt', device=device)
    stride, names = model.stride, model.names
    imgsz = (640, 640)
    image_subscriber = ImageSubscriber()
    rate = rospy.Rate(10)

    model.warmup(imgsz=(1, 3, *imgsz))

    while not rospy.is_shutdown():
        if image_subscriber.rgb_image is not None and image_subscriber.depth_image is not None:
            img0 = image_subscriber.rgb_image.copy()
            depth0 = image_subscriber.depth_image.copy()

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

                        # Calculate center coordinates
                        x1, y1, x2, y2 = map(int, xyxy)
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)

                        # Ensure coordinates are within image bounds
                        center_x = np.clip(center_x, 0, depth0.shape[1] - 1)
                        center_y = np.clip(center_y, 0, depth0.shape[0] - 1)

                        # Retrieve depth value
                        depth_value = depth0[center_y, center_x]/1000

                        # Handle invalid depth
                        if np.isnan(depth_value) or depth_value <= 0:
                            depth_text = 'Depth: N/A'
                        else:
                            depth_text = f'Depth: {depth_value:.2f} m'

                        # Combine label and depth text
                        display_text = f'{label} {depth_text}'

                        # Draw bounding box and label
                        annotator.box_label(xyxy, display_text, color=colors(c, True))

            img0 = annotator.result()

            # Display
            cv2.imshow('YOLOv5 ROS', img0)
            if cv2.waitKey(1) == ord('q'):
                break

        rate.sleep()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
