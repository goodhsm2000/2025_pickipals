#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf

class FloorClassifier:
    def __init__(self, model_dir, topic):
        """
        Initializes the FloorClassifier using a SavedModel (not a Keras .h5 or Keras model folder).
        Make sure the folder has 'saved_model.pb', 'variables/' etc.
        """
        # tf.saved_model.load() 로드
        self.saved_model = tf.saved_model.load(model_dir)

        # 보통 기본 서명은 "serving_default"임
        # 실제 서명 이름을 확인해야 할 수도 있습니다 (list(self.saved_model.signatures.keys()) 해보기).
        self.infer = self.saved_model.signatures["serving_default"]

        self.topic = topic
        self.bridge = CvBridge()
        self.latest_prediction = None

        # Initialize ROS node
        rospy.init_node('floor_classifier', anonymous=True)

        # Subscribe to the image topic
        rospy.Subscriber(self.topic, Image, self.callback)

    def callback(self, msg):
        """
        Callback function for ROS subscriber. Processes incoming images.
        """
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Preprocess the image for the model
            processed_image = self.preprocess_image(cv_image)

            # Predict the floor number
            self.latest_prediction = self.predict_floor(processed_image)

            # 디버깅용 출력
            print(f"[DEBUG] Predicted floor: {self.latest_prediction}")

        except Exception as e:
            print(f"[ERROR] Error in callback: {e}")

    def preprocess_image(self, image):
        """
        Preprocess the image for the model.
        :param image: Input image as a numpy array (BGR format).
        :return: Preprocessed image as a numpy array of shape (1, height, width, channels).
        """
        # 모델 입력 크기에 맞게 리사이즈 (예: 224x224)
        image_resized = cv2.resize(image, (224, 224))
        
        # 여기서 추가 정규화나 채널 변환(RGB) 등이 필요할 수도 있음
        # 예) image_resized = image_resized[..., ::-1]  # BGR→RGB 변환

        # (H, W, C) -> (1, H, W, C)
        return np.expand_dims(image_resized, axis=0)

    def predict_floor(self, processed_image):
        """
        Predict the floor number using the SavedModel signatures.

        :param processed_image: Numpy array of shape (1, 224, 224, 3).
        :return: Predicted floor number.
        """
        # 1) numpy -> 텐서로 변환
        input_tensor = tf.convert_to_tensor(processed_image, dtype=tf.float32)

        # 2) 서명(signature) 호출
        #    -> 결과는 dict 형태로 반환됨 (key가 "outputs", "predictions" 등이 될 수 있음)
        result = self.infer(input_tensor)

        # 어떤 키가 있는지 확인해보려면:
        # print("result keys:", result.keys())

        # 여기서는 임의로 첫 번째 값을 가져온다고 가정
        predictions = list(result.values())[0].numpy()

        # [DEBUG] 출력
        print(f"[DEBUG] Raw output from SavedModel: {predictions}")

        # 3) Argmax로 floor 분류
        predicted_floor = np.argmax(predictions, axis=-1)[0]  # (1,) 형태이므로 [0]으로 스칼라 추출

        return predicted_floor

    def get_latest_prediction(self):
        """
        Get the latest floor prediction.
        """
        return self.latest_prediction

if __name__ == "__main__":
    try:
        # SavedModel 디렉터리 (saved_model.pb가 있는 곳)
        MODEL_DIR = "/home/husky/catkin_ws/src/husky/husky_control/weights/efficientnet_b0_simple_model_4"
        
        # 이미지 토픽(예시)
        IMAGE_TOPIC = "/camera/color/image_raw"

        # Create the classifier
        classifier = FloorClassifier(MODEL_DIR, IMAGE_TOPIC)

        # Keep the node running
        rospy.spin()
    except rospy.ROSInterruptException:
        print("[INFO] Floor classification node terminated.")



# #!/usr/bin/env python3
# # -*- coding:utf-8 -*-

# import rospy
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import Normalization as BuiltinNormalization
# from tensorflow.keras.utils import register_keras_serializable
# from tensorflow.keras.optimizers import Adam

# # 1. Normalization 래퍼
# @register_keras_serializable(package="keras.layers.preprocessing.normalization", name="Normalization")
# class MyNormalization(BuiltinNormalization):
#     pass

# # 2. Optimizer 래퍼 (weight_decay를 무시하거나 원하는 방식으로 처리)
# @register_keras_serializable(package="Custom", name="NoDecayAdam")
# class NoDecayAdam(Adam):
#     def __init__(self, weight_decay=None, **kwargs):
#         super().__init__(**kwargs)
#         self.weight_decay = weight_decay  # 실제로 사용 안 해도 됨

#     def get_config(self):
#         config = super().get_config()
#         # config["weight_decay"] = self.weight_decay  # 필요하면 기록
#         return config

# class FloorClassifier:
#     def __init__(self, model_dir, topic):
#         self.model = load_model(
#             model_dir,
#             custom_objects={
#                 'Normalization': MyNormalization,
#                 'NoDecayAdam': NoDecayAdam  # 혹은 'Adam': MyAdam 등으로 맞춰야 할 수도 있음
#             }
#         )
#         self.topic = topic
#         self.bridge = CvBridge()
#         self.latest_prediction = None

#         rospy.init_node('floor_classifier', anonymous=True)
#         rospy.Subscriber(self.topic, Image, self.callback)

#     def callback(self, msg):
#         try:
#             print("[DEBUG] Received an image. Processing...")
#             cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
#             processed_image = self.preprocess_image(cv_image)
#             self.latest_prediction = self.predict_floor(processed_image)
#             print(f"[DEBUG] Latest predicted floor number: {self.latest_prediction}")
#         except Exception as e:
#             print(f"[ERROR] Error in callback: {e}")

#     def preprocess_image(self, image):
#         image_normalized = cv2.resize(image, (224, 224))
#         return np.expand_dims(image_normalized, axis=0)

#     def predict_floor(self, processed_image):
#         predictions = self.model.predict(processed_image)
#         print(f"[DEBUG] Raw model output (prediction vector): {predictions}")
#         predicted_floor = np.argmax(predictions)
#         print(f"[DEBUG] Argmax floor prediction: {predicted_floor}")
#         return predicted_floor

#     def get_latest_prediction(self):
#         return self.latest_prediction

# if __name__ == "__main__":
#     try:
#         MODEL_DIR = "/home/husky/catkin_ws/src/husky/husky_control/weights/efficientnet_b0_simple_model"
#         IMAGE_TOPIC = "/camera/color/image_raw"
#         classifier = FloorClassifier(MODEL_DIR, IMAGE_TOPIC)
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         print("[INFO] Floor classification node terminated.")

