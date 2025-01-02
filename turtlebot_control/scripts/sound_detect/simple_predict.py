#! /usr/bin/env python3

import rospy
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import onnxruntime as ort
import os
from std_msgs.msg import Bool, String

# Initialize ROS node
rospy.init_node('sound_detect')  # Node name updated to match Launch file
control_pub = rospy.Publisher('/control_astra_twist_ppgi', Bool, queue_size=10)

# Get script directory and model path
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'sound_detect/best_model.onnx')  # Ensure this path is correct
recorded_filename = os.path.join(script_dir, 'recorded_audio.wav')

# Initialize ONNX Runtime session once
try:
    ort_session = ort.InferenceSession(model_path)
except Exception as e:
    rospy.logerr(f"Failed to load ONNX model at {model_path}: {e}")
    exit(1)

# Initialize state
current_state = "IDLE"  # Default state
last_published_state = None  # Track last published state to prevent duplicates

# State callback
def state_callback(msg):
    global current_state
    current_state = msg.data

# Add subscriber for state
state_sub = rospy.Subscriber('/astra_twist_ppgi/state', String, state_callback)

# 1. 오디오 녹음 및 저장
def record_audio(filename, duration=5, fs=16000):
    """
    오디오를 녹음하여 PCM 형식의 .wav 파일로 저장하는 함수
    Args:
        filename: 저장할 파일명 (경로 포함)
        duration: 녹음 시간 (초)
        fs: 샘플 레이트 (기본값 16kHz)
    """
    rospy.loginfo("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')  # PCM 형식으로 녹음 (int16)
    sd.wait()  # 녹음 종료를 기다림
    write(filename, fs, recording)  # PCM 형식으로 녹음된 파일을 저장
    rospy.loginfo(f"Recording saved as {filename}")

# 2. 오디오 파일 로드 및 전처리
def preprocess_wav(filename):
    """
    오디오 데이터를 스펙트로그램으로 변환하는 함수
    Args:
        filename: 오디오 파일 경로
    Returns:
        스펙트로그램 numpy array (1, 500, 161, 1)
    """
    # Load audio with librosa
    wav, sr = librosa.load(filename, sr=16000, mono=True, duration=5.0)
    # Ensure length is 5 seconds (80000 samples at 16kHz)
    if len(wav) < 80000:
        wav = np.pad(wav, (0, max(0, 80000 - len(wav))), 'constant')
    else:
        wav = wav[:80000]
    # Compute spectrogram
    spectrogram = librosa.stft(wav, n_fft=320, hop_length=32, win_length=320)
    spectrogram = np.abs(spectrogram)
    # Fix length to [500, 161]
    spectrogram = librosa.util.fix_length(spectrogram, 500, axis=1)  # 시간 축
    spectrogram = librosa.util.fix_length(spectrogram, 161, axis=0)  # 주파수 축
    spectrogram = spectrogram.astype(np.float32)
    # Expand dimensions to match model input
    spectrogram = np.expand_dims(spectrogram, axis=-1)  # (161, 500, 1)
    spectrogram = np.expand_dims(spectrogram, axis=0)  # (1, 161, 500, 1)
    spectrogram = spectrogram.transpose(0, 2, 1, 3)    # (1, 500, 161, 1)
    return spectrogram

# 3. ONNX 모델을 불러와서 예측
def predict_audio(filename):
    """
    오디오 파일을 불러와서 ONNX 모델로 pos/neg를 예측하는 함수
    Args:
        filename: 예측할 .wav 파일 경로
    """
    global last_published_state
    # 오디오 파일 전처리
    spectrogram = preprocess_wav(filename)
    
    # ONNX 모델을 글로벌 변수에서 사용
    try:
        inputs = {ort_session.get_inputs()[0].name: spectrogram}
        outputs = ort_session.run(None, inputs)
        prediction = outputs[0][0][0]  # 예측 결과 추출
    except Exception as e:
        rospy.logerr(f"ONNX Runtime Error: {e}")
        return
    
    # 결과 해석
    if prediction >= 0.5:
        rospy.loginfo(f"{filename}: Positive (Crying sound detected)")
        if last_published_state != True:
            control_pub.publish(Bool(data=True))  # Send start command
            last_published_state = True
    else:
        rospy.loginfo(f"{filename}: Negative (Noise detected)")
        if last_published_state != False:
            control_pub.publish(Bool(data=False))  # Send stop command
            last_published_state = False

# 4. 메인 루프
def sound_detection_loop():
    rate = rospy.Rate(0.2)  # 0.2 Hz (every 5 seconds)
    while not rospy.is_shutdown():
        if current_state == "IDLE":
            # 오디오 녹음
            record_audio(recorded_filename, duration=5)
            # 오디오 예측
            predict_audio(recorded_filename)
        else:
            rospy.loginfo(f"astra_twist_ppgi is in state '{current_state}'. Skipping sound detection.")
        rate.sleep()

if __name__ == "__main__":
    try:
        sound_detection_loop()
    except rospy.ROSInterruptException:
        pass
