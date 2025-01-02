#! /usr/bin/env python3 
import rospy
import sounddevice as sd
from scipy.io.wavfile import write
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import onnxruntime as ort
import os
import subprocess  # 추가

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'sound_detect/best_model.onnx')

# 1. 오디오 녹음 및 저장
def record_audio(filename, duration=5, fs=16000):
    """
    오디오를 녹음하여 PCM 형식의 .wav 파일로 저장하는 함수
    Args:
        filename: 저장할 파일명 (경로 포함)
        duration: 녹음 시간 (초)
        fs: 샘플 레이트 (기본값 16kHz)
    """
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')  # PCM 형식으로 녹음 (int16)
    sd.wait()  # 녹음 종료를 기다림
    write(filename, fs, recording)  # PCM 형식으로 녹음된 파일을 저장
    #print(f"Recording saved as {filename}")
# 2. 오디오 파일 로드 및 전처리
def load_wav_16k_mono(filename):
    """
    16kHz로 리샘플링된 mono 오디오 파일을 불러오는 함수
    Args:
        filename: 불러올 .wav 파일 경로
    Returns:
        리샘플링된 오디오 데이터 (Tensor)
    """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav
def preprocess_wav(wav):
    """
    오디오 데이터를 스펙트로그램으로 변환하는 함수
    Args:
        wav: 오디오 Tensor
    Returns:
        스펙트로그램 numpy array
    """
    wav = wav[:80000]  # 길이를 5초(16kHz 샘플링일 경우 80,000 샘플)로 맞춤
    zero_padding = tf.zeros([80000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)  # (시간, 주파수, 1) 형태로 확장
    spectrogram = tf.image.resize(spectrogram, [500, 161])  # 모델 입력 크기에 맞춰 리사이즈
    spectrogram = tf.expand_dims(spectrogram, axis=0)  # 배치 차원 추가 
    return spectrogram.numpy()  # numpy array로 변환하여 반환

# 3. ONNX 모델을 불러와서 예측
def predict_audio(filename, model_path='best_model.onnx'):
    """
    오디오 파일을 불러와서 ONNX 모델로 pos/neg를 예측하는 함수
    Args:
        filename: 예측할 .wav 파일 경로
        model_path: 불러올 ONNX 모델 경로
    """
    # 오디오 파일 로드 및 전처리
    wav = load_wav_16k_mono(filename)
    spectrogram = preprocess_wav(wav)
    # ONNX 모델 세션 시작
    ort_session = ort.InferenceSession(model_path)
    # 예측 실행
    outputs = ort_session.run(None, {"input": spectrogram})
    prediction = outputs[0][0][0]  # 예측 결과 추출
    # 결과 해석
    if prediction >= 0.5:
        print(f"{filename}: Positive (Crying sound detected)")
        # subprocess를 사용하여 face_goal.py 실행
        script_path = os.path.join(script_dir, "move_bpm_test.py")
        subprocess.Popen(["python3", script_path])  # face_goal.py를 독립 프로세스로 실행
        #os.system("roslaunch turtlebot_control sound_face_detection.launch")
    else:
        print(f"{filename}: Negative (Noise detected)")

# 4. 전체 과정 실행
if __name__ == "__main__":
    # 5초 동안 오디오 녹음
    recorded_filename = 'recorded_audio.wav'
    record_audio(recorded_filename, duration=5)  # PCM 형식으로 바로 저장
    
    # ONNX 모델을 사용해 예측
    predict_audio(recorded_filename, model_path=model_path)