import tensorflow as tf
import numpy as np
import scipy.signal
from scipy.io import wavfile


LABELS = ['bruxism_score', 'snoring_score', 'cough_score', 'speech_score', 'is_normal']


def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) /
                               original_sample_rate * desired_sample_rate))
    waveform = scipy.signal.resample(waveform, desired_length)
  return desired_sample_rate, waveform


def look_from_data(sample_rate,wav_data):
    # sample_rate, wav_data는 이미 전달된 상태
    sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)  # 리샘플링
    # 만약 wav파일이 스테레오라면 스테레오 데이터를 모노로 변환
    if len(wav_data.shape) == 2:  # 스테레오 파일일 경우
        wav_data = np.mean(wav_data, axis=1)  # 두 채널을 평균 내어 모노로 변환
    # duration = len(wav_data) / sample_rate
    waveform = wav_data / tf.int16.max
    
    return waveform

def make_wave_form(wav_file_path):
    sample_rate, wav_data = wavfile.read(wav_file_path, 'rb')
    wav_data  = look_from_data(sample_rate,wav_data)
    waveform = tf.expand_dims(wav_data, axis=0)
    return waveform

def predict_with_tf_model(tensorflow_model, wav_file_path: str):
    waveform = make_wave_form(wav_file_path)
    predictions = tensorflow_model(waveform)  # (1, 5)
    values = predictions.numpy()[0]  # (5,)
    
    for label, value in zip(LABELS, values):
        print(f"{label}: {value:.4f}")
    print()
    
    return predictions

    
def predict_with_tflite_model(tflite_model_path :str, wav_file_path :str):
    waveform = make_wave_form(wav_file_path)
    waveform = tf.expand_dims(waveform, axis=0).numpy().astype(np.float32)

    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.resize_tensor_input(input_details[0]['index'], waveform.shape)
    interpreter.allocate_tensors()

    interpreter.set_tensor(input_details[0]['index'], waveform)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])  # shape: (1, 5)
    
    for i, label in enumerate(LABELS):
        print(f"{label}: {output[0][i] :.4f}")
    print()
