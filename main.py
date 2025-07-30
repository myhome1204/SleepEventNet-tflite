from src.utils import predict_with_tflite_model

def main(wav_file_path):
    predict_with_tflite_model(tflite_model_path='model/sleep_evnt_model_tflite.tflite',
                      wav_file_path=wav_file_path)

if __name__ == '__main__':
    # main의 인수로 원하는 소리 데이터의 경로를 넣어서 사용.
    wav_path = 'data/20_seconds/snoring/1/test3.wav'
    main(wav_path)