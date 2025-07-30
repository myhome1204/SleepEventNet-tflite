from .model_layers import SleepEventClassificationLayer, YamnetWrapper
import tensorflow as tf
import os

def build_tensorflow_model(yamnet_model_path : str ,save_path: str):
    yamnet_layer = YamnetWrapper(yamnet_model_path=yamnet_model_path)
    input_audio = tf.keras.Input(shape=(None,), dtype=tf.float32, name='waveform')
    scores, embeddings, spectrogram = yamnet_layer(input_audio)
    sleep_event_classification_layer = SleepEventClassificationLayer()
    output_score = sleep_event_classification_layer(scores)
    model = tf.keras.Model(inputs=[input_audio], outputs=[output_score])
    model.summary()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    return model

def load_tensorflow_model(tensorflow_model_path : str):
    model = tf.keras.models.load_model(
    tensorflow_model_path,
    custom_objects={"SleepEventClassificationLayer": SleepEventClassificationLayer}
    )
    return model 

def load_tensorflow_model_pb(tensorflow_model_path: str):
    model = tf.keras.models.load_model(
        tensorflow_model_path,
        custom_objects={
            "YamnetWrapper": YamnetWrapper,
            "SleepEventClassificationLayer": SleepEventClassificationLayer
        }
    )
    return model
def convert_tflite(tensorflow_model_path: str , save_path: str):
    model = load_tensorflow_model(tensorflow_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(save_path, 'wb') as f:
        f.write(tflite_model)