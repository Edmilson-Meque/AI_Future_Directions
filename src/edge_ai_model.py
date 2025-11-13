# Funções de inferência para o modelo TFLite
import tensorflow as tf
import numpy as np
from PIL import Image

def load_tflite_model(path="../models/model.tflite"):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

def predict_image(interpreter, image_path, img_height=64, img_width=64):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = Image.open(image_path).resize((img_height, img_width))
    input_data = np.expand_dims(np.array(img)/255.0, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data[0])
    return predicted_class
