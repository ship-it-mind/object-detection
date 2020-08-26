from model_utils import *

from PIL import Image 
import cloudpickle
import tensorflow_datasets
import tensorflow as tf
import cv2
import numpy as np


with open("weights/int2str.pkl", "rb") as f:
    int2str = cloudpickle.load(f)

def load_model():
    resnet50_backbone = get_backbone()
    loss_fn = RetinaNetLoss(80)
    model = RetinaNet(80, resnet50_backbone)
    optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
    model.compile(loss=loss_fn, optimizer=optimizer)
    latest_checkpoint = tf.train.latest_checkpoint("weights")
    model.load_weights(latest_checkpoint)
    image = tf.keras.Input(shape=[None, None, 3], name="image")
    predictions = model(image, training=False)
    detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
    inference_model = tf.keras.Model(inputs=image, outputs=detections)
    return inference_model

def predict(image_b, inference_model):
    image_t = tf.io.decode_image(image_b)
    image = tf.cast(image_t, dtype=tf.float32)
    input_image, ratio = prepare_image(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    class_names = [
        int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
    ]
    res_image = visualize_detections(
        image,
        detections.nmsed_boxes[0][:num_detections] / ratio,
        class_names,
        detections.nmsed_scores[0][:num_detections],
    )
    print(type(res_image), len(res_image), res_image[:20])
    return res_image
    

if __name__ == '__main__':
    model = load_model()
    # new_image = Image.open("11.jpg")
    # new_image = np.array(new_image)
    with open("111.jpg", "rb") as f:
    #     print("hhhhhhhhhhhhhhhhhhhhhhhhhhh", type(f.read()))
        new_image = f.read()
    predict(new_image, model)