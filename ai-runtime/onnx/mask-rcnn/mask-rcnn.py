# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datetime import datetime
import io
import json
import os
import logging
import time
from typing import Tuple

from flask import Flask, Response, Request, abort, request
import numpy as np
import onnxruntime
from PIL import Image, ImageDraw, ImageFont
import requests


class MaskRCNNModel:
    def __init__(self):
        model_path = 'MaskRCNN-10.onnx'
        self._session = onnxruntime.InferenceSession(model_path)

        tags_file = 'tags.txt'
        self.tags = [line.rstrip('\n') for line in open(tags_file)]

    def preprocess(self, image: Image) -> np.ndarray:
        # Resize
        ratio = 800.0 / min(image.size[0], image.size[1])
        image = image.resize((int(ratio * image.size[0]), int(ratio * image.size[1])), Image.BILINEAR)

        # Convert to BGR
        image = np.array(image)[:, :, [2, 1, 0]].astype('float32')

        # HWC -> CHW
        image = np.transpose(image, [2, 0, 1])

        # Normalize
        mean_vec = np.array([102.9801, 115.9465, 122.7717])
        for i in range(image.shape[0]):
            image[i, :, :] = image[i, :, :] - mean_vec[i]

        # Pad to be divisible of 32
        import math
        padded_h = int(math.ceil(image.shape[1] / 32) * 32)
        padded_w = int(math.ceil(image.shape[2] / 32) * 32)

        padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
        padded_image[:, :image.shape[1], :image.shape[2]] = image
        image = padded_image

        return image

    def postprocess(self, boxes, labels, scores, image: Image, confidenceThreshold: float = 0.0) -> list:
        detected_objects = []
        for box, label, score in zip(boxes, labels, scores):
            if score > confidenceThreshold:
                dobj = {
                    "type": "entity",
                    "entity": {
                        "tag": {
                            "value": self.tags[label],
                            "confidence": score.tolist()
                        },
                        "box": {
                            "l": box[0].tolist(),
                            "t": box[1].tolist(),
                            "w": (box[2] - box[0]).tolist(),
                            "h": (box[3] - box[1]).tolist()
                        }
                    }
                }
                detected_objects.append(dobj)
        return detected_objects

    def process_image(self, image: Image, object_type: str = None, confidence_threshold: float = 0.0) -> Tuple[
        list, float]:
        # Preprocess input according to the functions specified above
        image_data = self.preprocess(image)

        inference_time_start = time.time()
        boxes, labels, scores, masks = self._session.run(None, {"image": image_data})
        inference_time_end = time.time()
        inference_duration_s = inference_time_end - inference_time_start

        detected_objects = self.postprocess(boxes, labels, scores, image, confidence_threshold)
        return detected_objects, inference_duration_s


def init_logging():
    gunicorn_logger = logging.getLogger('gunicorn.error')
    if gunicorn_logger != None:
        app.logger.handlers = gunicorn_logger.handlers
        app.logger.setLevel(gunicorn_logger.level)


def draw_bounding_boxes(image: Image, detected_objects: list):
    objects_identified = len(detected_objects)

    draw = ImageDraw.Draw(image)

    textfont = ImageFont.load_default()

    for pos in range(objects_identified):
        entity = detected_objects[pos]['entity']
        box = entity["box"]
        x1 = box["l"]
        y1 = box["t"]
        x2 = box["w"]
        y2 = box["h"]

        x2 = x1 + x2
        y2 = y1 + y2
        tag = entity['tag']
        object_class = tag['value']

        draw.rectangle((x1, y1, x2, y2), outline='blue', width=1)
        draw.text((x1, y1), str(object_class), fill="white", font=textfont)

    return image


def load_image(request: Request):
    try:
        image_data = io.BytesIO(request.get_data())
        image = Image.open(image_data)
        return image
    except Exception:
        abort(Response(response='Could not decode image', status=400))


app = Flask(__name__)

init_logging()

model = MaskRCNNModel()
app.logger.info('Model initialized')


# / routes to the default function which returns 'Hello World'
@app.route('/', methods=['GET'])
def default_page():
    return Response(response='Hello from Yolov3 inferencing based on ONNX', status=200)


@app.route('/stream/<id>')
def stream(id):
    respBody = ("<html>"
                "<h1>Stream with inferencing overlays</h1>"
                "<img src=\"/mjpeg/" + id + "\"/>"
                                            "</html>")

    return Response(respBody, status=200)


# /score routes to scoring function
# This function returns a JSON object with inference duration and detected objects
@app.route("/score", methods=['POST'])
def score():
    confidence = request.args.get('confidence', default=0.0, type=float)
    object_type = request.args.get('object')
    stream = request.args.get('stream')

    image = load_image(request)
    detected_objects, inference_duration_s = model.process_image(image, object_type, confidence)

    if stream is not None:
        output_image = draw_bounding_boxes(image, detected_objects)

        image_buffer = io.BytesIO()
        output_image.save(image_buffer, format='JPEG')

        # post the image with bounding boxes so that it can be viewed as an MJPEG stream
        postData = b'--boundary\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + image_buffer.getvalue() + b'\r\n'
        requests.post('http://127.0.0.1:80/mjpeg_pub/' + stream, data=postData)

    if len(detected_objects) > 0:
        respBody = {
            "inferences": detected_objects,
            "cost": inference_duration_s
        }

        respBody = json.dumps(respBody)
        return Response(respBody, status=200, mimetype='application/json')
    else:
        return Response(status=204)


# /score-debug routes to score_debug
# This function scores the image and stores an annotated image for debugging purposes
@app.route('/score-debug', methods=['POST'])
def score_debug():
    image = load_image(request)

    detected_objects, inference_duration_s = model.process_image(image)
    app.logger.info('Inference took %.2f seconds', inference_duration_s)

    output_image = draw_bounding_boxes(image, detected_objects)

    # datetime object containing current date and time
    now = datetime.now()

    output_dir = 'images'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_image_file = now.strftime("%d_%m_%Y_%H_%M_%S.jpeg")
    output_image.save(output_dir + "/" + output_image_file)

    respBody = {
        "inferences": detected_objects
    }

    return respBody


# /annotate routes to annotation function
# This function returns an image with bounding boxes drawn around detected objects
@app.route('/annotate', methods=['POST'])
def annotate():
    confidence = request.args.get('confidence', default=0.0, type=float)
    object_type = request.args.get('object')
    image = load_image(request)

    detected_objects, inference_duration_s = model.process_image(image, object_type, confidence)
    app.logger.info('Inference took %.2f seconds', inference_duration_s)

    image = draw_bounding_boxes(image, detected_objects)

    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes = image_bytes.getvalue()

    return Response(response=image_bytes, status=200, mimetype="image/jpeg")


if __name__ == '__main__':
    # Running the file directly
    app.run(host='0.0.0.0', port=8888)
