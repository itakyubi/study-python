# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import math
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


class SSDModel:
    def __init__(self):
        model_path = 'ssd-10.onnx'
        self._session = onnxruntime.InferenceSession(model_path)
        tags_file = 'tags.txt'
        self.tags = [line.rstrip('\n') for line in open(tags_file)]

    def preprocess(self, image: Image) -> np.ndarray:
        image = image.resize((1200, 1200), Image.BILINEAR)
        img_data = np.array(image)
        img_data = np.transpose(img_data, [2, 0, 1])
        img_data = np.expand_dims(img_data, 0)
        mean_vec = np.array([0.485, 0.456, 0.406])
        stddev_vec = np.array([0.229, 0.224, 0.225])
        norm_img_data = np.zeros(img_data.shape).astype('float32')
        for i in range(img_data.shape[1]):
            norm_img_data[:, i, :, :] = (img_data[:, i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]
        return norm_img_data

    def postprocess(self, boxes, labels, scores, image: Image, confidenceThreshold: float = 0.0) -> list:
        detected_objects = []
        for i in range(0, scores[0].size):
            score = scores[0][i]
            if score > confidenceThreshold:
                box = boxes[0][i]
                dobj = {
                    "type": "entity",
                    "entity": {
                        "tag": {
                            "value": self.tags[labels[0][i]],
                            "confidence": score.tolist()
                        },
                        "box": {
                            "l": max(0, box[0]) * image.width,
                            "t": max(0, box[1]) * image.height,
                            "w": (box[2] - box[0]) * image.width,
                            "h": (box[3] - box[1]) * image.height
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
        boxes, labels, scores = self._session.run(None, {"image": image_data})
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

model = SSDModel()
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
    app.run(host='0.0.0.0', port=8889)
