import io
import json
import os
import sys
import time
from typing import Tuple

import numpy as np
import onnxruntime
from PIL import Image, ImageDraw, ImageFont
from cryptography import x509
from cryptography.hazmat._oid import NameOID
from cryptography.hazmat.backends import default_backend
from flask import Flask, Response, Request, abort, request
import yaml
import math


class Model:
    def __init__(self):
        model_path = 'res/model.onnx'
        device = onnxruntime.get_device()
        if device == "GPU":
            self._session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        elif device == "CPU":
            self._session = onnxruntime.InferenceSession(model_path)
        else:
            print("unknown device")
            return

        tags_file = 'res/tags.txt'
        self._tags = [line.rstrip('\n') for line in open(tags_file)]

        f = open("./conf/conf.yml", 'r', encoding='utf-8')
        params = yaml.load(f.read(), Loader=yaml.UnsafeLoader)
        self._channelOrder = params['channelOrder'] if 'channelOrder' in params else None
        self._colorFormat = params['colorFormat'] if 'colorFormat' in params else None
        self._imageNorm = params['imageNorm'] if 'imageNorm' in params else 1
        self._imageNormMean = params['imageNormMean'] if 'imageNormMean' in params else [0, 0, 0]
        self._imageNormStd = params['imageNormStd'] if 'imageNormStd' in params else [1, 1, 1]
        self._imageResize = params['imageResize'] if 'imageResize' in params else None
        self._imageResizePad = params['imageResizePad'] if 'imageResizePad' in params else None
        self._expandDim = params['expandDim'] if 'expandDim' in params else None
        self._imageDateType = params['imageDateType'] if 'imageDateType' in params else None
        self._modelType = params['modelType']

    def resize_image(self, image: Image):
        if self._imageResize is not None:
            if self._imageResizePad == "padding":
                iw, ih = image.size
                w, h = self._imageResize
                scale = min(w / iw, h / ih)
                nw = int(iw * scale)
                nh = int(ih * scale)

                image = image.resize((nw, nh), Image.BICUBIC)
                new_image = Image.new('RGB', self._imageResize, (128, 128, 128))
                new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
                image = new_image
            else:
                image = image.resize(self._imageResize, Image.BICUBIC)
        return image

    # preprocess image
    def preprocess(self, image: Image) -> np.ndarray:
        image = np.array(image, dtype='float32')

        # convert RGB
        if self._colorFormat == "RBG":
            image = np.array(image)[:, :, [0, 2, 1]].astype('float32')
        elif self._colorFormat == "GRB":
            image = np.array(image)[:, :, [1, 0, 2]].astype('float32')
        elif self._colorFormat == "GBR":
            image = np.array(image)[:, :, [1, 2, 0]].astype('float32')
        elif self._colorFormat == "BRG":
            image = np.array(image)[:, :, [2, 0, 1]].astype('float32')
        elif self._colorFormat == "BGR":
            image = np.array(image)[:, :, [2, 1, 0]].astype('float32')

        # transpose HWC
        if self._channelOrder == "HCW":
            image = np.transpose(image, [0, 2, 1])
        elif self._channelOrder == "WHC":
            image = np.transpose(image, [1, 0, 2])
        elif self._channelOrder == "WCH":
            image = np.transpose(image, [1, 2, 0])
        elif self._channelOrder == "CHW":
            image = np.transpose(image, [2, 0, 1])
        elif self._channelOrder == "CWH":
            image = np.transpose(image, [2, 1, 0])

        # normalize
        for i in range(0, 3):
            image[i, :, :] = (image[i, :, :] / self._imageNorm - self._imageNormMean[i]) / self._imageNormStd[i]

        # expand_dims
        if self._expandDim is not None:
            # imageDateType = "float32"
            if self._imageDateType is not None:
                image = np.expand_dims(image.astype(self._imageDateType), self._expandDim)
            else:
                image = np.expand_dims(image, 0)

        # Pad to be divisible of 32
        if self._modelType == 'rcnn':
            padded_h = int(math.ceil(image.shape[1] / 32) * 32)
            padded_w = int(math.ceil(image.shape[2] / 32) * 32)

            padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
            padded_image[:, :image.shape[1], :image.shape[2]] = image
            image = padded_image

        return image

    def process_image(self, image: Image, confidence_threshold: float = 0.0) -> Tuple[list, float]:
        image_data = self.preprocess(image)
        image_size = [image.width, image.height]
        if self._imageResize is not None:
            image_size = self._imageResize

        inference_time_start = time.time()
        if self._modelType == "rcnn":
            results = self._session.run(None, {self._session.get_inputs()[0].name: image_data})
            detected_objects = self.post_process_rcnn(results[0], results[1], results[2], image_size,
                                                      confidence_threshold)
        elif self._modelType == "ssd":
            boxes, labels, scores = self._session.run(None, {self._session.get_inputs()[0].name: image_data})
            detected_objects = self.post_process_ssd(boxes, labels, scores, confidence_threshold)
        elif self._modelType == "ssd_mobile_net":
            detections, boxes, scores, labels = self._session.run(
                ["num_detections:0", "detection_boxes:0", "detection_scores:0", "detection_classes:0"],
                {self._session.get_inputs()[0].name: image_data})
            detected_objects = self.post_process_ssd_mobile_net(detections, boxes, labels, scores, confidence_threshold)
        elif self._modelType == "yolov3":
            image_shape = np.array([image_size[0], image_size[1]], dtype=np.float32).reshape(1, 2)
            boxes, scores, indices = self._session.run(None, {self._session.get_inputs()[0].name: image_data,
                                                              self._session.get_inputs()[1].name: image_shape})
            detected_objects = self.post_process_yolov3(boxes, scores, indices, image_size, confidence_threshold)
        else:
            return [], 0

        inference_time_end = time.time()
        inference_duration_s = inference_time_end - inference_time_start
        return detected_objects, inference_duration_s

    def post_process_rcnn(self, boxes, labels, scores, image_size: Tuple[int, int],
                          confidenceThreshold: float = 0.0) -> list:
        detected_objects = []
        image_width, image_height = image_size
        for box, label, score in zip(boxes, labels, scores):
            if score > confidenceThreshold:
                dobj = {
                    "confidence": score.tolist(),
                    "label": self._tags[label],
                    "x1": box[0] / image_width,
                    "x2": box[2] / image_height,
                    "y1": box[1] / image_width,
                    "y2": box[3] / image_height,
                }
                detected_objects.append(dobj)
        return detected_objects

    def post_process_ssd(self, boxes, labels, scores, confidenceThreshold: float = 0.0) -> list:
        detected_objects = []
        for i in range(0, scores[0].size):
            score = scores[0][i]
            if score > confidenceThreshold:
                box = boxes[0][i]
                dobj = {
                    "confidence": score.tolist(),
                    "label": self._tags[labels[0][i]],
                    "x1": np.float64(max(0, box[0])),
                    "x2": np.float64(box[2]),
                    "y1": np.float64(max(0, box[1])),
                    "y2": np.float64(box[3]),
                }
                detected_objects.append(dobj)
            return detected_objects

    def post_process_ssd_mobile_net(self, detections, boxes, labels, scores, confidenceThreshold: float = 0.0) -> list:
        detected_objects = []
        batch_size = detections.shape[0]
        for batch in range(0, batch_size):
            for detection in range(0, int(detections[batch])):
                if scores[batch][detection] > confidenceThreshold:
                    box = boxes[batch][detection]
                    dobj = {
                        "confidence": scores[batch][detection].tolist(),
                        "label": self._tags[int(labels[batch][detection])],
                        "x1": np.float64(box[1]),
                        "x2": np.float64(box[3]),
                        "y1": np.float64(box[0]),
                        "y2": np.float64(box[2]),
                    }
                    detected_objects.append(dobj)
        return detected_objects

    def post_process_yolov3(self, boxes, scores, indices, image_size: Tuple[int, int],
                            confidenceThreshold: float = 0.0) -> list:
        detected_objects = []
        image_width, image_height = image_size
        if indices.ndim == 3:
            indices = indices[0]
        for index_ in indices:
            score = scores[tuple(index_)].tolist()
            if score > confidenceThreshold:
                y1, x1, y2, x2 = boxes[(index_[0], index_[2])].tolist()
                dobj = {
                    "confidence": score,
                    "label": self._tags[index_[1].tolist()],
                    "x1": x1 / image_width,
                    "x2": x2 / image_width,
                    "y1": y1 / image_height,
                    "y2": y2 / image_height,
                }
                detected_objects.append(dobj)
        return detected_objects


def draw_bounding_boxes(image: Image, detected_objects: list):
    objects_identified = len(detected_objects)
    draw = ImageDraw.Draw(image)
    for pos in range(objects_identified):
        x1 = detected_objects[pos]["x1"] * image.width
        y1 = detected_objects[pos]["y1"] * image.height
        x2 = detected_objects[pos]["x2"] * image.width
        y2 = detected_objects[pos]["y2"] * image.height
        object_class = detected_objects[pos]['label']
        draw.rectangle((x1, y1, x2, y2), outline='blue', width=1)
        draw.text((x1, y1), str(object_class), fill="white", font=ImageFont.load_default())
    return image


def load_image(request: Request):
    try:
        image_data = io.BytesIO(request.get_data())
        image = Image.open(image_data)
        return model.resize_image(image)
    except Exception:
        abort(Response(response='Could not decode image', status=400))


app = Flask(__name__)


@app.route("/score", methods=['POST'])
def score():
    confidence = request.args.get('confidence', default=0.0, type=float)
    image = load_image(request)

    detected_objects, inference_duration_s = model.process_image(image, confidence)
    print('Inference took %.2f seconds', inference_duration_s)

    if len(detected_objects) > 0:
        respBody = {
            "results": detected_objects,
            "cost_ms": inference_duration_s * 1000
        }

        respBody = json.dumps(respBody)
        return Response(respBody, status=200, mimetype='application/json')
    else:
        return Response(status=204)


@app.route('/annotate', methods=['POST'])
def annotate():
    confidence = request.args.get('confidence', default=0.0, type=float)
    image = load_image(request)

    detected_objects, inference_duration_s = model.process_image(image, confidence)
    print('Inference took %.2f seconds', inference_duration_s)

    image = draw_bounding_boxes(image, detected_objects)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes = image_bytes.getvalue()
    return Response(response=image_bytes, status=200, mimetype="image/jpeg")


def check_system_cert() -> bool:
    ca = "var/lib/baetyl/system/certs/ca.pem"
    key = "var/lib/baetyl/system/certs/key.pem"
    crt = "var/lib/baetyl/system/certs/crt.pem"

    # check system cert exist
    if not os.path.exists(ca) or not os.path.exists(key) or not os.path.exists(crt):
        print("system certificate is not found")
        return False

    # check system cert valid
    info = x509.load_pem_x509_certificate(str.encode(open(crt).read()), default_backend())
    if info is None \
            or len(info.subject.get_attributes_for_oid(NameOID.ORGANIZATIONAL_UNIT_NAME)) != 1 \
            or info.subject.get_attributes_for_oid(NameOID.ORGANIZATIONAL_UNIT_NAME)[0].value != "BAETYL":
        print("system certificate is invalid")
        return False
    return True


if __name__ == '__main__':
    if not check_system_cert():
        sys.exit(1)
    model = Model()
    app.run(host='0.0.0.0', port=8888)
