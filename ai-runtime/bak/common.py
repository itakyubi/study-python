import io
import json
import time
from typing import Tuple

import numpy as np
import onnxruntime
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, Response, Request, abort, request

# load_image -> preprocess -> run -> postprocess -> draw_bounding_boxes

image_size = (416, 416)
image_RGB = None
image_HWC = [2, 0, 1]
image_norm = 255
image_norm_mean = [0, 0, 0]
image_norm_std = [1, 1, 1]
handler_mode = "faster-rcn"  # faster-rcnn、mask-rcnn、ssd、ssd-mobileNet、tiny-yolov3、yolov3


class Model:
    def __init__(self):
        model_path = 'yolov3-10.onnx'
        self._session = onnxruntime.InferenceSession(model_path)

        tags_file = 'tags.txt'
        self.tags = [line.rstrip('\n') for line in open(tags_file)]

    # preprocess image
    def preprocess(self, image: Image) -> np.ndarray:
        # resize
        if image_size is not None:
            image = image.resize(image_size, Image.BICUBIC)

        image = np.array(image, dtype='float32')

        # convert to BGR
        if image_RGB is not None:
            image = np.array(image)[:, :, image_RGB].astype('float32')

        # transpose HWC
        if image_HWC is not None:
            image = np.transpose(image, image_HWC)

        # normalize
        for i in range(0, 3):
            image[i, :, :] = (image[i, :, :] / image_norm - image_norm_mean[i]) / image_norm_std[i]

        # expand_dims
        image = np.expand_dims(image, 0)

        return image

    def process_image(self, image: Image, confidence_threshold: float = 0.0) -> Tuple[list, float]:
        image_data = self.preprocess(image)

        inference_time_start = time.time()
        if handler_mode == "faster_rcnn":
            boxes, labels, scores = self._session.run(None, {"image": image_data})
            detected_objects = self.post_process_faster_rcnn(boxes, labels, scores, image, confidence_threshold)
        elif handler_mode == "mask_rcnn":
            boxes, labels, scores, masks = self._session.run(None, {"image": image_data})
            detected_objects = self.post_process_mask_rcnn(boxes, labels, scores, image, confidence_threshold)
        elif handler_mode == "ssd":
            boxes, labels, scores = self._session.run(None, {"image": image_data})
            detected_objects = self.post_process_ssd(boxes, labels, scores, image, confidence_threshold)
        elif handler_mode == "ssd_mobile_net":
            detections, boxes, scores, labels = self._session.run(
                ["num_detections:0", "detection_boxes:0", "detection_scores:0", "detection_classes:0"],
                {"image_tensor:0": image_data})
            detected_objects = self.post_process_ssd_mobile_net(detections, boxes, labels, scores, image,
                                                                confidence_threshold)
        elif handler_mode == "tiny_yolov3":
            boxes, scores, indices = self._session.run(None, {"input_1": image_data, "image_shape": image_size})
            detected_objects = self.post_process_tiny_yolov3(boxes, scores, indices, image.size, confidence_threshold)
        elif handler_mode == "yolov3":
            boxes, scores, indices = self._session.run(None, {"input_1": image_data, "image_shape": image_size})
            detected_objects = self.post_process_yolov3(boxes, scores, indices, image.size, confidence_threshold)
        else:
            return ([], 0)

        inference_time_end = time.time()
        inference_duration_s = inference_time_end - inference_time_start
        return detected_objects, inference_duration_s

    def post_process_faster_rcnn(self, boxes, labels, scores, image: Image, confidenceThreshold: float = 0.0) -> list:
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

    def post_process_mask_rcnn(self, boxes, labels, scores, image: Image, confidenceThreshold: float = 0.0) -> list:
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

    def post_process_ssd(self, boxes, labels, scores, image: Image, confidenceThreshold: float = 0.0) -> list:
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

    def post_process_ssd_mobile_net(self, detections, boxes, labels, scores, image: Image,
                                    confidenceThreshold: float = 0.0) -> list:
        detected_objects = []

        batch_size = detections.shape[0]
        for batch in range(0, batch_size):
            for detection in range(0, int(detections[batch])):
                if scores[batch][detection] > confidenceThreshold:
                    box = boxes[batch][detection]
                    dobj = {
                        "type": "entity",
                        "entity": {
                            "tag": {
                                "value": self.tags[int(labels[batch][detection])],
                                "confidence": scores[batch][detection].tolist()
                            },
                            "box": {
                                "l": box[1] * image.width,
                                "t": box[0] * image.height,
                                "w": (box[3] - box[1]) * image.width,
                                "h": (box[2] - box[0]) * image.height
                            }
                        }
                    }
                    detected_objects.append(dobj)
        return detected_objects

    def post_process_tiny_yolov3(self, boxes, scores, indices, image_size: Tuple[int, int],
                                 confidenceThreshold: float = 0.0) -> list:
        detected_objects = []
        image_width, image_height = image_size

        if indices.ndim == 3:
            # Tiny YOLOv3 uses a 3D numpy array, while YOLOv3 uses a 2D numpy array
            indices = indices[0]

        for index_ in indices:
            # See https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov3#output-of-model for more details
            object_tag = self.tags[index_[1].tolist()]
            confidence = scores[tuple(index_)].tolist()
            y1, x1, y2, x2 = boxes[(index_[0], index_[2])].tolist()
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height
            left = x1 / image_width
            top = y1 / image_height

            dobj = {
                "type": "entity",
                "entity": {
                    "tag": {
                        "value": object_tag,
                        "confidence": confidence
                    },
                    "box": {
                        "l": left,
                        "t": top,
                        "w": width,
                        "h": height
                    }
                }
            }

        return detected_objects

    def post_process_yolov3(self, boxes, scores, indices, image_size: Tuple[int, int],
                            confidenceThreshold: float = 0.0) -> list:
        detected_objects = []
        image_width, image_height = image_size

        if indices.ndim == 3:
            # Tiny YOLOv3 uses a 3D numpy array, while YOLOv3 uses a 2D numpy array
            indices = indices[0]

        for index_ in indices:
            # See https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov3#output-of-model for more details
            object_tag = self.tags[index_[1].tolist()]
            confidence = scores[tuple(index_)].tolist()
            y1, x1, y2, x2 = boxes[(index_[0], index_[2])].tolist()
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height
            left = x1 / image_width
            top = y1 / image_height

            dobj = {
                "type": "entity",
                "entity": {
                    "tag": {
                        "value": object_tag,
                        "confidence": confidence
                    },
                    "box": {
                        "l": left,
                        "t": top,
                        "w": width,
                        "h": height
                    }
                }
            }

        return detected_objects


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
model = Model()


@app.route("/score", methods=['POST'])
def score():
    confidence = request.args.get('confidence', default=0.0, type=float)
    image = load_image(request)

    detected_objects, inference_duration_s = model.process_image(image, confidence)
    print('Inference took %.2f seconds', inference_duration_s)

    if len(detected_objects) > 0:
        respBody = {
            "inferences": detected_objects,
            "cost": inference_duration_s
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)
