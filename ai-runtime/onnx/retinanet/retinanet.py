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
from torchvision import transforms
import torch


def flatten(inputs):
    return [[flatten(i) for i in inputs] if isinstance(inputs, (list, tuple)) else inputs]


def update_flatten_list(inputs, res_list):
    for i in inputs:
        res_list.append(i) if not isinstance(i, (list, tuple)) else update_flatten_list(i, res_list)
    return res_list


def to_numpy(x):
    if type(x) is not np.ndarray:
        x = x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()
    return x


def generate_anchors(stride, ratio_vals, scales_vals, angles_vals=None):
    'Generate anchors coordinates from scales/ratios'

    scales = torch.FloatTensor(scales_vals).repeat(len(ratio_vals), 1)
    scales = scales.transpose(0, 1).contiguous().view(-1, 1)
    ratios = torch.FloatTensor(ratio_vals * len(scales_vals))

    wh = torch.FloatTensor([stride]).repeat(len(ratios), 2)
    ws = torch.sqrt(wh[:, 0] * wh[:, 1] / ratios)
    dwh = torch.stack([ws, ws * ratios], dim=1)
    xy1 = 0.5 * (wh - dwh * scales)
    xy2 = 0.5 * (wh + dwh * scales)
    return torch.cat([xy1, xy2], dim=1)


def box2delta(boxes, anchors):
    'Convert boxes to deltas from anchors'

    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh
    boxes_wh = boxes[:, 2:] - boxes[:, :2] + 1
    boxes_ctr = boxes[:, :2] + 0.5 * boxes_wh

    return torch.cat([
        (boxes_ctr - anchors_ctr) / anchors_wh,
        torch.log(boxes_wh / anchors_wh)
    ], 1)


def delta2box(deltas, anchors, size, stride):
    'Convert deltas from anchors to boxes'

    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    ctr = anchors[:, :2] + 0.5 * anchors_wh
    pred_ctr = deltas[:, :2] * anchors_wh + ctr
    pred_wh = torch.exp(deltas[:, 2:]) * anchors_wh

    m = torch.zeros([2], device=deltas.device, dtype=deltas.dtype)
    M = (torch.tensor([size], device=deltas.device, dtype=deltas.dtype) * stride - 1)
    clamp = lambda t: torch.max(m, torch.min(t, M))
    return torch.cat([
        clamp(pred_ctr - 0.5 * pred_wh),
        clamp(pred_ctr + 0.5 * pred_wh - 1)
    ], 1)


def decode(all_cls_head, all_box_head, stride=1, threshold=0.05, top_n=1000, anchors=None, rotated=False):
    'Box Decoding and Filtering'

    if rotated:
        anchors = anchors[0]
    num_boxes = 4 if not rotated else 6

    device = all_cls_head.device
    anchors = anchors.to(device).type(all_cls_head.type())
    num_anchors = anchors.size()[0] if anchors is not None else 1
    num_classes = all_cls_head.size()[1] // num_anchors
    height, width = all_cls_head.size()[-2:]

    batch_size = all_cls_head.size()[0]
    out_scores = torch.zeros((batch_size, top_n), device=device)
    out_boxes = torch.zeros((batch_size, top_n, num_boxes), device=device)
    out_classes = torch.zeros((batch_size, top_n), device=device)

    # Per item in batch
    for batch in range(batch_size):
        cls_head = all_cls_head[batch, :, :, :].contiguous().view(-1)
        box_head = all_box_head[batch, :, :, :].contiguous().view(-1, num_boxes)

        # Keep scores over threshold
        keep = (cls_head >= threshold).nonzero().view(-1)
        if keep.nelement() == 0:
            continue

        # Gather top elements
        scores = torch.index_select(cls_head, 0, keep)
        scores, indices = torch.topk(scores, min(top_n, keep.size()[0]), dim=0)
        indices = torch.index_select(keep, 0, indices).view(-1)
        classes = (indices / width / height) % num_classes
        classes = classes.type(all_cls_head.type())

        # Infer kept bboxes
        x = indices % width
        y = (indices / width) % height
        a = indices / num_classes / height / width
        box_head = box_head.view(num_anchors, num_boxes, height, width)
        boxes = box_head[a, :, y, x]

        if anchors is not None:
            grid = torch.stack([x, y, x, y], 1).type(all_cls_head.type()) * stride + anchors[a, :]
            boxes = delta2box(boxes, grid, [width, height], stride)

        out_scores[batch, :scores.size()[0]] = scores
        out_boxes[batch, :boxes.size()[0], :] = boxes
        out_classes[batch, :classes.size()[0]] = classes

    return out_scores, out_boxes, out_classes


def nms(all_scores, all_boxes, all_classes, nms=0.5, ndetections=100):
    'Non Maximum Suppression'

    device = all_scores.device
    batch_size = all_scores.size()[0]
    out_scores = torch.zeros((batch_size, ndetections), device=device)
    out_boxes = torch.zeros((batch_size, ndetections, 4), device=device)
    out_classes = torch.zeros((batch_size, ndetections), device=device)

    # Per item in batch
    for batch in range(batch_size):
        # Discard null scores
        keep = (all_scores[batch, :].view(-1) > 0).nonzero()
        scores = all_scores[batch, keep].view(-1)
        boxes = all_boxes[batch, keep, :].view(-1, 4)
        classes = all_classes[batch, keep].view(-1)

        if scores.nelement() == 0:
            continue

        # Sort boxes
        scores, indices = torch.sort(scores, descending=True)
        boxes, classes = boxes[indices], classes[indices]
        areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1).view(-1)
        keep = torch.ones(scores.nelement(), device=device, dtype=torch.uint8).view(-1)

        for i in range(ndetections):
            if i >= keep.nonzero().nelement() or i >= scores.nelement():
                i -= 1
                break

            # Find overlapping boxes with lower score
            xy1 = torch.max(boxes[:, :2], boxes[i, :2])
            xy2 = torch.min(boxes[:, 2:], boxes[i, 2:])
            inter = torch.prod((xy2 - xy1 + 1).clamp(0), 1)
            criterion = ((scores > scores[i]) |
                         (inter / (areas + areas[i] - inter) <= nms) |
                         (classes != classes[i]))
            criterion[i] = 1

            # Only keep relevant boxes
            scores = scores[criterion.nonzero()].view(-1)
            boxes = boxes[criterion.nonzero(), :].view(-1, 4)
            classes = classes[criterion.nonzero()].view(-1)
            areas = areas[criterion.nonzero()].view(-1)
            keep[(~criterion).nonzero()] = 0

        out_scores[batch, :i + 1] = scores[:i + 1]
        out_boxes[batch, :i + 1, :] = boxes[:i + 1, :]
        out_classes[batch, :i + 1] = classes[:i + 1]

    return out_scores, out_boxes, out_classes


def detection_postprocess(image, cls_heads, box_heads):
    # Inference post-processing
    anchors = {}
    decoded = []

    for cls_head, box_head in zip(cls_heads, box_heads):
        # Generate level's anchors
        stride = image.size[-1] // cls_head.shape[-1]
        if stride not in anchors:
            anchors[stride] = generate_anchors(stride, ratio_vals=[1.0, 2.0, 0.5],
                                               scales_vals=[4 * 2 ** (i / 3) for i in range(3)])
        # Decode and filter boxes
        decoded.append(decode(cls_head, box_head, stride,
                              threshold=0.05, top_n=1000, anchors=anchors[stride]))

    # Perform non-maximum suppression
    decoded = [torch.cat(tensors, 1) for tensors in zip(*decoded)]
    # NMS threshold = 0.5
    scores, boxes, labels = nms(*decoded, nms=0.5, ndetections=100)
    return boxes, labels, scores


class RetinanetModel:
    def __init__(self):
        model_path = 'retinanet-9.onnx'
        self._session = onnxruntime.InferenceSession(model_path)

        tags_file = 'tags.txt'
        self.tags = [line.rstrip('\n') for line in open(tags_file)]

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

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(image)
        input_tensor = input_tensor.unsqueeze(0)

        inputs_flatten = flatten(input_tensor.detach().cpu().numpy())
        inputs_flatten = update_flatten_list(inputs_flatten, [])

        ort_inputs = dict(
            (self._session.get_inputs()[i].name, to_numpy(input)) for i, input in enumerate(inputs_flatten))

        inference_time_start = time.time()
        ort_outs = self._session.run(None, ort_inputs)
        inference_time_end = time.time()
        inference_duration_s = inference_time_end - inference_time_start

        cls_heads = ort_outs[:5]
        box_heads = ort_outs[5:]

        boxes, labels, scores = detection_postprocess(image, cls_heads, box_heads)
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


def letterbox_image(image, size):
    '''Resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image


def load_image(request: Request):
    try:
        image_data = io.BytesIO(request.get_data())
        image = Image.open(image_data)
    except Exception:
        abort(Response(response='Could not decode image', status=400))

    # If size is not 416x416 then resize
    if image.size != (480, 640):
        model_image_size = (480, 640)
        new_image = image
        image = letterbox_image(new_image, tuple(reversed(model_image_size)))

    return image


app = Flask(__name__)

init_logging()

model = RetinanetModel()
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
