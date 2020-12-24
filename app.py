import cv2
import time

from flask import Flask, request, jsonify, Response
from PIL import Image
import numpy as np
import json
from urllib.request import Request, urlopen
#import base64
#import io
import os
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from pytorch_loader import load_pytorch_model, pytorch_inference

from werkzeug.exceptions import BadRequestKeyError

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model = load_pytorch_model('models/model360.pth');

# anchor configuration
feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}

def inference(image,
              conf_thresh=0.5,
              iou_thresh=0.4,
              target_shape=(160, 160),
              draw_result=True,
              show_result=True
              ):
    '''
    Main function of detection inference
    :param image: 3D numpy array of image
    :param conf_thresh: the min threshold of classification probabity.
    :param iou_thresh: the IOU threshold of NMS
    :param target_shape: the model input size.
    :param draw_result: whether to daw bounding box to the image.
    :param show_result: whether to display the image.
    :return:
    '''
    # image = np.copy(image)
    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0  # 归一化到0~1
    image_exp = np.expand_dims(image_np, axis=0)

    image_transposed = image_exp.transpose((0, 3, 1, 2))

    y_bboxes_output, y_cls_output = pytorch_inference(model, image_transposed)
    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )

    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        if draw_result:
            if class_id == 0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
        output_info.append([class_id, conf, xmin, ymin, xmax, ymax])
    
    if show_result:
        Image.fromarray(image).show()
    return output_info


def run_on_video(video_path, conf_thresh, show_result=False):
    global run_video
    run_video = True
    cap = cv2.VideoCapture(video_path)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened():
        raise ValueError("Video open failed.")
        return
    status = True
    idx = 0
    while run_video and status:
        start_stamp = time.time()
        status, img_raw = cap.read()
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        read_frame_stamp = time.time()
        if (status):
            inference(img_raw,
                      conf_thresh,
                      iou_thresh=0.5,
                      target_shape=(360, 360),
                      draw_result=True,
                      show_result=False)
#            cv2.imshow('image', img_raw[:, :, ::-1])
#            cv2.waitKey(1)
            inference_stamp = time.time()
            write_frame_stamp = time.time()
            idx += 1
            print("%d of %d" % (idx, total_frames))
            print("read_frame:%f, infer time:%f, write time:%f" % (read_frame_stamp - start_stamp,
                                                                   inference_stamp - read_frame_stamp,
                                                                   write_frame_stamp - inference_stamp))
            if show_result:
                ret, jpeg = cv2.imencode('.jpg', img_raw[:, :, ::-1])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes()  + b'\r\n\r\n')
    cap.release()

app = Flask(__name__)

@app.route('/api/', methods=["GET"])
def main_interface():
    print(request.args)
    path = request.args['path']
    mode = request.args.get('mode', default = '1')
    show = request.args.get('show', default = False) == 'True'
    res = {'path': path, 'mode': mode}
    if mode == '1':
        if path.startswith('http'):
            resp = Request(path, headers={'User-Agent': 'Mozilla/5.0'})
            img = np.asarray(bytearray(urlopen(resp).read()), dtype="uint8")
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = inference(img, show_result=show, target_shape=(360,360))
        return json.dumps(res, cls=NpEncoder) 
    else:
        if path == '0':
            path = 0
        if show:
            return Response(run_on_video(path, conf_thresh=0.5, show_result=show),
                     mimetype='multipart/x-mixed-replace; boundary=frame')
        else:
            run_on_video(path, conf_thresh=0.5)
            return jsonify(res)

@app.route('/stop/', methods=["GET"])
def stop_video_stream():
    global run_video
    run_video = False
    return ('', 204)


@app.after_request
def add_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response

@app.errorhandler(BadRequestKeyError)
def handle_bad_request_key_exception(error):
    '''Return a custom missing key error message and 400 status code'''
    return jsonify({'missing key(s)': error.args}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')


