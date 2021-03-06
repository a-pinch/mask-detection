from threading import Thread
import cv2
import time

from PIL import Image
import numpy as np
import os
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from pytorch_loader import load_pytorch_model, pytorch_inference
from centroidtracker import CentroidTracker

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

ct = CentroidTracker()

def inference(image,
              conf_thresh=0.5,
              iou_thresh=0.4,
              target_shape=(160, 160),
              draw_result=True,
              show_result=True,
              track_results=True
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
            if not track_results:
                cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
        output_info.append([class_id, conf, xmin, ymin, xmax, ymax])

    if track_results:
        output_info = ct.update(output_info)
        for(objectId, info) in output_info.items():
            if info[0] == 0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            cv2.putText(image, "%d %s: %.2f" % (objectId, id2class[info[0]], info[1]), (info[2] + 2, info[3] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)

    if show_result:
        Image.fromarray(image).show()
    return output_info

class MaskStreamDetector:

    def __init__(self, fvs, track):
        self.fvs = fvs
        self.track = track

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.run_on_video_async, args=())
        t.daemon = True
        t.start()
        return self

    def run_on_video_async(self):
        for i in self.run_on_video():
            pass

    def run_on_video(self, boundary = None, conf_thresh=0.5):
        idx = 0
        print("running")
        while self.fvs.running() or self.fvs.more():
            if(self.fvs.more()):
                frame = self.fvs.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                inference(frame,
                          conf_thresh,
                          iou_thresh=0.5,
                          target_shape=(360, 360),
                          draw_result=True,
                          show_result=False,
                          track_results=self.track)
                idx += 1
                print("processed %d frames" % (idx))
                if boundary:
                    ret, jpeg = cv2.imencode('.jpg', frame[:, :, ::-1])
                    yield (b'--' + boundary.encode() + b'\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes()  + b'\r\n\r\n')
            else:
                time.sleep(0.1)
