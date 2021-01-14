import cv2
import time

from flask import Flask, request, jsonify, Response
import numpy as np
import json
from urllib.request import Request, urlopen
from video import FileVideoStream
from detector import MaskStreamDetector, inference

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

app = Flask(__name__)
 
@app.route('/api/', methods=["GET"])
def main_interface():
    print(request.args)
    path = request.args['path']
    mode = request.args.get('mode', default = '1')
    show = request.args.get('show', default = False) == 'True'
    track = request.args.get('track', default = 'True') == 'True'
    res = {'path': path, 'mode': mode}
    global fvs
    global msd
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
        
        try:
            if not fvs.running():
                fvs = FileVideoStream(path).start()
            else:
                return 'Another video stream is running'
        except NameError:
            fvs = FileVideoStream(path).start()    

        time.sleep(1.0)
        msd = MaskStreamDetector(fvs, track)

        if show:
            boundary = 'frame'
            return Response(msd.run_on_video(boundary),
                     mimetype='multipart/x-mixed-replace; boundary=' + boundary)
        else:
            msd.start()
            return 'Start processing video stream' 

@app.route('/stop/', methods=["GET"])
def stop_video_stream():
    try:
        global fvs
        if fvs.running():
            fvs.stop()
            return "Video stream is stopped."
        else:
            return "Video stream is already stopped. Nothing to do."
    except NameError:
        return "There is no video stream processed. Nothing to do."  


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


