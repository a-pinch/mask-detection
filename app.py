from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import base64
import io
import os

#from backend.tf_inference import load_model, inference

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#sess, detection_graph = load_model()

app = Flask(__name__)

@app.route('/api/', methods=["GET"])
def main_interface():
    return jsonify('Hello world!')

@app.after_request
def add_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
