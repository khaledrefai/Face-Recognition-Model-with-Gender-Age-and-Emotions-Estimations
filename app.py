
from face_reco_image import FaceImage
import time
import os
import shutil
import tempfile
import logging
from flask import jsonify
from flask import json
from flask import request
from flask import Flask
import dlib
from os import path, getcwd
import tensorflow as tf 

print ("dlib version: {}".format(dlib.__version__))

USE_SMALL_FRAME = False
VISUALIZE_DATASET = False
process_this_frame = True
face = FaceImage()
graph = tf.get_default_graph()
app = Flask(__name__)

temporary_directory = tempfile.mkdtemp()
_allow_origin = '*'
_allow_methods = 'PUT, GET, POST, DELETE, OPTIONS'
_allow_headers = 'Authorization, Origin, Accept, Content-Type, X-Requested-With'


@app.errorhandler(400)
def bad_request(e):
    return jsonify({"status": "not ok", "message": "this server could not understand your request"})


@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "not found", "message": "route not found"})


@app.errorhandler(500)
def not_found(e):
    return jsonify({"status": "internal error", "message": "internal error occurred in server"})


@app.route('/detect', methods=['POST'])
def postimage():
    global graph
    with graph.as_default():
        file = request.files.get('upload')
        filename, ext = os.path.splitext(file.filename)
        if ext not in ('.png', '.jpg', '.jpeg'):
            return 'File extension not allowed.'
        tmp = tempfile.TemporaryDirectory()
        temp_storage = path.join(tmp.name, file.filename)

        file.save(temp_storage)

        result_info = face.detect_face_info(temp_storage)
    return jsonify(result_info)


if __name__ == "__main__":
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    print ("Starting server on http://localhost:5000")
    print ("Serving ...",  app.run(host='0.0.0.0'))
    print ("Finished !")
    print ("Removing temporary directory ...",shutil.rmtree(temporary_directory))
    print ("Done !")