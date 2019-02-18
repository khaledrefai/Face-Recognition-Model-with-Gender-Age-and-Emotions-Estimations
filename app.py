from bottle import hook, route, response, run, post ,request
from statistics import mode
import cv2
from keras.models import load_model
import numpy as np
from face_reco_image import FaceImage
import os
from os import path, getcwd
import tempfile


USE_SMALL_FRAME =  False
VISUALIZE_DATASET = False
process_this_frame = True
face = FaceImage()

_allow_origin = '*'
_allow_methods = 'PUT, GET, POST, DELETE, OPTIONS'
_allow_headers = 'Authorization, Origin, Accept, Content-Type, X-Requested-With'

@hook('after_request')
def enable_cors():
    '''Add headers to enable CORS'''

    response.headers['Access-Control-Allow-Origin'] = _allow_origin
    response.headers['Access-Control-Allow-Methods'] = _allow_methods
    response.headers['Access-Control-Allow-Headers'] = _allow_headers
    
@post('/postimage')
def postimage():
    file     = request.files.get('upload')
    filename, ext = os.path.splitext(file.filename)
    if ext not in ('.png','.jpg','.jpeg'):
        return 'File extension not allowed.'
    tmp = tempfile.TemporaryDirectory()
    temp_storage = path.join(tmp.name , file.filename)

    file.save(temp_storage)

    result_info = face.detect_face_info(temp_storage)
    return dict(data=result_info)


if __name__ == '__main__':
  run(host='localhost', port=8080, debug=True)