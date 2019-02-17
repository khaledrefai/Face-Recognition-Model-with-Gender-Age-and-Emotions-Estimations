from bottle import Bottle, run
from statistics import mode
import cv2
from keras.models import load_model
import numpy as np
from face_reco_image import FaceImage
app = Bottle()
USE_SMALL_FRAME =  False
VISUALIZE_DATASET = False
process_this_frame = True
face = FaceImage()
 
@route('/postimage', method='POST')
def postimage():
    upload     = request.files.get('upload')
    name, ext = os.path.splitext(upload.filename)
    if ext not in ('.png','.jpg','.jpeg'):
        return 'File extension not allowed.'
    if process_this_frame:
       gray_image = cv2.cvtColor(upload, cv2.COLOR_BGR2GRAY)
       rgb_image = cv2.cvtColor(upload, cv2.COLOR_BGR2RGB)
    if USE_SMALL_FRAME:
      rgb_image = cv2.resize(rgb_image, (0, 0), fx=0.25, fy=0.25)

    result_img = face.detect_face(rgb_image)
return  send_file(result_img, mimetype='image/jpg')


if __name__ == '__main__':
  run(app, host='localhost', port=8080)