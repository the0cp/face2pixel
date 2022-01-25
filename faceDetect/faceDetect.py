import cv2
import sys
import os.path
import numpy as np
from glob import glob

def detect(filename, cascade_file="faceDetect/haarcascade_frontalcatface.xml"):
    print('Detecting faces...')
    print('Format Spport: *.jpg')
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    img0 = cv2.imread(filename)
    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(64, 64),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) > 0:
        faces = faces[np.argsort(faces[:, 1])]
        image = faces[0], img0
        
        if image is not None:
            save_filename = '%s.png' % (os.path.basename(filename).split('.')[0])
            x, y, w, h = image[0]
            img = image[1]
            img_w = img.shape[0]
            img_h = img.shape[1]
            min_wh = int(min(min(x * 2 + w, y * 2 + h + 15), min(img_w, img_h)))
            save_x = max(x - int((min_wh - w) / 2), 0)
            save_y = max(y - int((min_wh - h) / 2) - 15, 0)
            save_img = np.copy(img)[save_y: save_y + min_wh, save_x: save_x + min_wh]
            cv2.imwrite("faces/" + save_filename, save_img)
            print('Face detected!')

    else:
        print('No face detected!')

