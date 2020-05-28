import cv2
import os
import numpy as np


def resize(name):
    max_height = 125
    max_width = 250
    img = cv2.imread(name)
    h, w = img.shape[0:2]

    resize_ratio_h = 1
    resize_ratio_w = 1

    if h > max_height:
        resize_ratio_h = max_height / h
      
    if w > max_width:
        resize_ratio_w = max_width / w
      
    resize_ratio = min(resize_ratio_h, resize_ratio_w)

    if resize_ratio != 1:
        new_h, new_w = int(np.round(h * resize_ratio)), int(np.round(w * resize_ratio))
        img = cv2.resize(img, (new_w, new_h))
        cv2.imwrite(name, img)


def recognize(img_path):
    #resize if needed
    resize(img_path)

    #create annotation
    annotation = open('annotations.txt', 'w+')
    temp_plate = 'X000XX000'
    annotation.write(img_path + ' '+ temp_plate)
    annotation.close()

    #create tfrecord
    comm = 'aocr dataset annotations.txt test.tfrecords'
    os.system(comm)

    #inference 
    comm1 = 'aocr test test.tfrecords --model-dir models/Attention-OCR_car_plate_recognition --max-height 125 --max-width 250 --max-prediction 9 --output-dir ./results' 
    os.system(comm1)

    logs = open('aocr.log').read()
    
    