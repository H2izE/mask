# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import cv2
import cvlib as cv

image = cv2.imread('sample.png')
faces, confidences = cv.detect_face(image)

for (x,y,x2,y2), conf in zip(faces, confidences):
    cv2.putText(image, str(conf), (x,y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0),1)
    cv2.rectangle(image, (x,y), (x2,y2), (0,255,0), 2)

cv2.imshow('image',image)

key = cv2.waitKey(0)
cv2.destroyAllWindows()



is_readed, frame = webcam.read()

# # 1. 웹캠 연결하기

webcam = cv2.VideoCapture(0)#기본카메라 0번 사용
if not webcam.isOpened():
    raise Exception("카메라 읎음")


#프레임 받아오기
ret, frame = webcam.read() #2개의 리턴값을 튜플로 반환함.
print(frame)
if not ret:
    raise Exception("캡쳐가 없음")

faces, confidences = cv.detect_face(frame) #이미지에서 얼굴 위치, 얼굴일 확률 받아오기
print(faces[0])#좌표 4지점 
print(confidences)

start_x, start_y, end_x, end_y = faces[0]
cv2.imwrite('1.jpg', frame[start_y:end_y, start_x:end_x, :])

# +
import time

def capture(path, m=1):
    count = 0
    
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        raise Exception("카메라 읎음")
    
    while count < m:
        time.sleep(0.3) 
        ret, frame = webcam.read() #2개의 리턴값을 튜플로 반환함.
        if not ret:
            raise Exception("캡쳐가 없음")
            
        faces, confidences = cv.detect_face(frame) 
        
        for face, conf in zip(faces, confidences):
            if conf < 0.8:
                continue
            start_x, start_y, end_x, end_y = faces[0]
            cv2.imwrite(path+str(count)+'.jpg', frame[start_y:end_y, start_x:end_x, :])
            count += 1
            print(count,'장')
    print(count, end='')
    webcam.release()

capture('/Users/jangsujeong/Downloads/mask_project/nonMask', 300)

# -

capture('/Users/jangsujeong/Downloads/mask_project/Mask', 300)

# # 이미지 전처리

import os

non_list = os.listdir('/Users/jangsujeong/Downloads/mask_project/nonMask')
print(non_list)

yes_list = os.listdir('/Users/jangsujeong/Downloads/mask_project/Mask')
print(yes_list)

image = cv2.imread('/Users/jangsujeong/Downloads/mask_project/nonMask/nonMask85.jpg')

for i in non_list:
    image =  cv2.imread('/Users/jangsujeong/Downloads/mask_project/nonMask/' + i)

# +
w = []
h = []

for i in non_list:
    image = cv2.imread('/Users/jangsujeong/Downloads/mask_project/nonMask/' + i)
    h.append(image.shape[0])
    w.append(image.shape[1])
# -

for i in yes_list:
    image = cv2.imread('/Users/jangsujeong/Downloads/mask_project/Mask/' + i)
    h.append(image.shape[0])
    w.append(image.shape[1])

# +
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.scatter(w,h,alpha=0.5)#산점도로 표시

# +
import cv2
import cvlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from IPython.display import Image
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import array_to_img, load_img, img_to_array
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

img_w, img_h = 140, 180
images = [] #실제 데이터 
labels = [] #정답 데이터(1,0으로 분류)

for i in non_list:
    image = load_img('/Users/jangsujeong/Downloads/mask_project/nonMask/' + i, target_size=(img_w, img_h))#임시값
    image = img_to_array(image)
    images.append(image)
    labels.append(0)#마스크 쓰지 않았으므로 0
    
for i in yes_list:
    image = load_img('/Users/jangsujeong/Downloads/mask_project/Mask/' + i, target_size=(img_w, img_h))
    image = img_to_array(image)
    images.append(image)
    labels.append(1)#마스크 썼기 때문에 1

images[0].shape
# -

from sklearn.model_selection import train_test_split
import numpy as np


x_train, x_test, y_train, y_test = train_test_split(np.array(images), np.array(labels), test_size=0.1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

y_test


