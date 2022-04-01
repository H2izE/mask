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


