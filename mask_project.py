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


