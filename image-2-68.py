# _*_ coding:utf-8 _*_

import numpy as np
import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

img = cv2.imread("demo.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
rects = detector(img_gray, 0)

for i in range(len(rects)):
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()])
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        print(idx,pos)
        cv2.circle(img, pos, 10, color=(0, 255, 0), thickness=-1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(idx+1), pos, font, 0.8, (255, 0, 0), 2)

cv2.namedWindow("Dlib Demo | Social Robotics Course | ENSTA", 2)
cv2.imshow("Dlib Demo | Social Robotics Course | ENSTA", img)
cv2.waitKey(0)