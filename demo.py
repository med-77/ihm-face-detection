import dlib
import cv2

detector = dlib.get_frontal_face_detector()
img = cv2.imread('demo.jpg')
img = cv2.resize(img, (560, 380))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector(gray, 1) 
#to draw faces on image
for result in faces:
    x = result.left()
    y = result.top()
    x1 = result.right()
    y1 = result.bottom()
    img=cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
cv2.namedWindow("Dlib Demo | Social RObotics Course | ENSTA", cv2.WINDOW_NORMAL)
cv2.imshow("Dlib Demo | Social RObotics Course | ENSTA",img)
cv2.waitKey()
