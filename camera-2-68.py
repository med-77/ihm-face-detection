# _*_ coding:utf-8 _*_
import sys
import dlib  
import numpy as np  
import cv2  


class face_dlib():
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 480)
        self.cnt = 0

    def face(self):
        while (self.cap.isOpened()):
            flag, im_rd = self.cap.read()
            k = cv2.waitKey(1)
            img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)
            faces = self.detector(img_gray, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            if (len(faces) != 0):
                for i in range(len(faces)):
                    for k, d in enumerate(faces):
                        #draw red rectangle on the face
                        #cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255))
                        #use the detector to get the 68 points
                        shape = self.predictor(im_rd, d)
                        # draw a circle in landmark
                        for i in range(68):
                            cv2.circle(im_rd, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)
                            
            # enter "s" key save the image
            if (k == ord('s')):
                self.cnt += 1
                cv2.imwrite("screenshoot" + str(self.cnt) + ".jpg", im_rd)
            # enter "q" to quit
            if (k == ord('q')):
                break
            cv2.imshow("camera", im_rd)
        # free the camera
        self.cap.release()
        # delete the window
        cv2.destroyAllWindows()

if __name__ == "__main__":
    face_d = face_dlib()
    face_d.face()


