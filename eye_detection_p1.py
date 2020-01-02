import cv2
import numpy as np
import dlib
import pyautogui
from math import *
import os
import signal


la,lb=0.0,0.0

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\\Users\\user1\\Desktop\\hitum\\facial-landmarks\\shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def clicker(x,y,z,w):
    global la,lb
    a=sqrt((x[0]-y[0])*(x[0]-y[0])+(x[1]-y[1])*(x[1]-y[1]))
    b=sqrt((z[0]-w[0])*(z[0]-w[0])+(z[1]-w[1])*(z[1]-w[1]))
    print(a,b,la,lb,abs(la-a),abs(lb-b))
    la=a
    lb=b
    if(lb!=0.0 and abs(lb-b)>=0.6):
        os.kill(os.getpid(),signal.SIGTERM)
    if(la!=0.0 and abs(la-a)>=0.07):
        pyautogui.click()
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        #x, y = face.left(), face.top()
        #x1, y1 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        left_point = (landmarks.part(36).x, landmarks.part(36).y)
        right_point = (landmarks.part(39).x, landmarks.part(39).y)
        center_top = midpoint(landmarks.part(37), landmarks.part(38))
        center_bottom = midpoint(landmarks.part(41), landmarks.part(40))

        hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
        ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
        clicker(right_point,left_point,center_top,center_bottom)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
