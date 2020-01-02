import numpy as np
import cv2

import pyautogui 
#pyautogui.moveRel(0, 50, duration = 1)

face_cascade = cv2.CascadeClassifier(r'D:\\python3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'D:\\python3\\Lib\\site-packages\\cv2\\data\\haarcascade_righteye_2splits.xml')

#number signifies camera
cap = cv2.VideoCapture(0)
originx,originy=pyautogui.position().x,pyautogui.position().y
firstx,firsty=0,0
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes = eye_cascade.detectMultiScale(gray)
    for (ex,ey,ew,eh) in eyes:
        
        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        '''if firstx!=-1:
            pyautogui.moveRel(firstx*0.01, firsty*0.01, duration = 0.05)
            firstx,firsty=firstx+ex*0.01,firsty+ey*0.01
        else:
            firstx,firsty=ex*0.01,ey*0.01
            '''
        roi_gray2 = gray[ey:ey+eh, ex:ex+ew]
        roi_color2 = img[ey:ey+eh, ex:ex+ew]
        circles = cv2.HoughCircles(roi_gray2,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
        try:
            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(roi_color2,(i[0],i[1]),i[2],(255,255,255),2)
                pyautogui.moveTo(originx+i[0], originy+i[1], 2)
                print("drawing circle at %f,%f",i[0],i[1])
                # draw the center of the circle
                cv2.circle(roi_color2,(i[0],i[1]),2,(255,255,255),3)
        except Exception as e:
            pass
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
