import cv2 as cv
import numpy as np

# path to video of myself (not contained in the git repo)
video_path = '/home/felix/Desktop/project_wskl/data/OpenCV_Test/hand.mp4'

#cascades source: https://github.com/Balaje/OpenCV/haarcascades
hand_cascade = cv.CascadeClassifier('cascades/hand_3.xml')

cap = cv.VideoCapture(video_path)

while 1:
    ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hands = hand_cascade.detectMultiScale(gray, 1.3, 5)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # draw rectangle around the hands
    for (x,y,w,h) in hands:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    cv.imshow('img',img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
