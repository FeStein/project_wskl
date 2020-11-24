import cv2 as cv
import numpy as np

# path to video of myself (not contained in the git repo)
video_path = '/home/felix/Desktop/project_wskl/data/OpenCV_Test/hand.mp4'

#cascades source: https://github.com/opencv/opencv/blob/master/data/haarcascades
eye_cascade = cv.CascadeClassifier('cascades/eye.xml')
front_face_cascade = cv.CascadeClassifier('cascades/front_face.xml')

cap = cv.VideoCapture(video_path)

while 1:
    ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    faces = front_face_cascade.detectMultiScale(gray, 1.3, 5)

    # draw rectangle around the eyes
    for (x, y, w, h) in eyes:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(img, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        #annotation
        cv.putText(img, 'eye', (x + w, y + h), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)

    # draw rectangle around face
    for (x, y, w, h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(img, center, (w//2, h//2), 0, 0, 360, (255, 0, 0), 4)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        #annotation
        cv.putText(img, 'face', (x + int(w*0.9), y + int(h*0.9)), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)

    print('Eyes: {} | Faces: {}'.format(len(eyes), len(faces)))

    cv.imshow('img', img)
    k = cv.waitKey(20) & 0xff
    if k == 32:
        break

cap.release()
cv.destroyAllWindows()
