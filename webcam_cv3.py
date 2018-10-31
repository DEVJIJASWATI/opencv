import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

cascPath = "haarcascade_frontalface_default.xml"

eyesCascPath = "haarcascade_eye.xml"

smilePath = "haarcascade_smile.xml"

faceCascade = cv2.CascadeClassifier(cascPath)

eye_cascade = cv2.CascadeClassifier(eyesCascPath)

smile_cascade = cv2.CascadeClassifier(smilePath)


# log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)

anterior = 0

while True:
    if not video_capture.isOpened():
        print('WebCam Error.')
        sleep(5)
        pass

    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(

        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # cv2.imshow('face', frame)

        roi_gray = gray[y:y+h, x:x+w]

        # roi_gray_smile = smile_gray[y:y+h, ]

        roi_color = frame[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray)

    # eyes = eye_cascade.detectMultiScale(roi_color)

    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # smile_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # smile = smile_cascade.detectMultiScale(
    #     roi_gray,
    #     scaleFactor=1.7,
    #     minNeighbors=22,
    #     minSize=(25,25),
    #     flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    #
    # )
    #
    # for (x,y,w,h) in smile:
    #     print"SMile", len(smile), "smiles"
    #     cv2.rectangle(roi_color,(x,y),(x+w,y+h),(255, 0, 0), 1)
    #     print"sourire"

    if anterior != len(faces):
        anterior = len(faces)

        log.info("face: "+str(len(faces))+" detecter "+str(dt.datetime.now()))

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow('Video', frame)


video_capture.release()
cv2.waitKey(0)

cv2.destroyAllWindows()
