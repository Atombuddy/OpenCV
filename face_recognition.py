import numpy as np
import cv2
import os

face_cascade=cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")

p=[]
for i in os.listdir("images"):
    p.append(i)


face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("trained.yml")


cap=cv2.VideoCapture(0)




while True:
    ret,frame=cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=4)
    for (x,y,w,h) in faces_rect:
        faces_roi=gray[y:y+h,x:x+w]
        label,confidence=face_recognizer.predict(faces_roi)
        print(p[label], confidence)
        cv2.putText(frame, str(p[label]), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)


    cv2.imshow("face",frame)
    if cv2.waitKey(20) & 0xFF==ord("q"):
        break