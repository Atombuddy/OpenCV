import os
import numpy as np
import cv2

face_cascade=cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")
features=[]
labels=[]
p=[]
DIR=r"C:\Users\Admin\project\nazeer\opencv\images"
for i in os.listdir("images"):
    p.append(i)
def create_train():
    for person in p:
        path=os.path.join(DIR,person)
        label=p.index(person)
        for img in os.listdir(path):
            img_path=os.path.join(path,img)
            img_array=cv2.imread(img_path)
            gray=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=4)

            for (x,y,w,h) in faces:
                faces_roi=gray[y:y+h,x:x+w]
                features.append(faces_roi)
                labels.append(label)
create_train()

features=np.array(features,dtype="object")
labels=np.array(labels)
face_recognizer=cv2.face.LBPHFaceRecognizer_create()

face_recognizer.train(features,labels)
face_recognizer.save("trained.yml")

