import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path = 'imagesoffkbois'
images = []
classnames = []
list = os.listdir(path)
print(list)

for cl in list:
    curpic = cv2.imread(f'{path}/{cl}')
    images.append(curpic)
    classnames.append(os.path.splitext(cl)[0])
print(classnames)

def findecoding(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoode = face_recognition.face_encodings(img)[0]
        encodelist.append(encoode)
    return encodelist

def markattendance(name):
    with open('sheet.csv','w+') as f:
        mydata = f.readlines()
        namelist = []
        for line in mydata:
            if namelist != name:
                entry = line.split(',')
                namelist.append(entry[0])

        if name not in namelist:
            now = datetime.now()
            dtstring  = now.strftime('%H:%M')
            h = 1

            f.writelines(f'\n{name},{dtstring},{h}')


encodelistknown = findecoding(images)
print(len(encodelistknown))

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facescurframe = face_recognition.face_locations(imgS)
    encodescurframe = face_recognition.face_encodings(imgS,facescurframe)

    for encodeface, faceloc in zip(encodescurframe,facescurframe):
        matches = face_recognition.compare_faces(encodelistknown, encodeface)
        facedist = face_recognition.face_distance(encodelistknown, encodeface)
            # print(facedist)
        matchindex = np.argmin(facedist)

        if matches[matchindex]:
            name = classnames[matchindex].upper()
            markattendance(name)
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)