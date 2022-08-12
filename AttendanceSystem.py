import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import imutils
from imutils.video import VideoStream

# Creates a list of images present in ImagesAttendance
path = 'ImagesBasic'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

# Removes Extension of Images
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


# Function to create Encodings for all Images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Creating Attendance File
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtStringDate = now.strftime('%D')
            dtStringTime = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtStringDate},{dtStringTime}')


encodeListKnown = findEncodings(images)
print(len(encodeListKnown))
print('All Facial Encodings completed :)')


cap = VideoStream(src=0).start()  
# SPECIFY HERE YOUR DRONE CREDENTIALS TO GET LIVE FEED   
# cap = VideoStream(src="http://192.168.0.102:8080/video").start()    --> I used this command to get live feed from a mobile application
# AT PRESENT THIS PROJECT WILL BE RECORDING THE DATA USING SYSTEM IN-BUILT RECORDING DEVICE

while True:
    img = cap.read()
    img = imutils.resize(img, width=900)
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)                               
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)


    cv2.imshow('[TERMINATORS] Drone cam Stream', img)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.stop()

