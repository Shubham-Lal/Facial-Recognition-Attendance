import cv2
import face_recognition

# Image Location of Elon Musk
imgElon = face_recognition.load_image_file('C:\\Users\\Shubham\\Desktop\\MyProjects\\Face-Attendance-System\\ImagesBasic\\Elon Musk.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
# Image Location of Bill Gates
imgGate = face_recognition.load_image_file('C:\\Users\\Shubham\\Desktop\\MyProjects\\Face-Attendance-System\\ImagesBasic\\Bill Gates.jpg')
imgGate = cv2.cvtColor(imgGate, cv2.COLOR_BGR2RGB)

########################################################################################################################

# Face Location of Elon Musk
faceLocElon = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLocElon[3], faceLocElon[0]), (faceLocElon[1], faceLocElon[2]), (255, 0, 255), 2)

# Face Location of Bill Gates
faceLocGate = face_recognition.face_locations(imgGate)[0]
encodeGate = face_recognition.face_encodings(imgGate)[0]
cv2.rectangle(imgGate, (faceLocGate[3], faceLocGate[0]), (faceLocGate[1], faceLocGate[2]), (255, 0, 255), 2)

########################################################################################################################

results = face_recognition.compare_faces([encodeElon], encodeGate)
faceDist = face_recognition.face_distance([encodeElon], encodeGate)
print(results, faceDist)
cv2.putText(imgElon, f'{results}{round(faceDist[0],2)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

########################################################################################################################

cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Bill Gates', imgGate)
cv2.waitKey(0)

