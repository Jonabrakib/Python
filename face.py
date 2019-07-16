import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

videocapture = cv2.VideoCapture(0)
scale_factor = 1.3

while True:
    ret,pic = videocapture.read()
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scale_factor, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(pic, (x, y),(x+w,y+h),(255,0,0),2)

    
    print("Number of faces {}".format(len(faces)))
    cv2.imshow('face',pic)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
videocapture.release()
cv2.destroyAllWindows()