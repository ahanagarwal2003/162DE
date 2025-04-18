import cv2
import numpy as np

# read the image
img = cv2.imread("example4_1.jpg")

# resize the image
img = cv2.resize(img, (500, 500))

# load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# detect faces in the image
result = img.copy()
faces = face_cascade.detectMultiScale(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), 1.3, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(result, (x,y), (x+w, y+h), (255,0,0), 2)

# stack the image
stacked_image = np.vstack((img, result))

cv2.imshow("Face Detection", stacked_image)
cv2.waitKey(0)
