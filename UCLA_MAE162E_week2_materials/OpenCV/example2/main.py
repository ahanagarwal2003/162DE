import cv2
import numpy as np

# read the image
img = cv2.imread("example2.jpg")

# resize the image
img = cv2.resize(img, (500, 500))

# convert the image to grayscale and apply Canny edge detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)

# pick the largest contour
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# draw the largest contour on the image
result = img.copy()
result = cv2.drawContours(result, contours, -1, (243, 239, 131), 2)

# stacl the original and edge-detected images
stacked_image = np.vstack((img, result))

# show the original image with the largest contour drawn on it
cv2.imshow("Images", stacked_image)
cv2.waitKey(0)

