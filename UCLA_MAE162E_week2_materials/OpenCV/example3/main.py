import cv2
import numpy as np

# Load the video
#cap = cv2.VideoCapture("example3_green.mp4")
cap = cv2.VideoCapture("example3_red.mp4") # Use this for TODO 2

# Define the green color range in HSV
lower_green = np.array([0, 150, 50])
upper_green = np.array([11, 255, 255])

while True:
    ret, frame = cap.read()
    # resize the frame
    frame = cv2.resize(frame, (640, 480))
    if not ret:
        break  # End of video

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Bitwise-AND mask and original image
    result = frame.copy()

    # draw bounding box around the detected green color
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small contours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # stack the result with the original image
    stacked_image = np.vstack((frame, result))

    # Show original and masked result
    cv2.imshow("vedios", stacked_image)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
