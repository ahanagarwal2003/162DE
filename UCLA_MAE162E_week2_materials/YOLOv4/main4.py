# import useful libraries
import cv2
from yolo_utils import *
from picamera2 import Picamera2
import numpy as np

# check if the GPU is available
print("OpenCV version : ", cv2.__version__)
print("Available CUDA devices: ", cv2.cuda.getCudaEnabledDeviceCount(), "\n")

# load the obj/classes names
obj_file = './obj.names'
classNames = read_classes(obj_file)
print("Classes' names :", classNames, "\n")

# load the model config and weights
modelConfig_path = './cfg/yolov4.cfg'
modelWeights_path = './weights/yolov4.weights'

# read the model cfg and weights with the cv2 DNN module
neural_net = cv2.dnn.readNetFromDarknet(modelConfig_path, modelWeights_path)
neural_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
neural_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# confidence and non-max suppression threshold for this YoloV4 version
confidenceThreshold = 0.5
nmsThreshold = 0.1

# defining the input frame resolution for the neural network to process
network = neural_net
height, width = 128, 128  # feel free to change this value (it should be a multiple of 32)

# Initialize the Raspberry Pi Camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={'size': (640, 480)}))
picam2.start()

# Define color ranges in HSV
color_ranges = {
    "red": ([0, 120, 70], [10, 255, 255]),
    "green": ([36, 100, 100], [86, 255, 255]),
    "Yellow": ([22, 80, 2], [35, 255, 255]),
}

while True:
    # Capture frame from the camera
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Object detection
    outputs = convert_to_blob(frame, network, height, width)
    bounding_boxes, class_objects, confidence_probs = object_detection(outputs, frame, confidenceThreshold)
    indices = nms_bbox(bounding_boxes, confidence_probs, confidenceThreshold, nmsThreshold)
    box_drawing(frame, indices, bounding_boxes, class_objects, confidence_probs, classNames, color=(0, 255, 255), thickness=2)

    # Convert frame to HSV for color detection
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect colors
    for color_name, (lower, upper) in color_ranges.items():
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the frame with detections and color recognition
    cv2.imshow('Object and Color Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.close()