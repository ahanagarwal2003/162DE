# import useful libraries
import cv2
from yolo_utils import *
from picamera2 import Picamera2

# check if the GPU is available
print("OpenCV version : ", cv2. __version__)
print("Availible cuda number: ", cv2.cuda.getCudaEnabledDeviceCount(), "\n")

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

# confidence and non-max suppression threshold for this YoloV3 version
confidenceThreshold = 0.5
nmsThreshold = 0.1

# defining the input frame resolution for the neural network to process
network = neural_net
height, width = 128,128 # fell free to change this value (it should be a multiple of 32)

# Initialize the Raspberry Pi Camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={'size': (640, 480)}))
picam2.start()

while True:
    # Capture frame from the camera
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Object detection
    outputs = convert_to_blob(frame, network, height, width)    
    bounding_boxes, class_objects, confidence_probs = object_detection(outputs, frame, confidenceThreshold)   
    indices = nms_bbox(bounding_boxes, confidence_probs, confidenceThreshold, nmsThreshold)
    box_drawing(frame, indices, bounding_boxes, class_objects, confidence_probs, classNames, color=(0,255,255), thickness=2)
  
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.close()
