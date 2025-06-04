# System imports
import sys, os
import select
import ctypes, struct
import time
import tty
import termios
import argparse

# Package imports
import cv2

# Custom imports
from utils import *
from yolo_utils import *
from gps_utils import *
from MessageCenter import MessageCenter

def image_processing(message_center: MessageCenter):
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    gender = face_processing(frame)
    # send message only when a face is detected: 1 for male, 2 for female
    if gender > 0:
        message_center.add_face_detection(gender)

    # Object detection
    # objects: crosswalk, speedlimit, stop, trafficlight
    outputs = convert_to_blob(frame, network, 128, 128)
    bounding_boxes, class_objects, confidence_probs = object_detection(
        outputs, frame, 0.5
    )

    # sort the detected objets by confidence and only send the best 2 detections
    bounding_boxes, class_objects, confidence_probs = sort_by_confidence(
        2, confidence_probs, bounding_boxes, class_objects
    )

    if len(bounding_boxes) > 0:
        message_center.add_yolo_detection(
            class_objects[0], bounding_boxes[0], confidence_probs[0]
        )
        if class_objects[0] == 3:
            print(f"[INFO] Detected traffic light")
            status = 0 # red light = 0, green light = 1
            confidence = 0.0
            
            ## TODO: add traffic light detection logic here
            ## ~~~
            ## END TODO
            
            message_center.add_traffic_light(status, confidence)
    else:
        pass
        # just don't send anything for now
        # message_center.add_no_object_detected()

def face_processing(frame):
    h = frame.shape[0]
    w = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)

    face_detector = cv2.dnn.readNet('./cfg/opencv_face_detector_uint8.pb', './cfg/opencv_face_detector.pbtxt')
    gender_detector = cv2.dnn.readNet('./cfg/gender_net.caffemodel', './cfg/gender_deploy.prototxt')

    face_detector.setInput(blob)
    detections = face_detector.forward()

    # sort detections by confidence (from big to small)
    detections = sorted(detections[0, 0, :, :], key=lambda x: x[2], reverse=True)
    detection = detections[0]

    if detection[2] > 0.7:
        print("Face detected")
        faceBox = detection[3:7] * np.array([w, h, w, h])
        faceBox = faceBox.astype("int")

        face_detector = frame[max(0, faceBox[1]-15):min(faceBox[3]+15, frame.shape[0]-1),
                            max(0, faceBox[0]-15):min(faceBox[2]+15,frame.shape[1]-1)]
        blob = cv2.dnn.blobFromImage(face_detector, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_detector.setInput(blob)
        genderPreds = gender_detector.forward()
        gender = genderPreds[0].argmax() + 1 # 1 for male, 2 for female
        return gender
    else:
        return 0 # no face detected
            
def gps_processing(message_center):
    data, addr = sock.recvfrom(1024)
    line = data.decode().strip()

    try:
        distances_m = list(map(float, line.split(",")))
    except ValueError:
        pass

    position, error = trilaterate_2D(distances_m)
    if position is not None:
        # print(f"[POS] x = {position[0]:.2f} ft, y = {position[1]:.2f} ft, RMSE = {error:.2f} ft")
        message_center.add_gps_position(position[0], position[1])

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="My program with options")
    parser.add_argument(
        "-gps",
        "--gps",
        action="store_true",
        default=False,
        help="Enable GPS in the program",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode, printing debug messages",
    )
    args = parser.parse_args()

    # initialize the camera and GPS
    camera_initialization()
    # initialize gps if enabled
    if args.gps:
        gps_initialization()

    # initialize the message center
    message_center = MessageCenter("/dev/ttyUSB0", 9600, args.debug)

    # Welcome message :)
    print_seal()

    # system start time
    program_start_time = time.time()
    time_stamp = program_start_time
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)  # or tty.setraw(fd)
        while True:
            image_processing(message_center)

            if args.gps:
                gps_processing(message_center)

            # process messages
            message_center.processing_tick()

            # handle user input
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                # print(f"You typed: {ch}")
                if ch == "q":
                    print("\nUser quit command detected. Exiting program...")
                    break
                else:
                    print(f"\nUnrecognized command: {ch}")

            # delay for a short period to avoid busy waiting
            time.sleep(0.1)

            # print time elapsed since start and time interval without newline
            interval = (time.time() - time_stamp) * 1000
            time_stamp = time.time()
            print(
                f"\rTime elapsed: {get_time_millis(program_start_time):.2f} ms; loop interval: {interval:.02f} ms",
                end="",
            )
            print("\b" * 40, end="")  # Clear the line

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting...")
        # do some cleanup if necessary

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return


if __name__ == "__main__":
    main()
