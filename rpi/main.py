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


def image_processing(message_center):
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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
        for i in range(len(bounding_boxes)):
            message_center.add_yolo_detection(
                class_objects[i], bounding_boxes[i], confidence_probs[i]
            )  # Commented out as add_stop_sign is not defined
    else:
        message_center.add_no_object_detected()


def gps_processing():
    data, addr = sock.recvfrom(1024)
    line = data.decode().strip()

    try:
        distances_m = list(map(float, line.split(",")))
    except ValueError:
        pass

    position, error = trilaterate_2D(distances_m)
    # if position is not None:
    #     print(f"[POS] x = {position[0]:.2f} ft, y = {position[1]:.2f} ft, RMSE = {error:.2f} ft")


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
                gps_processing()

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
