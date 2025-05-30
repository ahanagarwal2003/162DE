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

# Define color ranges in HSV for stoplight detection
color_ranges = {
    "red": ([0, 120, 70], [10, 255, 255]),
    "yellow": ([22, 80, 2], [35, 255, 255]),
    "green": ([36, 100, 100], [86, 255, 255]),
}

# Focal length and known width of a stoplight (in meters)
KNOWN_WIDTH = 0.3  # Approximate width of a stoplight
FOCAL_LENGTH = 700  # Adjust based on your camera calibration

def estimate_distance(known_width, focal_length, perceived_width):
    """Estimate the distance to an object using the pinhole camera model."""
    if perceived_width > 0:
        return (known_width * focal_length) / perceived_width
    return -1  # Return -1 if perceived width is invalid

def detect_stoplight(frame, message_center):
    """Detect stoplight status and send signals."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    stoplight_status = "Unknown"
    stoplight_distance = -1
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
                stoplight_status = color_name.capitalize()
                stoplight_distance = estimate_distance(KNOWN_WIDTH, FOCAL_LENGTH, w)

    # Provide signal based on stoplight status and distance
    if stoplight_status == "Green":
        signal = "Move"
    elif stoplight_status in ["Red", "Yellow"] and stoplight_distance > 0 and stoplight_distance <= 0.3048:  # 1 foot in meters
        signal = "Stop"
    else:
        signal = "Unknown"

    # Send signal to message center
    message_center.add_stoplight_signal(stoplight_status, stoplight_distance, signal)

    # Display stoplight status, distance, and signal on the frame
    cv2.putText(frame, f"Stoplight: {stoplight_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if stoplight_distance > 0:
        cv2.putText(frame, f"Distance: {stoplight_distance:.2f} m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Signal: {signal}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def image_processing(message_center):
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Object detection
    outputs = convert_to_blob(frame, network, 128, 128)
    bounding_boxes, class_objects, confidence_probs = object_detection(
        outputs, frame, 0.5
    )

    # Sort the detected objects by confidence and only send the best 2 detections
    bounding_boxes, class_objects, confidence_probs = sort_by_confidence(
        2, confidence_probs, bounding_boxes, class_objects
    )

    if len(bounding_boxes) > 0:
        for i in range(len(bounding_boxes)):
            message_center.add_yolo_detection(
                class_objects[i], bounding_boxes[i], confidence_probs[i]
            )
    else:
        message_center.add_no_object_detected()

    # Detect stoplight status
    detect_stoplight(frame, message_center)

    # Display the frame
    cv2.imshow("Image Processing", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        return False
    return True

# The rest of the code remains unchanged