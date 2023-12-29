import cv2
import numpy as np
import time
from apriltag import Detector

def detect_apriltags(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = Detector()
    detections, dimg = detector.detect(gray, return_image=True)

    if len(detections) > 0:
        for detection in detections:
            for pt in detection.corners:
                pt = tuple(map(int, pt))
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)
            cv2.putText(frame, str(detection.tag_id), (int(detection.center[0]), int(detection.center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    # if len(detections) > 0:
    #         for detection in detections:
    #             for pt in detection.corners:
    #                 pt = tuple(map(int, pt))
    #                 cv2.circle(frame, pt, 5, (0, 0, 255), -1)
    #             cv2.putText(frame, str(detection.tag_id), (int(detection.center[0]), int(detection.center[1])),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    #             # Calculating distance using camera calibration
    #             tag_size_mm = 150  # gotta replace with the actual size of A-tag
    #             focal_length = 1000  # gotta replace with the focal length 

    #             # investigate if there's a native distance detection method in the apriltag library!!!

    #             # alright so we need to figure out a way to factor in the tilt of the april tag in the distacne calculation, and somehow get x y & z distance from the camera to the center of the april tag 
    #             # right now what we got is some dummy values for the actual tag size and camera focal length, and we basically use how big it is in the image to get an estimate of our distance from it bc it should be proportional
            

    #             # x y & z coordnate wrt the tag 

    #             apparent_tag_size_pixels = np.linalg.norm(detection.corners[0] - detection.corners[1])
    #             distance = (tag_size_mm * focal_length) / apparent_tag_size_pixels

    #             cv2.putText(frame, f"Distance: {distance:.2f} mm", (20, 20),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    #             print(distance)
    return frame


capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Error: Could not open camera.")
    exit()

cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)

while True:
    is_successful, frame = capture.read()

    if not is_successful:
        print("Error: Could not read frame.")
        break

    frame = detect_apriltags(frame)
    cv2.imshow('Altered', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.01)

capture.release()
cv2.destroyAllWindows()