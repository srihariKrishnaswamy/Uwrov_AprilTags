#!/usr/bin/env python

from argparse import ArgumentParser
import os
import cv2
import apriltag
import numpy as np
from scipy.spatial.transform import Rotation as tf

################################################################################
fx = 2.44171380e+03
fy = 2.39069151e+03
cx = 6.30504437e+02
cy = 5.70956578e+02
apriltag_size = 0.095

# camera_params=(3156.71852, 3129.52243, 359.097908, 239.736909)

def apriltag_video(input_streams=['../media/input/single_tag.mp4', '../media/input/multiple_tags.mp4'], # For default cam use -> [0]
                   output_stream=False,
                   display_stream=True,
                   detection_window_name='AprilTag',
                   cam_stream=True,
                  ):
    '''
    Detect AprilTags from video stream.

    Args:   input_streams [list(int/str)]: Camera index or movie name to run detection algorithm on
            output_stream [bool]: Boolean flag to save/not stream annotated with detections
            display_stream [bool]: Boolean flag to display/not stream annotated with detections
            detection_window_name [str]: Title of displayed (output) tag detection window
    '''
    parser = ArgumentParser(description='Detect AprilTags from video stream.')
    apriltag.add_arguments(parser)
    options = parser.parse_args()

    '''
    Set up a reasonable search path for the apriltag DLL.
    Either install the DLL in the appropriate system-wide
    location, or specify your own search paths as needed.
    '''

    detector = apriltag.Detector(options, searchpath=apriltag._get_dll_path())

    if cam_stream:
            video = cv2.VideoCapture(0)
            if not video.isOpened():
                print("Error: Could not open camera.")
                exit()
            output = None
            if output_stream:
                width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(video.get(cv2.CAP_PROP_FPS))
                codec = cv2.VideoWriter_fourcc(*'XVID')
                if type(stream) != int:
                    output_path = '../media/output/'+str(os.path.split(stream)[1])
                    output_path = output_path.replace(str(os.path.splitext(stream)[1]), '.avi')
                else:
                    output_path = '../media/output/'+'camera_'+str(stream)+'.avi'
                output = cv2.VideoWriter(output_path, codec, fps, (width, height))
            while True:
                success, frame = video.read()
                if not success:
                    break
                result, overlay, pose = apriltag.detect_tags(frame,
                                                    detector,
                                                    camera_params=(fx, fy, cx, cy),
                                                    tag_size=apriltag_size,
                                                    vizualization=3,
                                                    verbose=3,
                                                    annotation=True
                                                    )
                print("Pose: ", pose)
                if pose is not None:
                    pass
                    pitch, roll, yaw, x, y, z = extract_pry_xyz(pose)
                    print("Pitch: ", pitch, " roll: ", roll, " yaw: ", yaw)
                    print("X-disp: ", x, " Y-disp: ", y, " Z-disp: ", z)
                if output_stream:
                    output.write(overlay)
                if display_stream:
                    cv2.imshow(detection_window_name, overlay)
                    if cv2.waitKey(1) & 0xFF == ord(' '): # Press space bar to terminate
                        break
    else:
        for stream in input_streams:
            
            video = cv2.VideoCapture(stream)
            output = None
            if output_stream:
                width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(video.get(cv2.CAP_PROP_FPS))
                codec = cv2.VideoWriter_fourcc(*'XVID')
                if type(stream) != int:
                    output_path = '../media/output/'+str(os.path.split(stream)[1])
                    output_path = output_path.replace(str(os.path.splitext(stream)[1]), '.avi')
                else:
                    output_path = '../media/output/'+'camera_'+str(stream)+'.avi'
                output = cv2.VideoWriter(output_path, codec, fps, (width, height))
            while(video.isOpened()):
                success, frame = video.read()
                if not success:
                    break
                result, overlay = apriltag.detect_tags(frame,
                                                    detector,
                                                    camera_params=(3156.71852, 3129.52243, 359.097908, 239.736909),
                                                    tag_size=0.0762,
                                                    vizualization=3,
                                                    verbose=3,
                                                    annotation=True
                                                    )
                if output_stream:
                    output.write(overlay)
                if display_stream:
                    cv2.imshow(detection_window_name, overlay)
                    if cv2.waitKey(1) & 0xFF == ord(' '): # Press space bar to terminate
                        break

def extract_pry_xyz(pose_matrix):
    translation = pose_matrix[:3, 3]
    x, y, z = translation
    rotation_matrix = pose_matrix[:3, :3]
    rotation = tf.from_matrix(rotation_matrix)
    euler_angles = rotation.as_euler('xyz', degrees=True)
    pitch, roll, yaw = euler_angles
    return pitch, roll, yaw, x, y, z
################################################################################

if __name__ == '__main__':
    apriltag_video()
