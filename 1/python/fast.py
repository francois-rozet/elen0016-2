#############
# Libraries #
#############

import os
import cv2
import time

from utils import highlight
from soccer import Process


#############
# Variables #
#############

# Path to a video
ORIGIN = '../resources/videos/soccer/soccer_01.mp4'

# Path to export the processed video
DESTINATION = '../products/videos/'


#############
# Functions #
#############

def img_line_detection(img, process):
    """Detect and highlignt lines in an image
    according to a processing class

    Inputs
    ------
    img :       an image
    process :   a processing class

    Output
    ------
    the image in which lines are highlighted
    """

    # Line detection
    temp = process.preprocessing(img)
    temp = process.edge_points_extraction(temp)
    lines = process.line_detection(temp)

    return highlight(img, lines)

def vid_line_detection(vid_path, process, vid_dir = ''):
    """Detect and highlignt lines in a video
    according to a processing class

    Inputs
    ------
    vid_path :  a video path
    process :   a processing class
    vid_dir :   optional, the writing directory

    Output
    ------
    the path of the video in which lines are highlighted
    """

    # Read video
    vid_in = cv2.VideoCapture(vid_path)

    success, img = vid_in.read()
    height, width = img.shape[:2]

    # Write video
    vid_basename = os.path.splitext(os.path.basename(vid_path))[0] + '.avi'

    vid_out = cv2.VideoWriter(
        vid_dir + vid_basename,
        cv2.VideoWriter_fourcc(*'DIVX'),
        30,
        (width, height)
    )

    # Process each frame
    while success:
        vid_out.write(img_line_detection(img, process))
        success, img = vid_in.read()

    vid_out.release()

    return vid_dir + vid_basename


###############
# Main script #
###############

if __name__ == '__main__':
    # mkdir -p DESTINATION
    os.makedirs(DESTINATION, exist_ok=True)

    # Process video
    start = time.perf_counter()
    vid_line_detection(ORIGIN, Process, DESTINATION)
    end = time.perf_counter()

    print('video processed in %f seconds' % (end - start))
