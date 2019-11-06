"""
ELEN0016-2 - Computer vision
University of Liège
Academic year 2019-2020

Student project - Part 1
Main modules development and line extraction

Authors :
    - Maxime Meurisse       (m.meurisse@student.uliege.be)
    - François Rozet        (francois.rozet@student.uliege.be)
    - Océane Rumfels        (oceane.rumfels@student.uliege.be)
    - Valentin Vermeylen    (valentin.vermeylen@student.uliege.be)
"""

#############
# Libraries #
#############

import os
import cv2
import glob
import mimetypes

import numpy as np


#############
# Functions #
#############

def img_list_to_vid(img_list, vid_path, frame_rate):
    """Create and write a video based on a list of
    image paths

    Inputs
    ------
    img_list :      a list of image paths
    vid_path :    complete path where to save the
                    video file
    frame_rate :    framerate of the video
    """

    height, width = cv2.imread(img_list[0], cv2.IMREAD_GRAYSCALE).shape

    out = cv2.VideoWriter(
        vid_path,
        cv2.VideoWriter_fourcc(*'DIVX'),
        frame_rate,
        (width, height)
    )

    for img_path in img_list:
        out.write(cv2.imread(img_path, cv2.IMREAD_COLOR))

    out.release()


def vid_to_img_list(vid_path, img_dir):
    """Breaks down a video into its component images

    Inputs
    ------
    vid_path :  complete path to the video
    img_dir :   directory where to save the image

    Output
    ------
    img_list :  a list with all video's image paths
    """

    vid = cv2.VideoCapture(vid_path)
    img_list = []

    while True:
        success, img = vid.read()

        if not success:
            break

        img_list.append(
            os.path.dirname(img_dir) + '_{:06d}'.format(len(img_list)) + '.png'
        )
        cv2.imwrite(img_list[-1], img)

    return img_list


def dir_to_img_list(img_dir):
    """Gather the path of all images within a directory

    Input
    -----
    img_dir :   a path to a directory

    Output
    ------
    img_list :  a list of image paths
    """

    img_list = list()

    for img_path in glob.glob(os.path.join(img_dir, '*')):
        mime = mimetypes.guess_type(img_path)[0]

        if 'image' in mime:
            img_list.append(img_path)

    img_list.sort()

    return img_list


def to_img_list(path):
    """Gather a list of image paths from the path to an image,
    a video or a directory

    Input
    -----
    path :      a path to an image, a video or a directory

    Output
    ------
    img_list :  a list of image paths
    """

    img_list = list()

    if os.path.isfile(path):
        mime = mimetypes.guess_type(path)[0]

        if 'image' in mime:
            img_list.append(path)
        elif 'video' in mime:
            img_dir = os.path.splitext(path)[0] + '/'

            img_list = vid_to_img_list(path, img_dir)
    elif os.path.isdir(path):
        img_list = dir_to_img_list(path)

    return img_list


def highlight(img, lines, thick = 3):
    """Highlight lines in an image

    Inputs
    ------
    img :   an image
    lines : a list of lines

    Outputs
    -------
    the image in which lines are highlighted
    """

    if lines is None:
        return img

    over = img.copy()

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(over, (x1, y1), (x2, y2), (0, 0, 255), thick)

    return cv2.addWeighted(over, 0.5, img, 0.5, 0)


def classification(img, lines, thick = 3):
    """Classifies the edges

    Inputs
    ------
    img :   an image of extracted edges
    lines : a list of lines

    Outputs
    -------
    the image in which edges are classified
    """
    if lines is None:
        return img

    # Mask creation
    mask = np.zeros(img.shape, np.uint8)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(mask, (x1, y1), (x2, y2), 255, 3)

    # Edges selection
    selected = cv2.bitwise_and(img, img, mask=mask)
    selected = cv2.cvtColor(selected, cv2.COLOR_GRAY2BGR)

    # Edges coloring
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img[:, :, :2] -= selected[:, :, :2]

    return img


def procedure(img_path, process):
    """Apply a complete line detection procedure
    to an image

    Inputs
    ------
    img_path :  a path to an image
    process :   a processing class

    Outputs
    -------
    a generator over all processed images
    """

    # Read the image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    yield img

    # Preprocessing
    temp = process.preprocessing(img)
    yield temp

    # Edges extraction
    temp = process.edge_points_extraction(temp)
    yield temp

    # Lines detection
    lines = process.line_detection(temp)

    # Edges classification
    temp = classification(temp, lines)
    yield temp

    # Lines highlighting
    temp = highlight(img, lines)
    yield temp


def write_procedure(img_path, process, img_dir = ''):
    """Apply a complete procedure to detect lines on
    an image and write the processed images

    Inputs
    ------
    img_path :  a path to an image
    process :   a processing class
    img_dir :   optional, the writing directory

    Outputs
    -------
    img_list :  the list of path to all processed
                images
    """

    # mkdir -p img_dir
    os.makedirs(img_dir, exist_ok=True)

    # Split the image name and extension
    img_name, img_ext = os.path.splitext(os.path.basename(img_path))

    # Write the processed images
    img_list = list()

    for i, img in enumerate(procedure(img_path, process)):
        img_list.append(img_dir + img_name + '_{:02d}'.format(i) + img_ext)
        cv2.imwrite(img_list[-1], img)

    return img_list
