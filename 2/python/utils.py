"""
ELEN0016-2 - Computer vision
University of Liège
Academic year 2019-2020

Student project - Part 2
Sudoku digit recognition and performance assessment
"""

#############
# Libraries #
#############

import os
import glob
import mimetypes

import numpy as np


#############
# Functions #
#############

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

        if not mime is None and 'image' in mime:
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

        if not mime is None and 'image' in mime:
            img_list.append(path)
    elif os.path.isdir(path):
        img_list = dir_to_img_list(path)

    return img_list
