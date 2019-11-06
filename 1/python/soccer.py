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

import numpy as np
import cv2


#########
# Class #
#########

class Process:

    ####################
    # Class attributes #
    ####################

    # Preprocessing parameters
    lower = np.uint8([35, 50, 50])
    upper = np.uint8([70, 255, 255])

    # Edge points extraction parameters
    lo_thresh = 80
    hi_thresh = 180
    filter_size = 3

    # Line detection parameters
    rho = 1
    theta = np.pi / 180
    thresh = 100
    min_line_length = 100
    max_line_gap = 25

    ###########
    # Methods #
    ###########

    def preprocessing(img):
        # Hue filter
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, Process.lower, Process.upper)

        img = cv2.bitwise_and(img, img, mask=mask)

        # Gray scale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img

    def edge_points_extraction(img):
        # Canny's algorithm
        img = cv2.Canny(
            img,
            Process.lo_thresh,
            Process.hi_thresh,
            apertureSize=Process.filter_size,
            L2gradient=True
        )

        return img

    def line_detection(img):
        # Hough transform
        lines = cv2.HoughLinesP(
            img,
            Process.rho,
            Process.theta,
            Process.thresh,
            minLineLength=Process.min_line_length,
            maxLineGap=Process.max_line_gap
        )

        return lines
