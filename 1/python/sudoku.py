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
    d = 10
    sigma_color = 10
    sigma_space = 100

    # Edge points extraction parameters
    lo_thresh = 80
    hi_thresh = 180
    filter_size = 3
    blur_size = 3

    # Line detection parameters
    rho = 1
    theta = np.pi / 180
    thresh = 200
    min_line_length = 300
    max_line_gap = 10

    ###########
    # Methods #
    ###########

    def preprocessing(img):
        # Gray scale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Bilateral filtering
        img = cv2.bilateralFilter(
            img,
            Process.d,
            Process.sigma_color,
            Process.sigma_space
        )

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

        # Gaussian Blur
        img = cv2.GaussianBlur(img, (Process.blur_size, Process.blur_size), 0)

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
