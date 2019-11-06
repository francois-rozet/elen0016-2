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

    # Edge points extraction parameters
    lo_thresh = 100
    hi_thresh = 200
    filter_size = 3

    # Line detection parameters
    rho = 3
    theta = np.pi / 180
    thresh = 80
    min_line_length = 50
    max_line_gap = 35

    ###########
    # Methods #
    ###########

    def _select_rgb_white(img):
        """Take an image in RGB and extract the white components
        Remark : image is expected in RGB color space

        Input
        -----
        img :   an image in RGB

        Output
        ------
        masked : the image of which all non white components are black
        """

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        sensitivity = 20
        lower = np.array([0, 0, 255 - sensitivity])
        upper = np.array([255, 255, 255])

        w_mask = cv2.inRange(img, lower, upper)

        masked = cv2.bitwise_and(img, img, mask=w_mask)
        masked = cv2.cvtColor(masked, cv2.COLOR_HSV2BGR)

        return masked

    def preprocessing(img):
        # Highlighting the white spots
        img = Process._select_rgb_white(img)

        # Gray scale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Remove the sky
        thresh, img = cv2.threshold(img, 127, 255, 0)
        mask = np.zeros(img.shape, np.uint8)

        contours, hier = cv2.findContours(
            img,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            if 8000 < cv2.contourArea(cnt):
                cv2.drawContours(mask, [cnt], 0, 255, -1)

        # Remove the big blobs of white (sky)
        cv2.bitwise_not(img, img, mask)

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

        # Cut the higher quarter of the image as it is usually sky
        height, width = img.shape
        img[:height // 4, :] = 0

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
