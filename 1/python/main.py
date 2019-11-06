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
import utils

from soccer import Process


#############
# Variables #
#############

# Path to an image, a video or a directory of images to process
ORIGIN = '../resources/images/soccer/'

# Path to export processed images
DESTINATION = '../products/soccer/'

# Assemble processed images into a video
MAKE_VIDEO = False

# Video framerate
FRAME_RATE = 2


###############
# Main script #
###############

if __name__ == '__main__':
    try:
        # Gather a list of image paths
        img_list = utils.to_img_list(ORIGIN)

        print('%d images found' % len(img_list))

        # Execute the line detection procedure for each image
        img_highlight_list = list()

        for i, img_path in enumerate(img_list):
            temp = utils.write_procedure(img_path, Process, DESTINATION)
            img_highlight_list.append(temp[-1])

            print('%d images processed' % (i + 1), end='\r')

        print('all images processed')

        # Assemble highlighted images into a video
        if MAKE_VIDEO:
            utils.img_list_to_video(
                img_highlight_list,
                DESTINATION + 'overlay_video.avi',
                FRAME_RATE
            )

            print('video processed')
    except Exception as e:
        print(e)
