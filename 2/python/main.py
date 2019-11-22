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
import cv2
import utils

import sudoku as process


##############
# Parameters #
##############

# Path to an image, a video or a directory of images to process
ORIGIN = '../../1/resources/images/sudoku/'

# Path to export processed images
DESTINATION = '../products/sudoku/'


###############
# Main Script #
###############

if __name__ == '__main__':
    try:
        # Gather a list of image paths
        img_list = utils.to_img_list(ORIGIN)

        print('%d images found' % len(img_list))

        # mkdir -p DESTINATION
        os.makedirs(DESTINATION, exist_ok=True)

        # Execute the detection procedure for each image
        for img_path in img_list:
            # Split the image name and extension
            img_name, img_ext = os.path.splitext(os.path.basename(img_path))
            img_dst = DESTINATION + img_name + '_{:02d}' + img_ext

            # Write the processed images
            for i, img in enumerate(process.procedure(img_path)):
                cv2.imwrite(img_dst.format(i), img)

            print(img_name + img_ext + ' processed')

        print('all images processed')
    except Exception as e:
        print(e)
