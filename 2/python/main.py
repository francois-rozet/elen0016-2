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
import numpy as np
import utils

import sudoku as process


##############
# Parameters #
##############

# Path to an image or a directory of images to process
ORIGIN = '../resources/images/sudoku/'

# Path to export processed files
DESTINATION = '../products/images/sudoku/'

# Model
MODEL = None

# Write step images
SAVE_STEPS = True


########
# Main #
########

if __name__ == '__main__':
    try:
        # Gather a list of image paths
        img_list = utils.to_img_list(ORIGIN)

        print('%d images found' % len(img_list))

        # mkdir -p DESTINATION
        os.makedirs(DESTINATION, exist_ok=True)

        # Execute the procedure for each image
        for img_path in img_list:
            # Split the image name and extension
            img_name, img_ext = os.path.splitext(os.path.basename(img_path))
            img_dst = DESTINATION + img_name + '_{:02d}' + img_ext

            try:
                if SAVE_STEPS:
                    # Write the processed images
                    for i, img in enumerate(process.procedure(img_path, MODEL)):
                        cv2.imwrite(img_dst.format(i), img)

                if MODEL is not None:
                    # Write the processed digits
                    grid = process.fast(img_path, MODEL)
                    np.savetxt(DESTINATION + img_name + '.dat', grid, fmt='%d')
            except Exception as e:
                print(e)

            print(img_name + img_ext + ' processed')

        print('all images processed')
    except Exception as e:
        print(e)
