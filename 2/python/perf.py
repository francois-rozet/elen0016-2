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
import utils

import numpy as np


#############
# Functions #
#############

def common(x, y):
    '''Proportion of common elements in x and y.'''
    return (x == y).mean()


def common_but_zeros(x, y):
    '''Proportion of common elements in x and y aside zeros.'''
    return np.logical_and(x == y, x != 0).sum() / (x != 0).sum()


##############
# Parameters #
##############

# Extension of anotation files
EXT = '.dat'

# Path to an image or a directory of images that have been processed
ORIGIN = '../resources/images/sudoku/'

# Path where annotation files were exported
DESTINATION = '../products/images/sudoku/'

# Performance evaluation function
EVAL = common


########
# Main #
########

if __name__ == '__main__':
    try:
        # Gather a list of image paths
        img_list = utils.to_img_list(ORIGIN)

        # Performance list
        perf = []

        # Confusion matrix
        conf = np.zeros((10, 10), dtype=int)

        for img_path in img_list:
            # Split the image name and extension
            img_name, _ = os.path.splitext(os.path.basename(img_path))

            try:
                # Load annotations
                ref_anot = np.loadtxt(ORIGIN + img_name + EXT, dtype=int)
                anot = np.loadtxt(DESTINATION + img_name + EXT, dtype=int)

                # Evaluate performance
                perf.append(EVAL(ref_anot, anot))

                # Confusion
                for a, b in zip(ref_anot.ravel(), anot.ravel()):
                    conf[a, b] += 1
            except Exception as e:
                print(e)

        # Mean performance
        perf = sum(perf) / len(perf)
        print('performance evaluated at {:f}'.format(perf))

        # Confusion matrix
        print('confusion matrix', conf)

    except Exception as e:
        print(e)
