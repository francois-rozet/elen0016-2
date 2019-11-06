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

import numpy as np                      # numpy
from matplotlib import pyplot as plt    # matplotlib


#############
# Functions #
#############

def multi_plot(n, m, img_tuple, title_tuple, cmap_tuple=None):
    """Plot multiple images in a grid.
    This function is inspired by the 'multiPlot'
    function of Ph. Latour given in T.P. 1 of the course.

    Parameters
    ----------
    n : number of elements in the x-axis
    m : number of elements in the y-axis
    img_tuple : a tuple or list of the n * m images to plot
    title_tuple : a tuple or list of the image titles
    cmap_tuple : a tuple or list of the image color maps
    """
    plt.figure(figsize=(20, 10))

    for i in np.arange(n * m):
        if img_tuple[i] is None:
            continue

        if cmap_tuple is not None:
            cmap = cmap_tuple[i]
        else:
            cmap = None

        plt.subplot(n, m, i + 1)
        plt.imshow(img_tuple[i], cmap=cmap)

        plt.xticks([])
        plt.yticks([])

        plt.title(title_tuple[i])

    plt.show()
