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

import numpy as np
import cv2


###########
# Methods #
###########

def vertices(rect):
    # Array
    rect = np.array(cv2.boxPoints(rect))

    # Find top-left
    i = np.argmin(rect.sum(axis=1))

    # Circular shift
    rect = np.roll(rect, -i, axis=0)

    return rect

def warp(img, rect):
    # Dimensions
    w, h = rect[1]
    size = int(np.sqrt(w * h))

    # Perspective Transform
    src = vertices(rect).astype('f')
    dst = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ]).astype('f')

    matrix = cv2.getPerspectiveTransform(src, dst)

    # Warp
    img = cv2.warpPerspective(img, matrix, (size, size), borderValue=255)

    return img

def preprocessing(img):
    # Gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img

def grid_threshold(img):
    # Parameters
    shape = np.array(img.shape)

    block_size = shape.mean().astype(int) // (9 * 9)
    block_size += block_size % 2 + 1
    block_size = block_size if block_size > 1 else 3

    c = 20

    # Threshold
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, c)

    return img

def grid_detection(img):
    # Parameters
    ratio_thresh = 1 / 5

    # Find contours
    ctns, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Grid search
    grid = None
    area = np.array(img.shape).prod() / 4

    for ctn in ctns:
        _, (w, h), _ = rect = cv2.minAreaRect(ctn)
        if area < w * h and abs(1 - w / h) < ratio_thresh:
            grid = rect
            area = w * h

    return grid

def cell_threshold(img):
    # Parameters
    shape = np.array(img.shape)

    block_size = shape.mean().astype(int) // (3 * 9)
    block_size += block_size % 2 + 1
    block_size = block_size if block_size > 1 else 3

    c = 5

    # Adaptive threshold
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, c)

    return img

def cell_detection(img):
    # Parameters
    shape = np.array(img.shape)

    area_thresh = 2

    kernel = np.ones((3, 3))
    iterations = 5

    # Find contours
    ctns, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Contours filtering
    cell_shape = shape / 9
    cell_area = cell_shape.prod()

    for ctn in ctns:
        peri = cv2.arcLength(ctn, True)
        area = cv2.contourArea(ctn)
        if (area < cell_shape.mean() or area > peri / 2) and area < cell_area / area_thresh:
            cv2.drawContours(img, [ctn], 0, 0, -1)

    # Dilate
    img = cv2.dilate(img, kernel, iterations=iterations)

    # Cells search
    ctns, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cells = []

    for ctn in ctns:
        _, (w, h), _ = rect = cv2.minAreaRect(ctn)
        if w * h < cell_area * area_thresh and w * h > cell_area / area_thresh ** 2:
            cells.append(rect)

    return cells

def draw(img, cells):
    # Parameters
    color = (0, 0, 255)
    thickness = 1

    # RBG
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw contours
    cells = [vertices(i).astype(int) for i in cells]
    cv2.drawContours(img, cells, -1, color, thickness)

    return img

def procedure(img_path):
    # Read the image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    yield img

    # Preprocessing
    img = preprocessing(img)
    yield img

    # Grid threshold
    thresh = grid_threshold(img)
    yield thresh

    # Grid detection
    grid = grid_detection(thresh)
    if not grid is None:
        img = warp(img, grid)
    yield img

    # Thresholding
    thresh = cell_threshold(img)
    yield thresh

    # Cell detection
    cells = cell_detection(thresh)
    if len(cells) != 81:
        print(len(cells))
    img = draw(img, cells)
    yield img

def fast(img_path):
    # Read the image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # Preprocessing
    img = preprocessing(img)

    # Grid threshold
    thresh = grid_threshold(img)

    # Grid detection
    grid = grid_detection(thresh)
    if not grid is None:
        img = warp(img, grid)

    # Cell threshold
    thresh = cell_threshold(img)

    # Cell detection
    cells = cell_detection(thresh)
    return img, cells
