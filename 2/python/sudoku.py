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
    '''Transforms a rotated rectangle into vertices.'''
    # Array
    rect = np.array(cv2.boxPoints(rect))

    # Find top-left
    i = np.argmin(rect.sum(axis=1))

    # Circular shift
    rect = np.roll(rect, -i, axis=0)

    return rect


def warp(img, ctn, contour=False):
    '''Corrects perspective according to a rectangle/contour.'''

    # Source and destination
    if contour:
        tl = np.argmin(ctn[:,0,0] + ctn[:,0,1])
        tr = np.argmin(ctn[:,0,0] - ctn[:,0,1])
        br = np.argmin(-ctn[:,0,0] - ctn[:,0,1])
        bl = np.argmin(-ctn[:,0,0] + ctn[:,0,1])

        ctn = ctn[[tl, bl, br, tr]]

        area = cv2.contourArea(ctn)
        src = ctn[:,0,:].astype('f')
    else:
        w, h = ctn[1]

        area = w * h
        src = vertices(ctn).astype('f')

    size = int(np.sqrt(area))

    dst = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ]).astype('f')

    # Perspective Transform
    matrix = cv2.getPerspectiveTransform(src, dst)

    # Warp
    img = cv2.warpPerspective(img, matrix, (size, size), borderValue=255)

    return img


def preprocessing(img):
    '''Preprocesses an image.'''

    # Gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def threshold(img, n_block, c):
    '''Performs an adaptive threshold on an image.'''

    # Parameters
    shape = np.flip(np.array(img.shape), 0)

    block_size = shape.mean().astype(int) // n_block
    block_size += block_size % 2 + 1
    block_size = block_size if block_size > 1 else 3

    # Threshold
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, c)

    return img


def grid_detection(img):
    '''Detects the grid within a thresholded image.'''

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
            grid = ctn
            area = w * h

    return grid


def cell_detection(img):
    '''Detects cells within a thresholded image.'''

    # Parameters
    shape = np.flip(np.array(img.shape), 0)

    area_thresh = 2
    ratio_thresh = 1 / 4

    kernel = np.ones((3, 3))
    iterations = 5

    thickness = 2
    margin = 4

    # Find contours
    ctns, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Contours filtering
    cell_shape = shape / 9
    cell_area = cell_shape.prod()

    for ctn in ctns:
        peri = cv2.arcLength(ctn, True)
        area = cv2.contourArea(ctn)
        if area < cell_shape.mean() or (area > peri / 2 and area < cell_area / area_thresh):
            # Remove inadequate contours
            cv2.drawContours(img, [ctn], 0, 0, -1)

    # Dilate
    img = cv2.dilate(img, kernel, iterations=iterations)

    # Draw outer rectangle
    cv2.rectangle(img, (0, 0), tuple(shape - 1), 255, thickness=thickness)

    # Find contours
    ctns, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Cells list
    cells = []

    # Contours filtering
    for ctn in ctns:
        (x, y), (w, h), angle = cv2.minAreaRect(ctn)
        w, h = w + margin, h + margin
        if abs(1 - w / h) < ratio_thresh and  w * h < cell_area * area_thresh and w * h > cell_area / area_thresh:
            # Add to cells list
            rect = (x, y), (w, h), angle
            cells.append(rect)

    return cells


def cell_filter(cells):
    '''Orders cells and guesses location of missing cells.'''

    # Parameters
    n = 9

    # Condition
    if len(cells) < 2 / 3 * n ** 2:
        return []

    # Numpy arrays
    pos = np.array([[x, y] for (x, y), _, _ in cells])
    shape = np.array([[w, h] for _, (w, h), _ in cells])
    area = shape.prod(axis=1)

    size = shape.mean()

    # Outliers
    outlier = np.full(pos.shape[0], False)

    for i in range(pos.shape[0]):
        dist = np.abs(pos - pos[i]).sum(axis=1)
        # Nearest cell from cell i
        j = np.argpartition(dist, 1)[1]

        if dist[j] < size / 2:
            # Remove smallest between i and j
            outlier[i if area[i] < area[j] else j] = True

    good = np.logical_not(outlier)

    # Keep non outliers
    cells = [cell for i, cell in enumerate(cells) if good[i]]
    pos = pos[good]
    shape = shape[good].mean(axis=0)

    # Size as mean distance between cells
    dist = np.empty(pos.shape[0])
    for i in range(pos.shape[0]):
        dist[i] = np.partition(np.abs(pos - pos[i]).sum(axis=1), 1)[1]

    z_score = (dist - dist.mean()) / dist.std()
    size = dist[z_score < 1.96].mean()

    # Grid building
    grid = np.full((n, n), None, dtype=object)
    x, y = 0, 0

    while True:
        # Expected postion from neighbors
        if x > 0 and y > 0:
            expected = np.array(grid[x - 1, y][0]) + np.array(grid[x, y - 1][0]) + np.array(grid[x - 1, y - 1][0]) + 2 * np.array([size, size])
            expected /= 3
        elif x > 0:
            expected = np.array(grid[x - 1, y][0]) + np.array([size, 0])
        elif y > 0:
            expected = np.array(grid[x, y - 1][0]) + np.array([0, size])
        else:
            expected = np.amin(pos, axis=0)

        dist = np.abs(pos - expected).sum(axis = 1)
        i = np.argmin(dist)

        # If cell[i] is close enough to grid[x, y]
        if dist[i] < size / 2:
            grid[x, y] = cells[i]
        else:
            # Create new cell
            grid[x, y] = tuple(expected), tuple(shape), 0

        # Next cell
        if x < n - 1:
            if y > 0:
                x, y = x + 1, y - 1
            else:
                x, y = y, x + 1
        else:
            if y < n - 1:
                x, y = y + 1, x
            else:
                break

    cells = grid.ravel().tolist()

    return cells


def digit_recognition(img, cells, model):
    '''Uses model to recognize digits within cells.'''

    digits = np.zeros(len(cells), dtype=int)

    for i, cell in enumerate(cells):
        digits[i] = model(warp(img, cell))

    return digits


def draw(img, cells):
    '''Draws cells over an image.'''

    # Parameters
    color = (0, 0, 255)
    thickness = 1

    # RBG
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw
    ctns = [vertices(i).astype(int) for i in cells]
    cv2.drawContours(img, ctns, -1, color, thickness)

    return img


def overlay(img, cells, digits):
    '''Draws digits over an image.'''

    # Parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.66
    color = (0, 0, 255)

    # RGB
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Text
    for i, cell in enumerate(cells):
        if digits[i] != 0:
            cv2.putText(img, str(digits[i]), cell[0], font, scale, color)

    return img


def procedure(img_path, model=None):
    '''Executes the digit recognition procedure. Returns images.'''

    # Read the image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    yield img

    # Preprocessing
    img = preprocessing(img)
    yield img

    # Grid detection
    thresh = threshold(img, 9 * 9, 15)
    yield thresh

    grid = grid_detection(thresh)
    if grid is None:
        return None

    img = warp(img, grid, contour=True)
    yield img

    # Cell detection
    thresh = threshold(img, 3 * 9, 10)
    yield thresh

    cells = cell_detection(thresh)
    yield draw(img, cells)

    # Cell filter
    cells = cell_filter(cells)
    if len(cells) == 0:
        return None

    yield draw(img, cells)

    # Digit recognition
    if model is None:
        return None

    digits = digit_recognition(img, cells, model)
    yield overlay(img, cells, digits)


def fast(img_path, model):
    '''Executes the digit recognition procedure. Returns digit matrix.'''

    # Read the image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # Preprocessing
    img = preprocessing(img)

    # Grid detection
    thresh = threshold(img, 9 * 9, 15)
    grid = grid_detection(thresh)
    if grid is None:
        return None

    img = warp(img, grid, contour=True)

    # Cell detection
    thresh = threshold(img, 3 * 9, 10)
    cells = cell_detection(thresh)

    # Cell filter
    cells = cell_filter(cells)
    if len(cells) == 0:
        return None

    # Digit recognition
    digits = digit_recognition(img, cells, model)
    return digits.reshape((9, 9)).transpose()
