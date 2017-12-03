import cv2
import numpy as np
from matplotlib import pyplot as plt
from enum import Enum

# constant variables
SKIP_VALUE = 200

plt.close('all')

# this is just to unconfuse pycharm
# this also fixes autocomplete when typed cv2. instead of cv2.cv2
try:
    from cv2 import cv2
except ImportError:
    pass


class NeighborhoodType(Enum):
    EIGHT = 8
    FOUR = 4


class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def to_string(self):
        return "({0}, {1})".format(self.x, self.y)


def getEmptyMask():
    return np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])


def loadImage(filename):
    return cv2.imread(filename)


def convertImageToGreyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def getGaussianFilterForSmoothing():
    return np.array([[2, 4, 5, 2, 2], [4, 9, 12, 9, 4], [5, 12, 15, 12, 5], [4, 9, 12, 9, 4], [2, 4, 5, 4, 2]]) / 156


# iterate the kernel over the image
def iterate(img, mask):
    imgRows = len(img)
    imgCols = len(img[0])
    maskRows = len(mask)
    maskCols = len(mask[0])
    maskCenterX = np.int32(maskRows / 2);
    maskCenterY = np.int32(maskCols / 2);
    retImg = np.zeros(shape=(imgRows, imgCols))
    for i in range(0, imgRows):
        for j in range(0, imgCols):
            for m in range(0, maskRows):
                mm = maskRows - 1 - m
                for n in range(0, maskCols):
                    nn = maskCols - 1 - n
                    # check boundaries
                    ii = i + m - maskCenterX
                    jj = j + n - maskCenterY
                    if (ii >= 0 and ii < imgRows and jj >= 0 and jj < imgCols):
                        retImg[i][j] += img[ii][jj] * mask[mm][nn];
    return retImg


def smoothing(img):
    print("starting smoothing image...")
    return iterate(img, getGaussianFilterForSmoothing())


def getBinaryImage():
    return np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


def computeConnectedComponents(img):
    rows = len(img)
    cols = len(img[0])
    visited = np.zeros(shape=(rows, cols))
    # We perform one - pass algorithm -> Check each pixel of the image
    currentLabel = 1;
    count=0
    for i in range(0, rows):
        for j in range(0, cols):
            # If it is background skip it
            if (img[i][j] == 0):
                visited[i][j] = SKIP_VALUE
            # If it is already visited skip it
            if (visited[i][j] == SKIP_VALUE):
                continue
            # assigning a label to output image
            img[i][j] = currentLabel

            points = []
            # add location point
            points.append(Point(i, j))
            # checking its neighbors
            while len(points) != 0:
                # get last element of the point list
                p = points[(len(points) - 1)]
                # then remove it
                points.pop()

                # Check if it is visited in the while loop
                if (visited[p.x][p.y] == SKIP_VALUE):
                    continue
                # Update the label
                img[p.x][p.y] = currentLabel
                visited[p.x][p.y] = SKIP_VALUE

                # Get the neighbors - 4 Connectivity
                if (p.x > 1):
                    if (visited[p.x - 1][p.y] == 0 and img[p.x - 1][p.y] != 0):
                        points.append(Point(p.x - 1, p.y))

                if (p.y > 1):
                    if (visited[p.x][p.y - 1] == 0 and img[p.x][p.y - 1] != 0):
                        points.append(Point(p.x, p.y - 1))
                if (p.x < rows - 1):
                    if (visited[p.x + 1][p.y] == 0 and img[p.x + 1][p.y] != 0):
                        points.append(Point(p.x + 1, p.y))
                if (p.y < cols - 1):
                    if (visited[p.x][p.y + 1] == 0 and img[p.x][p.y + 1] != 0):
                        points.append(Point(p.x, p.y + 1))
            # increase current label
            currentLabel += 1

    return img


bird1 = loadImage("bird 1.jpg")
bird2 = loadImage("bird 2.jpg")
bird3 = loadImage("bird 3.bmp")

# convert grayscale
# img = convertImageToGreyscale(bird1);
# smooting the image before segmentation
# img = smoothing(img)
# get binary image from the smoothed image
img = getBinaryImage()
img = computeConnectedComponents(img)
img = img* (255/img.max())#perform normalization

# show image
plt.close('all')
plt.imshow(img, cmap="gray")
plt.show()
