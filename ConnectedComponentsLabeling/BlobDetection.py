import cv2
import numpy as np
from matplotlib import pyplot as plt
from enum import Enum

# constant variables
SKIP_VALUE = 200
MAX_GREY_VALUE = 256

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


def convertToBinaryImage(img):
    return img


# this computes the automatic threshold value from the greyscale image intensity values
def computeThreshold(img):
    print("compute thresholding...")
    histData = computeHistogram(img)
    currentMaxValue, threshold, sum, sumF, sumB, wB, wF, varBetween, meanB, meanF = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    totalPixel = len(img) * len(img[0])
    count = 0
    while count < MAX_GREY_VALUE:
        sum += count * histData[0][count]
        count = count + 1

    for i in range(0, MAX_GREY_VALUE):
        wB += histData[0][i]
        wF = totalPixel - wB
        if wF == 0:
            break
        sumB += i * histData[0][i]
        sumF = sum - sumB
        meanB = sumB / wB
        meanF = sumF / wF
        between = wB * wF * (meanB - meanF) * (meanB - meanF)
        if between > currentMaxValue:
            currentMaxValue = between
            threshold = i
    print("calculated threshold value is " + str(threshold))
    return threshold


# calculate histogram of grayscale image
def computeHistogram(img):
    histData = []
    rows = len(img)
    cols = len(img[0])
    for i in range(0, rows):
        for j in range(0, cols):
            intensity = img[i][j]
            histData.append(intensity)
    return np.histogram(histData, bins=MAX_GREY_VALUE)


def computeConnectedComponents(img):
    rows = len(img)
    cols = len(img[0])
    visited = np.zeros(shape=(rows, cols))
    # We perform one - pass algorithm -> Check each pixel of the image
    currentLabel = 1;
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


# applying threshold value to image in order to get binary image(0,1) - 0 background 1 foregground
def applyThreshold(img):
    rows = len(img)
    cols = len(img[0])
    t = computeThreshold(img)
    for i in range(0, rows):
        for j in range(0, cols):
            if img[i][j] >= t:
                img[i][j] = 1.0
            else:
                img[i][j] = 0.0

    return img


# normaliza the labeled data
def performNormalization(img):
    rows = len(img)
    cols = len(img[0])
    for i in range(0, rows):
        for j in range(0, cols):
            if (img[i][j] > 0):
                img[i][j] = 255 / img[i][j]
    return img


bird1 = loadImage("bird1.jpg")
bird2 = loadImage("bird2.jpg")
bird3 = loadImage("bird3.bmp")

# convert grayscale
img = convertImageToGreyscale(bird1);
# smoothing the image before segmentation
img = smoothing(img)
# applying threshold to the image
img = applyThreshold(img)
# calculate connected components labeling
img = computeConnectedComponents(img)
# perform normalization
img = performNormalization(img)

count = int(img.max())
print("count: " + str(count))

# show image
plt.close('all')
plt.imshow(img, cmap="gray")
plt.show()
