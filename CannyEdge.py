import cv2
import numpy as np
from matplotlib import pyplot as plt
from enum import Enum
import math

plt.close('all')

# this is just to unconfuse pycharm
# this also fixes autocomplete when typed cv2. instead of cv2.cv2
try:
    from cv2 import cv2
except ImportError:
    pass

class SobelType(Enum):
    X = "x"
    Y = "y"

def loadImage(filename):
    return cv2.imread(filename)

def convertImageToGreyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def getSobelOperator(type):
    if (type == SobelType.X.value):
        return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    if (type == SobelType.Y.value):
        return np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

def getGaussianFilterForSmoothing():
    return np.array([[2, 4, 5, 2, 2], [4, 9, 12, 9, 4], [5, 12, 15, 12, 5], [4, 9, 12, 9, 4], [2, 4, 5, 4, 2]]) / 156

def getEmptyMask():
    return np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

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

def findGradients(img):
    print("find gradients")
    gmx = getGradientMagnitute(iterate(img, getSobelOperator(SobelType.X.value)))
    gmy = getGradientMagnitute(iterate(img, getSobelOperator(SobelType.Y.value)))
    # total gradient magnitute
    gmag = gmx + gmy
    # gradient orientation(direction)
    theta = quantizedTheta(getGradientDirection(gmx, gmy))
    return nonMaximumSuppression(gmag, theta)

def quantizedTheta(theta):
    row = len(theta)
    col = len(theta[0])
    for i in range(0, row):
        for j in range(0, col):
            t = theta[i][j]
            if ((t >= 0 and t <= 22.5) or (t > 157.5 and t <= 180)):
                # set to zero
                theta[i][j] = 0
            elif (t > 22.5 and t <= 67.5):
                # set to 45
                theta[i][j] = 45
            elif (t > 67.5 and t <= 112.5):
                theta[i][j] = 90
            elif (t > 112.5 and t <= 157.5):
                theta[i][j] = 135
    return theta

def getGradientMagnitute(grad):
    print("image gradient magnitute")
    imgRows = len(grad)
    imgCols = len(grad[0])
    mag = np.zeros(shape=(imgRows, imgCols))
    for i in range(0, imgRows):
        for j in range(0, imgCols):
            mag[i][j] = math.sqrt(grad[i][j] * grad[i][j])
    return mag

def getGradientDirection(gx, gy):
    print("gradient direction")
    imgRows = len(gx)
    imgCols = len(gx[0])
    theta = np.zeros(shape=(imgRows, imgCols))
    for i in range(0, imgRows):
        for j in range(0, imgCols):
            theta[i][j] = math.atan2(gy[i][j], gx[i][j])
    # degree convert for each pixel
    return theta * 180 / math.pi

# use theta and gradient magnitute vekctors
def nonMaximumSuppression(gmag, theta):
    print("non maximum suppression")
    imgRows = len(gmag)
    imgCols = len(gmag[0])
    retImg = np.zeros(shape=(imgRows, imgCols))
    for i in range(1, imgRows - 1):
        for j in range(1, imgCols - 1):
            # 0 degrees
            if (theta[i][j] == 0):
                if gmag[i][j] >= gmag[i][j + 1] and gmag[i][j] >= gmag[i][j - 1]:
                    retImg[i][j] = gmag[i][j]
            # 45 degrees
            if (theta[i][j] == 45):
                if gmag[i][j] >= gmag[i - 1][j + 1] and gmag[i][j] >= gmag[i + 1][j - 1]:
                    retImg[i][j] = gmag[i][j]
            # 90 degrees
            if (theta[i][j] == 90):
                if gmag[i][j] >= gmag[i - 1][j] and gmag[i][j] >= gmag[i + 1][j]:
                    retImg[i][j] = gmag[i][j]
            if (theta[i][j] == 135):
                if gmag[i][j] >= gmag[i - 1][j - 1] and gmag[i][j] >= gmag[i + 1][j + 1]:
                    retImg[i][j] = gmag[i][j]
    return retImg

def applyHysteresis(img):
    print("calculate hysteresis")
    imgRows = len(img)
    imgCols = len(img[0])
    isAnyStrong = False
    print(img)
    for i in range(0, imgRows - 1):
        for j in range(0, imgCols - 1):
            # center is weak then check rest if there is any strong edge
            if img[i + 1][j + 1] == 0.5:
                for m in range(0, 3):
                    for n in range(0, 3):
                        if img[m + i][n + j] == 1.0:
                            isAnyStrong = True
                            break
            if isAnyStrong == False:
                img[i][j] == 0.0
    return img

def smoothing(img):
    return iterate(img, getGaussianFilterForSmoothing())

def thresholding(img, lowFactor, highFactor):
    print("thresholding")
    imgRows = len(img)
    imgCols = len(img[0])
    retImg = np.zeros(shape=(imgRows, imgCols))
    # get the maximum value of image then decide low and high threshold values by multiplying low and high factors
    maxValue = img.max()
    low = lowFactor * maxValue
    high = highFactor * maxValue
    for i in range(0, imgRows):
        for j in range(imgCols):
            if img[i][j] >= high:
                retImg[i][j] = 1.0
            elif img[i][j] >= low:
                retImg[i][j] = 0.5
    # return retImg
    return retImg

def computeCanny(img):
    # smoothing image by using gausian filter
    img = smoothing(img)
    # find gradient magnitudes and directions
    img = findGradients(img)
    # apply double thresholding
    img = thresholding(img, lowFactor=0.2, highFactor=0.8)
    # apply hysteresis to remove weak edges
    img = applyHysteresis(img)
    return img



# start loading image
img_lenna = loadImage("Lenna.png")
img_fruit = loadImage("fruit-bowl.jpg")
img_house = loadImage("house.jpg")
img_woman = loadImage("woman.JPG")
img_camemaran = loadImage("cameraman.jpg")

# convert grayscale
img = convertImageToGreyscale(img_lenna);
img = computeCanny(img)

# show image
plt.close('all')
plt.imshow(img, cmap="gray")
plt.show()
