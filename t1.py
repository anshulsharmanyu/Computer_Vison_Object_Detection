import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from itertools import product

def plothist(bins, weights,title,xlabel,ylabel):
    plt.bar(bins, weights, align='center')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def greyscale_Image(img):
    return (0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2])

def imageSmoothning(img):
    ####------- 3x3 kernel
    threeXThreekernel = np.ones((3,3),np.float32)/9
    #after3x3Smoothning1 = cv2.filter2D(img, -1, threeXThreekernel)
    after3x3Smoothning = np.copy(img)
    for i in range(1,len(img)-1):
        for j in range(1,len(img[i])-1):
            after3x3Smoothning[i][j] = (sum(map(sum, (threeXThreekernel * img[i-1:i+2,j-1:j+2]))))


    ####------- 5x5 kernel
    fiveXfivekernel = np.ones((5,5),np.float32)/25
    after5x5Smoothning = np.copy(img)
    for i in range(2,len(img)-2):
        for j in range(2,len(img[i])-2):
            after5x5Smoothning[i][j] = (sum(map(sum, (fiveXfivekernel * img[i-2:i+3,j-2:j+3]))))

    #after5x5Smoothning = cv2.blur(img,(5,5))

    plt.figure(1)
    plt.subplot(121), plt.imshow(img,cmap='gray'), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(after3x3Smoothning,cmap='gray'), plt.title('after3x3_Smoothing')
    plt.xticks([]), plt.yticks([])
    plt.show()

    plt.figure(2)
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(after5x5Smoothning, cmap='gray'), plt.title('after5x5_Smoothing')
    plt.xticks([]), plt.yticks([])
    plt.show()

    ####------zoomed difference
    plt.figure(3)
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.xlim(100, 200)
    plt.ylim(300, 200)
    plt.subplot(122), plt.imshow(after3x3Smoothning, cmap='gray'), plt.title('after3x3Smoothing_zoomed')
    plt.xticks([]), plt.yticks([])
    plt.xlim(100, 200)
    plt.ylim(300, 200)
    plt.show()

    plt.figure(4)
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.xlim(100, 200)
    plt.ylim(300, 200)
    plt.subplot(122), plt.imshow(after5x5Smoothning, cmap='gray'), plt.title('after5x5Smoothing_zoomed')
    plt.xticks([]), plt.yticks([])
    plt.xlim(100, 200)
    plt.ylim(300, 200)
    plt.show()
    return(after5x5Smoothning)

def edgeDetection(img):

    Ix, Iy, edge_map, orientation_map = np.zeros(img.shape), np.zeros(img.shape), np.zeros(img.shape), np.zeros(img.shape)
    # 3x1 and 1x3 masks
    Wdx = np.array([[-0.5], [0], [0.5]]).reshape(1,3)
    Wdy = np.array([-0.5,0,0.5]).reshape(1, 3)

    for i in range(1,len(img)-1):
        for j in range(1,len(img[0])-1):
            Ix[i][j] = (Wdx*img[i, j-1:j+2]).sum()
            Iy[i][j] = (Wdy*img[i-1:i+2, j]).sum()
    edge_map = np.hypot(Ix, Iy)
    orientation_map = np.arctan2(Iy, Ix)
    orientation_map= np.absolute(orientation_map)


    plt.figure(1)
    plt.subplot(121), plt.imshow(Ix, cmap='gray')
    plt.title('x derivative'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(Iy, cmap='gray'),
    plt.title('y derivative'), plt.xticks([]), plt.yticks([])
    plt.show()

    plt.figure(2)
    plt.imshow(edge_map, cmap='gray'),plt.title('Edge Map')
    plt.show()

    plt.figure(3)
    plt.imshow(orientation_map, cmap='gray'),plt.title('Orientation Map')
    plt.show()
    return (edge_map, orientation_map)


def thresholding(img, thresholdValue):
    img_copy = np.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] < thresholdValue: img_copy[i, j] = 0
    return img_copy

def templateMatching(templateImage,mainImage):
    # Displaying template image and main image
    plt.figure(1)
    plt.imshow(mainImage, cmap='gray'), plt.title('Main Images')
    plt.show()
    plt.figure(2)
    plt.imshow(templateImage, cmap='gray'), plt.title('Template Image')
    plt.show()

    # Calculating template mean intensity and image mean intensity
    temp_mean_intensity,img_gray_mean = (templateImage.sum()) / (templateImage.shape[0]**2),(mainImage.sum()) / (mainImage.shape[0]**2)
    # Initialising image after correaltion with template image, normalised template
    correlation_Image = np.zeros(mainImage.shape)
    normalisedTemplate = np.copy(templateImage)
    # Calculating zero mean template
    normalisedTemplate = normalisedTemplate - temp_mean_intensity

    a,b = templateImage.shape[0],templateImage.shape[1]

    x_boundary,y_boundary = math.ceil(a / 2),math.ceil(b / 2)

    for i in range(x_boundary, mainImage.shape[0] - x_boundary):
        for j in range(y_boundary, mainImage.shape[1] - y_boundary):
            correlation_Image[i, j] = (normalisedTemplate * (mainImage[i - x_boundary:i + a - x_boundary, j - y_boundary:j + b - y_boundary] - img_gray_mean)).sum()

    correlation_Image = correlation_Image[x_boundary:mainImage.shape[0] - x_boundary, y_boundary:mainImage.shape[1] - y_boundary]

    correlation_Image_smoothened = np.copy(correlation_Image)

    # Smoothing coorelated image with 5x5 kernel
    fiveXfivekernel = np.ones((5, 5), np.float32) / 25
    after5x5Smoothning = np.copy(correlation_Image)
    for i in range(2, len(correlation_Image) - 2):
        for j in range(2, len(correlation_Image[i]) - 2):
            correlation_Image_smoothened[i][j] = (sum(map(sum, (fiveXfivekernel * correlation_Image[i - 2:i + 3, j - 2:j + 3]))))
    # Applying threshholding to the correlated image
    img_after_threshold = thresholding(correlation_Image,0.6)
    # Multiplying laplacian with a value to have more prominent results
    laplacian = (np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).reshape(3, 3))
    peak_Image = np.copy(correlation_Image)
    # Calculating the peak image by correlating it with the above laplacian
    for i in range(1, correlation_Image.shape[0] - 1):
        for j in range(1, correlation_Image.shape[1] - 1):
            peak_Image[i, j] = (laplacian * correlation_Image_smoothened[i - 1:i + 2, j - 1:j + 2]).sum()
    # Applying threshholding to the peak image
    peak_after_threshold = thresholding(peak_Image, 0.6)

    plt.figure(3)
    plt.imshow(correlation_Image, cmap='gray'), plt.title('Coorelated image')
    plt.show()

    plt.figure(4)
    plt.imshow(img_after_threshold, cmap='gray'), plt.title('Image after Threshold')
    plt.show()

    plt.figure(5)
    plt.imshow(peak_Image, cmap='gray'), plt.title('peak image')
    plt.show()

    plt.figure(6)
    plt.imshow(peak_after_threshold, cmap='gray'), plt.title('Peak after Threshhold')
    plt.show()

if __name__== "__main__":

    ####------------ Image smothning

    img1 = cv2.imread('Images/cycle.jpg')
    img1 = greyscale_Image(img1)
    after5x5Smoothning = imageSmoothning(img1)

    ####------------ Edge Detection

    edgeDetection(after5x5Smoothning)


    ####------------ Template Matching

    templateImage = cv2.imread('Images/star.png')
    templateImage = greyscale_Image(templateImage)
    mainImage = cv2.imread('Images/shapes-bw.jpg')
    mainImage = greyscale_Image(mainImage)
    templateMatching(templateImage, mainImage)






