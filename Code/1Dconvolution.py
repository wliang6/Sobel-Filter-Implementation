#Sobel Separable Filter implementation in 1D Convolution
#Author: Winnie Liang

import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

image = cv2.imread('lena_gray.jpg')

#Sobel Separable Filter in x direction
sobelx1 = np.array([[1,2,1]], dtype = np.float)  #column
sobelx2 = np.array([[-1,0,1]], dtype = np.float) #row


#Sobel Separable Filter in y direction
sobely1 = np.array([[-1,0,1]], dtype = np.float) #column
sobely2 = np.array([[1,2,1]], dtype = np.float)  #row

#Start time
timestart = time.clock()

#Calculate gx and gy using Sobel (horizontal and vertical gradients)
gx = cv2.sepFilter2D(image, -1, sobelx2, sobelx1)
gy = cv2.sepFilter2D(image, -1, sobely2, sobely1)

#Calculate the gradient magnitude
g = np.sqrt(gx * gx + gy * gy)

#Normalize output to fit the range 0-255
g *= 255.0 / np.max(g)

#End time
timeend = time.clock() - timestart
print("1D Convolution with Sobel Separable Filters: ", timeend)


#Display result

plt.figure('1D Convolution Gradient')
plt.imshow(g, cmap=plt.cm.gray)   
plt.xticks([]), plt.yticks([]) #rids axes

plt.figure('1D Convolution Gy')
plt.imshow(gy, cmap=plt.cm.gray)   
plt.xticks([]), plt.yticks([]) #rids axes

plt.figure('1D Convolution Gx')
plt.imshow(gx, cmap=plt.cm.gray)   
plt.xticks([]), plt.yticks([]) #rids axes

plt.show()