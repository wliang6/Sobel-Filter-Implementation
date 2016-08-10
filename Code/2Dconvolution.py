#Sobel Filter implementation in 2D Convolution
#Author: Winnie Liang

import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

image = cv2.imread('lena_gray.jpg')

sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float)
sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = np.float)

#Start time
timestart = time.clock()

#Calculate gx and gy using Sobel (horizontal and vertical gradients)
gx = cv2.filter2D(image, -1, sobelx)
gy = cv2.filter2D(image, -1, sobely)
#Calculate the gradient magnitude
g = np.sqrt(gx * gx + gy * gy)

#Normalize output to fit the range 0-255
g *= 255.0 / np.max(g)

#End time
timeend = time.clock() - timestart
print("2D Convolution with Sobel Filters: ", timeend)


#Display results
plt.figure('2D Convolution Gradient')
plt.imshow(g, cmap=plt.cm.gray)   
plt.xticks([]), plt.yticks([]) #rids axes

plt.figure('2D Convolution Gy')
plt.imshow(gy, cmap=plt.cm.gray)   
plt.xticks([]), plt.yticks([]) #rids axes

plt.figure('2D Convolution Gx')
plt.imshow(gx, cmap=plt.cm.gray)   
plt.xticks([]), plt.yticks([]) #rids axes

plt.show()

