#Sobel Filter implementation in 2D Convolution
#Author: Winnie Liang

import cv2
import numpy as np
import time
#from matplotlib import pyplot as plt

image = cv2.imread('lena_gray.jpg', 0)

sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float)
sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = np.float)


N = image.shape[0] #row 
M = image.shape[1] #column


row = 512
column = 512
sobelxImage = np.zeros((row,column))
sobelyImage = np.zeros((row,column))
sobelGrad = np.zeros((row,column))

#Start time
timestart = time.clock()

## Usage of CV2 Library for sobel
# gx = cv2.filter2D(image, -1, sobelx)
# gy = cv2.filter2D(image, -1, sobely)

#Surrounds array with 0's on the outside perimeter
image = np.pad(image, (1,1), 'edge')

for i in range(1, N-1):
    for j in range(1, M-1):        
        #Calculate gx and gy using Sobel (horizontal and vertical gradients)
        gx = (sobelx[0][0] * image[i-1][j-1]) + (sobelx[0][1] * image[i-1][j]) + \
             (sobelx[0][2] * image[i-1][j+1]) + (sobelx[1][0] * image[i][j-1]) + \
             (sobelx[1][1] * image[i][j]) + (sobelx[1][2] * image[i][j+1]) + \
             (sobelx[2][0] * image[i+1][j-1]) + (sobelx[2][1] * image[i+1][j]) + \
             (sobelx[2][2] * image[i+1][j+1])

        gy = (sobely[0][0] * image[i-1][j-1]) + (sobely[0][1] * image[i-1][j]) + \
             (sobely[0][2] * image[i-1][j+1]) + (sobely[1][0] * image[i][j-1]) + \
             (sobely[1][1] * image[i][j]) + (sobely[1][2] * image[i][j+1]) + \
             (sobely[2][0] * image[i+1][j-1]) + (sobely[2][1] * image[i+1][j]) + \
             (sobely[2][2] * image[i+1][j+1])     

        sobelxImage[i-1][j-1] = gx
        sobelyImage[i-1][j-1] = gy

        #Calculate the gradient magnitude
        g = np.sqrt(gx * gx + gy * gy)
        sobelGrad[i-1][j-1] = g

#End time
timeend = time.clock() - timestart
print("2D Convolution with Sobel Filters: ", timeend)


cv2.imwrite('custom_2d_convolution_gx.png',sobelxImage) 
cv2.imwrite('custom_2d_convolution_gy.png',sobelyImage)
cv2.imwrite('custom_2d_convolution_gradient.png',sobelGrad)



