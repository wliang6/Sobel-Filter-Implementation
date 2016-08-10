#Sobel Separable Filter implementation in 1D Convolution 
#Author: Winnie Liang

import numpy as np
import cv2
import time

image = cv2.imread('lena_gray.jpg', 0)

#Sobel Separable Filter in x direction
sobelx1 = np.array([1,2,1])[:,None] #column
sobelx2 = np.array([-1,0,1])[None,:] #row

#Sobel Separable Filter in y direction
sobely1 = np.array([-1,0,1])[:,None] #column
sobely2 = np.array([1,2,1])[None,:] #row


N = image.shape[0] #row
M = image.shape[1] #column


row = 512
column = 512
sobelxImage = np.zeros((row,column))
sobelyImage = np.zeros((row,column))
sobelGrad2 = np.zeros((row,column))


#Start time
timestart = time.clock()

## Usage of CV2 Library for sobel separable filter
# gx = cv2.sepFilter2D(image, -1, sobelx2, sobelx1)
# gy = cv2.sepFilter2D(image, -1, sobely2, sobely1)

#Calculate gx and gy by convolution using first 1D Sobel separable filter
for i in range(1,N-1):
    for j in range(1,M-1):
        gx = sobelx1[0][0] * image[i-1][j] + \
        	sobelx1[1][0] * image[i][j] + \
        	sobelx1[2][0] * image[i+1][j]
        gy = sobely1[0][0] * image[i-1][j] + \
        	sobely1[1][0] * image[i][j] + \
        	sobely1[2][0] * image[i+1][j]
        
        sobelxImage[i-1][j-1] = gx
        sobelyImage[i-1][j-1] = gy

#Surrounds array with 0's on the outside perimeter
sobelxImagePad = np.pad(sobelxImage, (1,1), 'edge')
sobelyImagePad = np.pad(sobelyImage, (1,1), 'edge')

#Calculate gx and gy by convolution using second 1D Sobel separable filter
for i in range(1,sobelxImagePad.shape[0]-1):
    for j in range(1,sobelxImagePad.shape[1]-1):
        gx = sobelx2[0][0] * sobelxImagePad[i][j-1] + \
        	sobelx2[0][1] * sobelxImagePad[i][j] + \
        	sobelx2[0][2] * sobelxImagePad[i][j+1]
        gy = sobely2[0][0] * sobelyImagePad[i][j-1] + \
        	sobely2[0][1] * sobelyImagePad[i][j] + \
        	sobely2[0][2] * sobelyImagePad[i][j+1]
        
        sobelxImage[i-1][j-1] = gx
        sobelyImage[i-1][j-1] = gy

        #Calculate the gradient magnitude
        g = np.sqrt(gx * gx + gy * gy)
        sobelGrad2[i-1][j-1] = g

#End time
timeend = time.clock() - timestart
print("1D Convolution with Sobel Separable Filters: ", timeend)
                
cv2.imwrite('custom_1d_convolution_gx.png', sobelxImage) 
cv2.imwrite('custom_1d_convolution_gy.png', sobelyImage)
cv2.imwrite('custom_1d_convolution_gradient.png', sobelGrad2)