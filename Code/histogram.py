#Histogram Equalization
#Author: Winnie Liang

import cv2
import numpy as np 
import Image
from matplotlib import pyplot as plt 

original = cv2.imread('rose.jpg')
#Read our input color image and converts to grayscale
image = cv2.imread('rose.jpg', 0) 
# #Convert color image to gray image.
# grayImage = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
# #Save gray image to file named rose.jpg
# cv2.imwrite('grayrose.jpg', grayImage)

# #Read the saved gray image
# grayImg = cv2.imread('grayrose.jpg')
#Make a copy
grayCopy = image


#For an N x M image of G gray-levels (often 256), create an array H of length G initialized with 0 values
N = image.shape[0] #row 
M = image.shape[1] #column

H = np.zeros(256) # image histogram
C = np.zeros(256) # cumulative histogram
T = np.zeros(256) # transformation function
E = np.zeros(256) # equalized histogram


#Calculate Histogram H by scanning every pixel and increment member of H
for p in range(0, N):
	for q in range(0, M):
		H[image[p][q]] += 1  

#Calculate Cumulative histogram C
C[0] = H[0]
for p in range(1, 256):
	C[p] = C[p-1] + H[p]

#Create lookup table T 
value = (256.0 - 1)/(N * M)
for p in range(0, 256):
	T[p] = round(value * C[p])

#Remap pixel intensities of original image 
for p in range(0, N):
	for q in range(0, M):
		grayCopy[p][q] = T[grayCopy[p][q]]

#Calculate Equalized histogram E
for p in range(0, N):
	for q in range(0, M):
		E[grayCopy[p][q]] += 1




#Display results on matplotlib

#Display original and enhanced images
plt.figure('Images')

plt.subplot(131).imshow(original, cmap = plt.cm.gray)
plt.xticks([]), plt.yticks([]) #rids axes
plt.title('Original Image')

plt.subplot(132).imshow(image, cmap = plt.cm.gray)
plt.xticks([]), plt.yticks([]) #rids axes
plt.title('Grayscale Image')

plt.subplot(133).imshow(grayCopy, cmap=plt.cm.gray)
plt.xticks([]), plt.yticks([]) #rids axes
plt.title('Equalized / Enhanced Image')

#Display histograms H, C, T, E
plt.figure('Histograms')
plt.subplot(221).plot(H)
plt.xlabel('Intensity Value')
plt.ylabel('# of Pixels')
plt.title('Gray Image Histogram')

plt.subplot(222).plot(C)
plt.xlabel('Intensity Value')
plt.ylabel('# of Pixels')
plt.title('Cumulative Histogram')

plt.subplot(223).plot(T)
plt.xlabel('Original intensity value')
plt.ylabel('New intensity value')
plt.title('Transformation Function')

plt.subplot(224).plot(E)
plt.xlabel('Intensity value')
plt.ylabel('# of Pixels')
plt.title('Equalized Histogram')




plt.show()





