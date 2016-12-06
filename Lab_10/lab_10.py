import cv2
import numpy as np
from matplotlib import pyplot as plt

nrows = 2
ncols = 4

#Import image
img = cv2.imread('GMIT.jpg')

#Image Treatment
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Write image to disk.
cv2.imwrite('gray_image.png',gray_image)
#Blor Images
blur3 = cv2.GaussianBlur(gray_image,(3, 3),0)
blur13 = cv2.GaussianBlur(gray_image,(13, 13),0)
#Apply sobel filter
sobelHorizontal = cv2.Sobel(gray_image,cv2.CV_64F,1,0,ksize=5) # x dir
sobelVertical = cv2.Sobel(gray_image,cv2.CV_64F,0,1,ksize=5) # y dir
#Sobel Addition (Horizontal+Vertical)
sobel = sobelVertical + sobelHorizontal
#Canny Filter
canny = cv2.Canny(gray_image,100,200)



#Image Display

#Add images to plot
#
plt.subplot(nrows, ncols,1),plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
#
plt.subplot(nrows, ncols,2),plt.imshow(gray_image, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])
#
plt.subplot(nrows, ncols,3),plt.imshow(blur3, cmap = 'gray')
plt.title('Blur 3x3'), plt.xticks([]), plt.yticks([])
#
plt.subplot(nrows, ncols,4),plt.imshow(blur13, cmap = 'gray')
plt.title('Blur 13x13'), plt.xticks([]), plt.yticks([])
#
plt.subplot(nrows, ncols,5),plt.imshow(sobelHorizontal, cmap = 'gray')
plt.title('Sobel Horizontal'), plt.xticks([]), plt.yticks([])
#
plt.subplot(nrows, ncols,6),plt.imshow(sobelVertical, cmap = 'gray')
plt.title('Sobel Vertical'), plt.xticks([]), plt.yticks([])
#
plt.subplot(nrows, ncols,5),plt.imshow(sobel, cmap = 'gray')
plt.title('Sobel++'), plt.xticks([]), plt.yticks([])
#
plt.subplot(nrows, ncols,6),plt.imshow(canny, cmap = 'gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])

#Show Images
plt.show()