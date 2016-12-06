import cv2
import numpy as np
from matplotlib import pyplot as plt

nrows = 3
ncols = 1
#Import image
img = cv2.imread('GMIT1.jpg')

imgHarris = img
np.copyto(imgHarris,img)

imgShiTomasi = img
np.copyto(imgShiTomasi,img)

imgOrb = img
np.copyto(imgOrb,img)
#Image Treatment
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Write image to disk.
cv2.imwrite('gray_image.png',gray_image)

#Harris Corner Detection Images
#blockSize = 2
#aperture_size = 3
#k = 0.04
#dst = cv2.cornerHarris(gray_image, blockSize, aperture_size, k)
#deeepDst = dst.clone()

#threshold = 0.04; #number between 0 and 1
#for i in range(len(dst)):
#	for j in range(len(dst[i])):
#		if dst[i][j] > (threshold*dst.max()):
#			cv2.circle(imgHarris,(j,i),3,(255, 255, 0),-1)

#Shi Tomasi
corners = cv2.goodFeaturesToTrack(gray_image,50,0.01,10)

for i in corners:
	x,y = i.ravel()
	cv2.circle(imgShiTomasi,(x,y),3,(0, 0, 255),-1)


# Initiate ORB-SIFT detector
orb = cv2.ORB()
# find the keypoints and descriptors with ORB-SIFT

kp, des1 = orb.detectAndCompute(gray_image,None)
# draw only keypoints location,not size and orientation
imgOrb = cv2.drawKeypoints(imgOrb,kp,color=(0, 0, 255))

#Image Display
#Add images to plot
#
#plt.subplot(nrows, ncols,2),plt.imshow(cv2.cvtColor(imgHarris,cv2.COLOR_BGR2RGB), cmap = 'gray')
#plt.title('Harris'), plt.xticks([]), plt.yticks([])
#
plt.subplot(nrows, ncols,1),plt.imshow(cv2.cvtColor(imgShiTomasi,cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Shi Tomasi'), plt.xticks([]), plt.yticks([])
#
plt.subplot(nrows, ncols,2),plt.imshow(cv2.cvtColor(imgOrb,cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Orb Image'), plt.xticks([]), plt.yticks([])

#Show Images
plt.show()