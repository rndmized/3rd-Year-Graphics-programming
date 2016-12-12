import cv2
import numpy as np
from matplotlib import pyplot as plt

nrows = 1
ncols = 3

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread('cage2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

img2 = cv2.imread("cage.jpg")

b = img2
b = np.copy(img2)
b[:,:,2] = 0
b[:,:,1] = 0

g = img2
g = np.copy(img2)
g[:,:,0] = 0
g[:,:,2] = 0

r = img2
r = np.copy(img2)
r[:,:,0] = 0
r[:,:,1] = 0

#Image Display
#Add images to plot
plt.subplot(nrows, ncols,1),plt.imshow(cv2.cvtColor(b,cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Blue'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2),plt.imshow(cv2.cvtColor(g,cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Green'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,3),plt.imshow(cv2.cvtColor(r,cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Red'), plt.xticks([]), plt.yticks([])
#Show Images
plt.show()