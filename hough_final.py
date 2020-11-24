import numpy as np

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage import data
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage import filters
import math
import cv2
from skimage.io import imsave
font = cv2.FONT_HERSHEY_SIMPLEX

i = 0
j = 200
y = [0,0]
limites = [0,0]
parlimites = [0,0]
coorx = 0
coory = 0
pi = math.pi

code = str(20)
# Constructing test image
img = imread(code+'.png')
noisy_image = rgb2gray(img)
noisy_image1 = 0.4 <= noisy_image  
noisy_image2 = 0.7 >= noisy_image
noisy_image3 = 0.9 <= noisy_image
noisy_image2 = ~noisy_image2
noisy_image3 = ~noisy_image3
noisy_image = noisy_image1*noisy_image2*noisy_image3


image = filters.sobel(noisy_image)


# Classic straight-line Hough transform
# Set a precision of 0.5 degree.
tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
h, theta, d = hough_line(image, theta=tested_angles)
a,angle,dists = hough_line_peaks(h,theta,d)
maximo = a[1]

# Generating figure 1
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')
ax[0].set_axis_off()

ax[1].imshow(np.log(1 + h),
             extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
             cmap=cm.gray, aspect=1/1.5)
ax[1].set_title('Hough transform')
ax[1].set_xlabel('Angles (degrees)')
ax[1].set_ylabel('Distance (pixels)')
ax[1].axis('image')

ax[2].imshow(noisy_image, cmap=cm.gray)
x = np.array((0, image.shape[1]))

for accum, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    y0, y1 = (dist - x * np.cos(angle)) / np.sin(angle)
    if accum >= maximo:
        ax[2].plot(x, (y0,y1), '-r')
        m = (y1-y0)/(x[1]-x[0])
        y[i] = y0
        i = i+1
        angulo = angle
b = (y[1]+y[0])/2



while True:
    if angulo < 0.79 and angulo > -0.79:
        centro = (j-b)/m
        if noisy_image[j,int(centro)] == True:
            punto = [j,int(centro)]
            break
        else:
            j = j+1
    else:
        centro = m*j+b
        if noisy_image[int(centro),j] == True:
            print(1,j)
            punto = [int(centro),j]
            break
        else:
            j = j+1    



while True:
    if angulo < 0.79 and angulo > -0.79:
        centro = (j-b)/m
        if noisy_image[j,int(centro)] == True:
            j = j+1
            i = 0
        else:
            i = i+1
            j = j+1
            if i > 10 or j > 399:
                break
    else:
        centro = m*j+b
        if noisy_image[int(centro),j] == True:
            j = j+1
            i = 0
        else:
            i = i+1
            j = j+1
            if i > 10 or j > 399:
                break   

limites[0] = j
parlimites[0] = centro
j = punto[0]

while True:
    if angulo < 0.79 and angulo > -0.79:
        centro = (j-b)/m
        if noisy_image[j,int(centro)] == True:
            j = j-1
            i = 0
        else:
            i = i+1
            j = j-1
            if i > 10 or j < 1:
                break
    else:
        centro = m*j+b
        if noisy_image[int(centro),j] == True:
            j = j-1
            i = 0
        else:
            i = i+1
            j = j-1
            if i > 10 or j < 1:
                break  

limites[1] = j
parlimites[1] = centro




if angulo < 0.79 and angulo > -0.79:
    coorx = (parlimites[0]+parlimites[1])/2
    coory = (limites[0]+limites[1])/2
else:
    coory = (parlimites[0]+parlimites[1])/2
    coorx = (limites[0]+limites[1])/2

coorangle = (pi/2)-angulo
dosdeci = '%.3f' % coorangle


text = '('+str(int(coorx))+', '+str(int(coory))+')'
text2 = 'Angulo: '+dosdeci

cv2.putText(img,text,(0,30), font, 1,(255,0,0),2)
cv2.putText(img,text2,(0,65), font, 1,(255,0,0),2)
cv2.circle(img,(int(coorx), int(coory)), 5, (255,0,0), 2)
imsave(code+'.jpg',img)
ax[2].set_xlim(x)
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_axis_off()
ax[2].set_title('Detected lines')

plt.tight_layout()
plt.show()