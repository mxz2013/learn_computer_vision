import homography
from scipy import ndimage
import warp
from scipy.ndimage import filters
from numpy import *
from pylab import *
from PIL import Image
import csv
import os

## set from points to corners of im1

im1 = array(Image.open('../images/beatles.jpg').convert('L'))
im2 = array(Image.open('../images/billboard_high_resolution.jpg').convert('L'))

m,n = im1.shape[:2]
#m2,n2 = im2.shape[:2]

y1 = [0,m,m,0]
x1 = [0,0,n,n]

y2 = [92,338,338,92]
x2 = [116,110, 738,732]

## the sequence of the corners are:
# left-above, left-below, right-below, right-above
fp = array([y1,x1,[1,1,1,1]])
tp = array([y2,x2,[1,1,1,1]])

# first triangle
tp2 = tp[:,:3]
fp2 = fp[:,:3]
print 'fp2 \n', fp2, fp2[1], fp2[0]

# compute H
H = homography.Haffine_from_points(tp2,fp2)
im1_t = ndimage.affine_transform(im1,H[:2,:2],(H[0,2],H[1,2]),im2.shape[:2])


# alpha for triangle
alpha = warp.alpha_for_triangle(tp2,im2.shape[0],im2.shape[1])
im3 = (1-alpha)*im2 + alpha*im1_t


# second triangle

tp2_2nd = tp[:,[0,2,3]]
fp2_2nd = fp[:,[0,2,3]]
print 'second fp2', fp2_2nd

# compute H
H = homography.Haffine_from_points(tp2_2nd,fp2_2nd)
im1_t = ndimage.affine_transform(im1,H[:2,:2],(H[0,2],H[1,2]),im2.shape[:2])


# alpha for triangle
alpha = warp.alpha_for_triangle(tp2_2nd,im2.shape[0],im2.shape[1])
im4 = (1-alpha)*im3 + alpha*im1_t

figure()
gray()
imshow(im4)
#plot(tp2[1],tp2[0], 'r*')
#plot(tp2_2nd[1], tp2_2nd[0], 'gD')
#plot(fp2[1],fp2[0], 'r*')
#plot(fp2_2nd[1], fp2_2nd[0], 'gD')
axis('equal')
axis('off')
show()

