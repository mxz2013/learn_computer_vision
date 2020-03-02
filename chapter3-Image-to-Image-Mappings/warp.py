import homography
from scipy.ndimage import filters
from numpy import *
from pylab import *
from PIL import Image
import csv
import os

def image_in_image(im1,im2, tp):
    """
    put im1 in im2 with an affine transformation
    such that corners are as colse to tp as possible.
    tp are homogenous and counter-clockwise from top left.
    """
    from scipy import ndimage
    ## points to warp from

    m, n = im1.shape[:2]
    print'm and n values:', m, n
    #fp = array([[m,0,0,m],[0,0,n,n],[1,1,1,1]])
    #y = [0, 0, m, m]
    #x = [0, n, n, 0]
    y = [0, m, m, 0]
    x = [0, 0, n, n]
    ## the sequence of the corners are:
    # left-above, left-below, right-below, right-above
    fp = array([y,x,[1,1,1,1]])

    #print'fp[:2]', fp[:2]
    
    # compute affine transform and apply
    H = homography.Haffine_from_points(tp,fp)
    im1_t = ndimage.affine_transform(im1, H[:2,:2],(H[0,2],H[1,2]),im2.shape[:2])
    alpha = (im1_t > 0)

    return (1-alpha)*im2 + alpha*im1_t

def alpha_for_triangle(points,m,n):
    """Creats alpha map of size (m,n) for a triangle with corners
    defined by poins (given in normalized homogeneous coordinates).
    """
    alpha = zeros((m,n))
    for i in range(min(points[0]),max(points[0])):
        for j in range(min(points[1]),max(points[1])):
            x = linalg.solve(points,[i,j,1])
            if min(x) > 0: # all coefficients positive
                alpha[i,j] = 1
    return alpha




## example of affine warp of im1 onto im2


im1 = array(Image.open('../images/beatles.jpg').convert('L'))
im2 = array(Image.open('../images/billboard_for_rent.jpg').convert('L'))
m1,n1=im1.shape[:2]
y1 = [0, 0, m1, m1]
x1 = [0, n1, n1, 0]
m2,n2=im2.shape[:2]
print'the m2 and n2 values:', m2, n2
#y2 = [0, 0, m2, m2]
#x2 = [0, n2, n2, 0]

y2 = [264, 538, 540, 264]
x2 = [40, 36, 605,605]
## the sequence of the corners are:
# left-above, left-below, right-below, right-above

# set to points
tp = array([y2,x2, [1, 1, 1, 1]])


#im3 = image_in_image(im1,im2,tp)
#
#figure()
#gray()
#imshow(im3)
#plot(x2,y2, 'r*')
#axis('equal')
#axis('off')
#show()
#


