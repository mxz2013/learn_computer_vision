import homography
from scipy.ndimage import filters
from numpy import *
from pylab import *
from PIL import Image
from scipy import ndimage
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
    defined by points (given in normalized homogeneous coordinates).
    """
    alpha = zeros((m,n))

    for i in range(int(min(points[0])),int(max(points[0]))):
        for j in range(int(min(points[1])),int(max(points[1]))):
            x = linalg.solve(points,[i,j,1])
            if min(x) > 0: # all coefficients positive
                alpha[i,j] = 1
    return alpha


def triangulate_points(x,y):
    """Delaunay trigangulation of 2D points"""

    import matplotlib.delaunay as md
    centers,edges,tri,neighbors = md.delaunay(x,y)
    return tri


def pw_affine(fromim,toim,fp,tp,tri):
    """Warp triangular pathes from an image
    fromim = image to warp
    toim = destination image
    fp = from points in homo. coordinates
    tp = to points in homo. coordinates
    tri = triangulation.
    """
    im = toim.copy()
    #check if image is grayscale or color
    is_color = len(fromim.shape) == 3

    # create image to warp to (needed if iterate colors)
    im_t = zeros(im.shape, 'uint8')
    for t in tri:
        #compute affine transformation
        H = homography.Haffine_from_points(tp[:,t],fp[:,t])

        if is_color:
            for col in range(fromim.shape[2]):
                im_t[:,:,col] = ndimage.affine_transform(fromim[:,:,col], H[:2,:2],\
                                                         (H[0,2],H[1,2]),im.shape[:2])

                #alpha for triangle
                alpha = alpha_for_triangle(tp[:,t],im.shape[0],im.shape[1])

                #add triangel to image
                im[alpha>0] = im_t[alpha>0]

    return im

plot_piece_wise_affine_warping = True
if plot_piece_wise_affine_warping:
    def plot_mesh(x,y,tri):
        """plot triangles."""
        for t in tri:
            t_ext = [t[0], t[1], t[2], t[0]]
            plot(x[t_ext],y[t_ext],'r')
    # open image to warp
    fromim = array(Image.open('../images/sunset_tree.jpg'))
    n = 5
    m = 6
    ## below 4 lines of the code makes a n-1 x m-1 grid
    # put n equal distance points in x starting from x[0] to x[-1]
    # and m equal distance points in y stating from y[0] to y[-1]
    x,y = meshgrid(range(n), range(m))
    x = (fromim.shape[1]/(n-1)) * x.flatten()
    #print"the x values", x
    y = (fromim.shape[0]/(m-1)) * y.flatten()

    # triangulate
    tri = triangulate_points(x,y)

    # open image and destination points
    im = array(Image.open('../images/turningtorso1.jpg'))
    ## how to find these points in  turningtorso1_points.txt ??
    ## by reading the coordinates in the ploting?

    tp = loadtxt('turningtorso1_points.txt')
    tmp_tp = tp.copy()
    #print'tmp_tp', tmp_tp[:,0]
    # convert points to homo. coordinates
    fp = vstack((y,x,ones((1,len(x)))))
    tp = vstack((tp[:,1],tp[:,0],ones((1,len(tp)))))
    #print'tp', tp[:,1]

    # warp triangeles
    im_final = pw_affine(fromim,im,fp,tp,tri)

    # plot
    figure()
    imshow(im_final)
    plot_mesh(tp[1],tp[0],tri)
    plot(tmp_tp[:,0], tmp_tp[:,1], 'bD')
    axis('off')
    show()



plot_test_triangulate_points = False # True
if plot_test_triangulate_points:
    from numpy import random
    x,y = array(random.standard_normal((2,10)))
    print 'x and y', x,y
    tri = triangulate_points(x,y)
    print 'tri', tri
    figure()
    for t in tri:
        t_ext = [t[0], t[1], t[2], t[0]] # add first point to the end
        plot(x[t_ext],y[t_ext],'r')  # form a closed triangular p1-p2-p3-p1.
    plot(x,y,'*')
    axis('off')
    show()

plot_affine_warp = False ## True
if plot_affine_warp:
    ## example of affine warp of im1 onto im2


    im1 = array(Image.open('../images/beatles.jpg').convert('L'))
    im2 = array(Image.open('../images/billboard_for_rent.jpg').convert('L'))
    m1,n1=im1.shape[:2]
    y1 = [0, 0, m1, m1]
    x1 = [0, n1, n1, 0]
    m2,n2=im2.shape[:2]
    print'the m2 and n2 values:', m2, n2

    y2 = [264, 538, 540, 264]
    x2 = [40, 36, 605,605]
    ## the sequence of the corners are:
    # left-above, left-below, right-below, right-above

    # set to points
    tp = array([y2,x2, [1, 1, 1, 1]])

    im3 = image_in_image(im1,im2,tp)
    
    figure()
    gray()
    imshow(im3)
    plot(x2,y2, 'r*')
    axis('equal')
    axis('off')
    show()
    #


