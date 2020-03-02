from scipy.ndimage import filters
from numpy import *
from pylab import *
from PIL import Image
import csv
import os



def normalize(points):
    """normalize a collection of points in
    homogeneous coordinates so that the last row = 1."""

    for row in points:
        row /= points[-1]
    return points



def make_homog(points):
    """
    convert a set of poins (dim*n array) to homogeneous coordinates
    """
    return vstack((points, ones(1, points.shape[1])))


def H_from_points(fp,tp): 
    """
    Find homography H, such that fp is mapped to tp using the 
    linear direct linear transformation (DLT) method.
    Points are conditioned automatically.
    The points are conditioned by normalizing so that they have zero mean and unit
    standard deviation. This is very important for numerical reasons since the stability
    of the algorithm is dependent of the coordinate representation. Then the matrix A is
    created using the point correspondences. The least squares solution is found as the
    last row of the matrix V of the SVD. The row is reshaped to create H . This matrix is
    then de-conditioned and normalized before returned.
    """

    if fp.shap != tp.shape:
        raise RuntimeError('number of points do not match')

    # condition points (important for numerical reasons)
    # normalizing so that they have zero mean and unit standard deviation
    # --from points--

    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 =  diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp = dot(C1,fp)

    ##--to points--
    m = mean(tp[:2], axis=1)
    maxstd = max(std(tp[:2], axis=1)) + 1e-9
    C2 =  diag([1/maxstd, 1/maxstd, 1])
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp = dot(C2,tp)

    ## create matrix for linear method, 2 rows for each corresponding pair
    nbr_correspondences = fp.shape[1]
    A = zeros((2*nbr_correspondences , 9))
    for i in range(nbr_correspondences):
        A[2*i] = [-fp[0][i],-fp[1][i],-1,0,0,0,tp[0][i]*fp[0][i],tp[0][i]*fp[1][i],tp[0][i]]
        A[2*i+1] = [0,0,0,-fp[0][i],-fp[1][i],-1,tp[1][i]*fp[0][i],tp[1][i]*fp[1][i],tp[1][i]]

    U,S,V = linalg.svd(A)
    # dimansions: A=2n*9, U=2n*2n, S=2n*9, V=9*9 so that A=USV
    H = V[8].reshape((3,3))
    ## decondition
    H = dot(linalg.inv(C2), dot(H,C1))

    # normalize and return
    return H / H[2,2]

def Haffine_from_points(fp,tp):
    """ Find H, affine transformation, such that tp is affine transform of
    fp"""

    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    # condition points (important for numerical reasons)
    # normalizing so that they have zero mean and unit standard deviation
    # --from points--
    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 =  diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp_cond = dot(C1,fp)

    ##--to points--
    m = mean(tp[:2], axis=1)
    C2 = C1.copy() ## much use the same scaling for both sets
    #C2 =  diag([1/maxstd, 1/maxstd, 1])
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp_cond = dot(C2,tp)
    # conditioned points have mean zero, so translation is zero
    A = concatenate((fp_cond[:2],tp_cond[:2]), axis=0)
    U,S,V = linalg.svd(A.T)
    # dimansions: A=2n*9, U=2n*2n, S=2n*9, V=9*9 so that A=USV

    # create B and C matrices as Hartley-Zisserman p130
    ## R. I. Hartley and A. Zisserman. Multiple View Geometry in Computer Vision.
    #Cambridge University Press, ISBN: 0521540518, second edition, 2004.
    tmp = V[:2].T
    B =  tmp[:2]
    C = tmp[2:4]

    tmp2 = concatenate((dot(C,linalg.pinv(B)),zeros((2,1))), axis=1)
    H = vstack((tmp2, [0,0,1]))

    # decondition
    H = dot(linalg.inv(C2), dot(H,C1))

    # normalize and return
    return H / H[2,2]


#from scipy import ndimage
#
#im = array(Image.open('../images/empire.jpg').convert('L'))
#H = array([[1.4,0.05,-100],[0.05,1.5,-100],[0,0,1]])
#im2 = ndimage.affine_transform(im,H[:2,:2],(H[0,2],H[1,2]))
#figure()
#gray()
#imshow(im2)
#show()



    

