#!/home/sky/anaconda3/envs/python2.7

from scipy.ndimage import filters
from numpy import *
from pylab import *
from PIL import Image
import csv

def compute_harris_response(im, sigma=3):
    """Compute the Harris corner detector response function
    for each pixel in a graylevel image.
    """

    ## Derivatives
    imx = zeros(im.shape)
    #print 'the image dimansion is:'. im.shape
    filters.gaussian_filter(im, (sigma,sigma), (0,1), imx)
    imy = zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (1,0), imy)

    # compute components of the Harris matrix
    Wxx = filters.gaussian_filter(imx*imx, sigma)
    Wxy = filters.gaussian_filter(imx*imy, sigma)
    Wyy = filters.gaussian_filter(imy*imy, sigma)

    # determinant and trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy

    return Wdet / Wtr


def get_harris_points(harrisim, min_dist=10, threshold=0.1):
    """
    return corners from a Harris response image
    min_dist is the minimum number of pixels separating corners
    and image boundary.
    """
    # find top corner candidates above a threshold

    corner_threshold = harrisim.max() * threshold
    print'the corner_threshold is:',corner_threshold
    harrisim_t = (harrisim > corner_threshold) * 1
    print'the harrisim_t is:', harrisim_t.shape

    # get coordinates of candiages
    coords = array(harrisim_t.nonzero()).T
    #print'the coords is:', coords[:10]

    # get their values in a list
    candidate_values = [harrisim[c[0],c[1]] for c in coords]


    # sort candiate
    index = argsort(candidate_values)

    #########check my code s ###
    coords_sky = array(np.where(harrisim>corner_threshold)).T
    candidate_values_sky = harrisim[harrisim > corner_threshold]
    index_sky = argsort(candidate_values_sky)
    print 'check my code', np.array_equal(index, index_sky),\
                           np.array_equal(coords, coords_sky)
    #########check my code f ############
    # store allowed point locations in array

    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1

    # select the best points taking min_distance into account
    filtered_coords = []
    allowed_locations_size = []
    for i in index:  #[:10]:
        #print 'the allowed_locations', np.size(allowed_locations.nonzero())
        #allowed_locations_size.append(np.size(allowed_locations.nonzero()))
        if allowed_locations[coords[i,0],coords[i,1]] == 1:
            filtered_coords.append(coords[i])
            ## remove locations that close to the filtered ones.
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),
                             (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0

   # with open('allowed_locations_size', 'w') as f:
   #     writer = csv.writer(f, delimiter='\t')
   #     writer.writerows(zip(allowed_locations_size))
   #     #for i in allowed_locations_size:
   #     #    f.write(i)
#    print"filtered_coords",  array(filtered_coords).T[0,:]
#    for p in filtered_coords:
#        print p[1], p[0]
    #figure()
    #plot(allowed_locations_size)
    #show()
    return filtered_coords




def get_descriptors(image,filtered_coords,wid=5):
    """ For each point return pixel values around the point
    using a neighbourhood of width 2*wid+1. (Assume points are
    extracted with min_distance > wid). """
    desc = []
    for coords in filtered_coords:
        patch = image[coords[0]-wid:coords[0]+wid+1,
        coords[1]-wid:coords[1]+wid+1].flatten()
        desc.append(patch)
    return desc

def match(desc1,desc2,threshold=0.5):
    """ For each corner point descriptor in the first image,
    select its match to second image using
    normalized cross correlation.
    """
    n = len(desc1[0])
    # pair-wise distances
    d = -ones((len(desc1),len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - mean(desc1[i])) / std(desc1[i])
            d2 = (desc2[j] - mean(desc2[j])) / std(desc2[j])
            ncc_value = sum(d1 * d2) / (n-1)
            if ncc_value > threshold:
                d[i,j] = ncc_value
    ndx = argsort(-d)
    matchscores = ndx[:,0]
    return matchscores


def match_twosided(desc1,desc2,threshold=0.5):
    """ Two-sided symmetric version of match(). """
    matches_12 = match(desc1,desc2,threshold)
    matches_21 = match(desc2,desc1,threshold)
    ndx_12 = where(matches_12 >= 0)[0]
    # remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1
    return matches_12


def appendimages(im1,im2):
    """ Return a new image that appends the two images side-by-side. """
    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]
    if rows1 < rows2:
        im1 = concatenate((im1,zeros((rows2-rows1,im1.shape[1]))),axis=0)
    elif rows1 > rows2:
        im2 = concatenate((im2,zeros((rows1-rows2,im2.shape[1]))),axis=0)
    # if none of these cases they are equal, no filling needed.
    return concatenate((im1,im2), axis=1)


def plot_matches(im1,im2,locs1,locs2,matchscores,show_below=True):
    """ Show a figure with lines joining the accepted matches
    input: im1,im2 (images as arrays), locs1,locs2 (feature locations),
    matchscores (as output from 'match()'),
    show_below (if images should be shown below matches). """
    im3 = appendimages(im1,im2)
    if show_below:
        im3 = vstack((im3,im3))
    imshow(im3)
    #print'the shape of locs1', locs1[:2,:2]
    cols1 = im1.shape[1]
    for i,m in enumerate(matchscores):
        if m>0:
            plot([locs1[i][1],locs2[m][1]+cols1],[locs1[i][0],locs2[m][0]],'c')
    axis('off')



wid = 5
im1 = array(Image.open('fig1.png').convert('L'))
im2 = array(Image.open('fig2.png').convert('L'))
harrisim = compute_harris_response(im1,5)
filtered_coords1 = get_harris_points(harrisim,wid+1)
d1 = get_descriptors(im1,filtered_coords1,wid)
harrisim = compute_harris_response(im2,5)
filtered_coords2 = get_harris_points(harrisim,wid+1)
d2 = get_descriptors(im2,filtered_coords2,wid)
print 'starting matching...'
matches = match_twosided(d1,d2)
figure()
gray()
plot_matches(im1,im2,filtered_coords1,filtered_coords2,matches)
show()




def plot_harris_points(image,filtered_coords):
    """
    plots corners found in image.
    """
    figure()
    gray()
    imshow(image)
    plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '*')
    plot([p[0] for p in filtered_coords], [p[1] for p in filtered_coords], 'rD')
    axis('off')
    show()

#im = array(Image.open('../images/empire.jpg').convert('L'))
#harrisim = compute_harris_response(im)
#filtered_coords = get_harris_points(harrisim, 6)
#plot_harris_points(im, filtered_coords)
#plot_harris_points(harrisim, filtered_coords)

