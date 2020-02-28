#!/home/sky/anaconda3/envs/python2.7

from scipy.ndimage import filters
from numpy import *
from pylab import *
from PIL import Image
import csv
import sift
import os

#imname = '../images/empire.jpg'
#im1 = array(Image.open(imname).convert('L'))
#sift.process_image(imname,'empire.sift')
#l1,d1 = sift.read_features_from_file('empire.sift')
#figure()
#gray()
#sift.plot_features(im1,l1,circle=True)
#show()



wid = 5


imname1 = 'fig1.png'
imname2 = 'fig2.png'
im1 = array(Image.open(imname1).convert('L'))
im2 = array(Image.open(imname2).convert('L'))
sift.process_image(imname1,'fig1.sift')
l1,d1 = sift.read_features_from_file('fig1.sift')

#figure()
#gray()
#sift.plot_features(im1,l1,circle=True)

sift.process_image(imname2,'fig2.sift')
l2,d2 = sift.read_features_from_file('fig2.sift')
print 'starting matching...'
#sift.plot_features(im2,l2,circle=True)


matches = sift.match_twosided(d1,d2)

figure()
gray()
sift.plot_matches(im1,im2,l1,l2,matches)
show()