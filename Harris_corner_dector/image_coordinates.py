from PIL import Image
from pylab import *
import numpy as np



#def black_and_white(input_image_path,
#    output_image_path):
#   color_image = Image.open(input_image_path)
#   bw = color_image.convert('L')
#   bw.save(output_image_path)
##if __name__ == '__main__':
##    black_and_white('caterpillar.jpg',
##        'bw_caterpillar.jpg')
#
#black_and_white('../images/empire.jpg','../images/empire_gray.jpg')

# read image to array
im = Image.open('../images/empire.jpg')
im_gray = im.convert('L')
im.save('empire.jpg')
im_gray.save('empire_gray.jpg')

#im1 = array(Image.open('../images/empire.jpg').convert('1'))
# plot the image
print'the size of the image', np.array(im).shape
print'the size of the image', np.array(im_gray).shape
imshow(im_gray)
# some points
x = [100,100,400,400]
y = [200,500,200,500]
# plot the points with red star-markers
plot(x,y,'r*')
#plot(im_gray[:,0], 'r+', markersize=10)
#plot(im[500,0,0], 'k+', markersize=10)
# line plot connecting the first two points
#plot(x[:2],y[:2])
# add title and show the plot
show()
