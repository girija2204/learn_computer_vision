import cv2
import numpy as np
import math
from utils import multiple_plots, multiple_images
from gaussianNoise import addGaussianNoiseToImg
import matplotlib.pyplot as plt
from skimage import img_as_float, img_as_ubyte
from scipy import signal

image = cv2.imread("image-03.jpg")
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#_,image = addGaussianNoiseToImg(image)
image_copy = image.copy()

#Prewitt Filter in x direction
kxx,kyx = cv2.getDerivKernels(1,0,3)
kyx[1][0]=1.0

#Prewitt Filter in y direction
kxy,kyy = cv2.getDerivKernels(0,1,3)
kxy[1][0]=1.0

# np.convolve flips the kernel array in x direction and then correlate it. To flip it uses, v[::-1], v is the kernel array.
img_gd_x = cv2.sepFilter2D(image,ddepth=-1,kernelX=kxx,kernelY=kyx)
img_gd_x = cv2.convertScaleAbs(img_gd_x)

img_gd_y = cv2.sepFilter2D(image,ddepth=-1,kernelX=kxy,kernelY=kyy)
img_gd_y = cv2.convertScaleAbs(img_gd_y)

img_gd_xy = cv2.addWeighted(img_gd_x,0.5,img_gd_y,0.5,0)

sobelled_x = cv2.Sobel(image,ddepth=-1,dx=1,dy=0,ksize=3)
sx = cv2.convertScaleAbs(sobelled_x)

sobelled_y = cv2.Sobel(image,ddepth=-1,dx=0,dy=1,ksize=3)
sy = cv2.convertScaleAbs(sobelled_y)

sxy = cv2.addWeighted( sx, 0.5, sy, 0.5, 0)

images1 = [np.array([image,img_gd_x,img_gd_y,img_gd_xy]),np.array([image,sobelled_x,sobelled_y,sxy])]
names1 = [np.array(['Original Image','Vertical edges using Prewitt','Horizontal edges using Prewitt','Image derivative using Prewitt']),np.array(['Original Image','Vertical edges using Sobel','Horizontal edges using Sobel','Image derivative using Sobel'])]
#multiple_images(images1,names1)

# ---------------------------------------------- #
k12d,k22d = cv2.getDerivKernels(2,0,3)

k1y2d,k2y2d = cv2.getDerivKernels(0,2,3)

lapx = cv2.sepFilter2D(image,ddepth=-1,kernelX=k12d,kernelY=k22d)
lx = cv2.convertScaleAbs(lapx)

lapy = cv2.sepFilter2D(image,ddepth=-1,kernelX=k1y2d,kernelY=k2y2d)
ly = cv2.convertScaleAbs(lapy)

lxy = cv2.addWeighted(lx,0.5,ly,0.5,0)

laplaced = cv2.Laplacian(image,ddepth=-1,ksize=3)

multiple_images([np.array([lx,ly,lxy,laplaced])],[np.array(['My Laplacianx','My Laplacian Y','My Laplacian XY','Laplacian'])])