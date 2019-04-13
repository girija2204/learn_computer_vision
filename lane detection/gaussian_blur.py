import cv2
import numpy as np
from utils import multiple_images
from gaussianNoise import addGaussianNoiseToImg

image = cv2.imread("lenna.png")
_,image = addGaussianNoiseToImg(image)

# To get a 1D gaussian kernel of size 5 and sigma = 1
print(cv2.getGaussianKernel(5,1))

# To get a 2D kernel, take the 1D kernels and then take a dot product of them
xkernel = cv2.getGaussianKernel(5,1)
ykernel = cv2.getGaussianKernel(5,1)
kernel2D = xkernel*ykernel.transpose()
print(kernel2D)

xkernel = cv2.getGaussianKernel(5,1)
ykernel = cv2.getGaussianKernel(7,2)

image_blurred_boxFilter = cv2.boxFilter(image,ddepth=-1,ksize=(5,5))
image_blurred_sepFilter = cv2.sepFilter2D(image,ddepth=-1,kernelX=xkernel,kernelY=ykernel)
image_blurred_GaussianBlur = cv2.GaussianBlur(image,(11,11),sigmaX=2,sigmaY=2)

images = [np.array([image,image_blurred_boxFilter,image_blurred_sepFilter,image_blurred_GaussianBlur])]
names = [np.array(['original image','boxFilter','sepFilter2D','GaussianBlur'])]
multiple_images(images,names)