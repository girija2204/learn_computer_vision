import cv2
import numpy as np
import math
from utils import multiple_plots, multiple_images
from gaussianNoise import addGaussianNoiseToImg

image = cv2.imread("lenna.png")
image_copy = image.copy()

_, noisy_image = addGaussianNoiseToImg(image)
cv2.imshow("Noisy image",noisy_image)
cv2.waitKey(0)

# When we draw the line on the image, instead of doing it on a single channel, it draws the line in all the channels
cv2.line(image_copy,(0,110),(220,110),(255,255,255),2)
cv2.imshow("Line on the image",image_copy)
cv2.waitKey(0)

# Differentiation of the pixels of original image
diff_kernel = [1,2,0,-2,-1]
img_diff = np.convolve(image[110,:,0],diff_kernel)

# Differentiation of the pixels of the noisy image
nimg_diff = np.convolve(noisy_image[110,:,0],diff_kernel)

# Noisy image smoothening with gaussian kernel
gaussianKernel = [0.05448868,0.24420134,0.40261995,0.24420134,0.05448868] # cv2.getGaussianKernel(5,1)
sm_nimg = np.convolve(noisy_image[110,:,0],gaussianKernel)

# Take Differentiation after smoothening noisy image
sm_nimg_diff = np.convolve(sm_nimg,diff_kernel)

arr = [np.array([image[110,:,0],img_diff]),np.array([noisy_image[110,:,0],nimg_diff]),np.array([sm_nimg,sm_nimg_diff])]
names = [np.array(['(a) Original Image','(b) Differentiation of original image']),np.array(['(c) Noisy Image','(d) Differentiation of noisy image']),np.array(['(e) Smoothened noisy image','(f) Differentiation of smoothened noisy image'])]
multiple_plots(arr,names)

# # Note: The test has been done on a single row first, to see its effect. Then will be repeated on the whole image.
# # First a gaussian kernel is applied on the image to make it smoother. More kernel size makes it blurred. So try with different
# # kernel sizes. Then after blurring, differentiate the image with a differentiation kernel. Differentiation tells us the pixel values
# # where intensity of the image changes rapidly.
# # We can do both the operations in a single operation, by multiplying the 1D kernels to make a 2D kernel, and then convolve the image
# # with the 2D kernel. That also produces the same result.