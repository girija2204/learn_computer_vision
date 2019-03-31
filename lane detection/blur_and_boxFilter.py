import cv2
import numpy as np
from utils import plot_multiple

image = cv2.imread("lenna.png")

mat = np.random.randint(0,255,(50,50,3),dtype="uint8")
kernel = np.full((7,7),1/49)
x_flipped = np.flip(kernel,1)
yx_flipped = np.flip(x_flipped,0)


image_blurred_blur = cv2.blur(image,ksize=(3,3),anchor=(-1,-1))
image_blurred_box_3 = cv2.boxFilter(image,ddepth=-1,ksize=(3,3))
image_blurred_box_5 = cv2.boxFilter(image,ddepth=-1,ksize=(5,5))
image_blurred_2D = cv2.filter2D(image,-1,yx_flipped)

plot_multiple(np.array([image,image_blurred_blur,image_blurred_box_5,image_blurred_2D]),np.array(['Original Image', 'Image with cv2.blur()','Image with cv2.boxFilter()','Image with cv2.filter2D()']))

# To check if the resultant matrices are equal
print(np.array_equal(image_blurred_blur,image_blurred_box_3))
# print(np.array_equal(image_blurred_2D,image_blurred_box))