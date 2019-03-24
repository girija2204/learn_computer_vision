import cv2
import numpy as np
import matplotlib.pyplot as plt

# Convert the image into GrayScale
image_2 = cv2.imread("dog_resized2.png")
gray_image = cv2.cvtColor(image_2,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray scaled",gray_image)
cv2.waitKey(0)

print("hello")
cv2.imshow("image",image_2)
cv2.waitKey(5000)

blue_image = image_2.copy()
blue_image[:,:,1] = 0
blue_image[:,:,2] = 0

green_image = image_2.copy()
green_image[:,:,0] = 0
green_image[:,:,2] = 0

red_image = image_2.copy()
red_image[:,:,0] = 0
red_image[:,:,1] = 0

# cv2.imshow("blue image",blue_image)
# cv2.imshow("green image",green_image)
# cv2.imshow("red image",red_image)
# cv2.waitKey(0)

image_2[:,:,0] # Blue channel
image_2[:,:,1] # Green channel
image_2[:,:,2] # Red channel