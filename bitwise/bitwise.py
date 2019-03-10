import cv2
import numpy as np

image = cv2.imread("image-01.jpg")

cv2.imshow("image",image)
cv2.waitKey(0)

mask = np.zeros(image.shape,dtype="uint8")

cv2.rectangle(mask,(44,44),(700,200),(255,255,255),-1)

cv2.imshow("Mask",mask)
cv2.waitKey(0)

maskedImg = cv2.bitwise_and(image,mask)
cv2.imshow("Masked Image",maskedImg)
cv2.waitKey(0)