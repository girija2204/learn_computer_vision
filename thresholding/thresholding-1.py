import numpy as np
import cv2

image = cv2.imread("image-01.jpg")
cv2.imshow("image",image)
cv2.waitKey(0)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Scale image",gray)
cv2.waitKey(0)

blurred = cv2.GaussianBlur(gray,(11,11),0)
cv2.imshow("Blurred iamge",blurred)
cv2.waitKey(0)

