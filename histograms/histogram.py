import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("image-01.jpg")
cv2.imshow("Original Image",image)
cv2.waitKey(0)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("GrayScale Image",gray)
cv2.waitKey(0)

histogram = cv2.calcHist([gray],[0],None,[256],[0,256])
plt.figure()
plt.title("Histogram of the GrayScaled image")
plt.xlabel("Pixels")
plt.ylabel("Frequency of the pixels")
plt.xlim([0,256])
plt.plot(histogram)
plt.show()

# ------------------------------------------- #

channels = cv2.split(image)
colors = ("b","g","r")

plt.figure()
plt.xlim([0,256])
plt.xlabel("Pixels")
plt.ylabel("Frequency of the pixels")
plt.title("Histogram of the Colored image")
for (channel,color) in zip(channels,colors):
    histogram = cv2.calcHist([channel],[0],None,[256],[0,256])
    plt.plot(histogram,color=color)

plt.show()