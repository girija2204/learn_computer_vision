import cv2
import numpy as np
from skimage.util import random_noise
import matplotlib.pyplot as plt

image = cv2.imread("image-02.png")

image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

noisy_image = random_noise(image,"gaussian")
noisy_image = (255*noisy_image).astype("uint8")

histogram_image = cv2.calcHist([image],[0],None,[256],[0,256])
histogram_noisy_image = cv2.calcHist([noisy_image],[0],None,[256],[0,256])

threshold = 80

modified_image = np.zeros([image.shape[0],image.shape[1]],dtype="uint8")
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if image[i,j] > threshold:
            modified_image[i,j] = 255
        else: modified_image[i,j] = 0


modified_noisy_image = np.zeros([noisy_image.shape[0],noisy_image.shape[1]],dtype="uint8")
for i in range(noisy_image.shape[0]):
    for j in range(noisy_image.shape[1]):
        if noisy_image[i,j] > threshold:
            modified_noisy_image[i,j] = 255
        else:
            modified_noisy_image[i,j] = 0

plt.figure(figsize=(8,6))

plt.subplot(1,2,1)
plt.imshow(image,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(modified_image,cmap="gray")

plt.show()

plt.subplot(1,2,1)
plt.imshow(noisy_image,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(modified_noisy_image,cmap="gray")

plt.show()

plt.title("Histogram of the image")
plt.xlabel("Pixels")
plt.ylabel("Frequency of the pixels")
plt.plot(histogram_image)


plt.title("Histogram of the noisy image")
plt.xlabel("Pixels")
plt.ylabel("Frequency of the pixels")
plt.plot(histogram_noisy_image)

plt.show()