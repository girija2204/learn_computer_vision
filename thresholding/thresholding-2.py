import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("image-02.png")

gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

gray_image_th1 = np.zeros([gray_image.shape[0],gray_image.shape[1]],dtype="uint8")
gray_image_th2 = np.zeros([gray_image.shape[0],gray_image.shape[1]],dtype="uint8")
gray_image_th3 = np.zeros([gray_image.shape[0],gray_image.shape[1]],dtype="uint8")

for i in range(gray_image.shape[0]):
    for j in range(gray_image.shape[1]):
        if gray_image[i,j] < 20:
            gray_image_th1[i,j] = 255
        else: gray_image_th1[i,j] = 0
        if gray_image[i,j] < 10:
            gray_image_th2[i,j] = 255
        else: gray_image_th2[i,j] = 0
        if gray_image[i,j] < 50:
            gray_image_th3[i,j] = 255
        else: gray_image_th3[i,j] = 0

plt.figure(figsize=(8,4))

plt.subplot(1,4,1)
plt.imshow(gray_image,cmap="gray")
plt.subplot(1,4,2)
plt.imshow(gray_image_th1,cmap="gray")
plt.subplot(1,4,3)
plt.imshow(gray_image_th2,cmap="gray")
plt.subplot(1,4,4)
plt.imshow(gray_image_th3,cmap="gray")

#plt.show()

hist = cv2.calcHist([gray_image],[0],None,[256],[0,256])

plt.subplot(1,1,1)
plt.xlabel("Pixel Values")
plt.ylabel("Frequencies")
plt.plot(hist)
plt.show()