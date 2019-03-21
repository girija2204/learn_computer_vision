import cv2
import numpy as np
import matplotlib.pyplot as plt

ones = np.ones((512,1),dtype="uint8")
m = np.zeros((1,512))
x = np.linspace(-.8,.8,num=512)
m[0] = x

image = np.matmul(ones,m)

plt.imshow(image,cmap="gray")
plt.show()

cv2.circle(image,(256,256),40,color=0.1,thickness=4)

plt.imshow(image,cmap="gray")
plt.show()

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if image[i,j] < 0.2:
            image[i,j] = -.8

plt.imshow(image,cmap="gray")
plt.show()