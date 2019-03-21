import cv2
import matplotlib.pyplot as plt

original_image = cv2.imread("image-01.jpg")
image = original_image

plt.figure(figsize=(8,4))

ksize = 11
blurred = cv2.blur(image,(ksize,ksize))

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurred_gray = cv2.blur(gray,(ksize,ksize))

# plt.subplot(1,4,1)
# plt.imshow(image)
# plt.subplot(1,4,2)
# plt.imshow(blurred)
# plt.subplot(1,4,3)
# plt.imshow(gray,cmap="gray")
# plt.subplot(1,4,4)
# plt.imshow(blurred_gray,cmap="gray")

gblurred = cv2.GaussianBlur(original_image,(ksize,ksize),0)
gblurred_gray = cv2.GaussianBlur(gray,(ksize,ksize),0)

plt.subplot(1,2,1)
plt.imshow(blurred)
plt.subplot(1,2,2)
plt.imshow(gblurred)

plt.show()

plt.subplot(1,2,1)
plt.imshow(blurred_gray,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(gblurred_gray,cmap="gray")

plt.show()