import imutils
import cv2

image = cv2.imread("day-2/jp.png")
(h,w,d) = image.shape
print(f"width={w}, height={h}, depth={d}")

cv2.imshow("Image", image)
cv2.waitKey(0)

(B,G,R) = image[100,500]
print("R={}, G={}, B={}".format(R,G,B))

roi = image[60:100,200:230]
cv2.imshow("ROI",roi)
cv2.waitKey(0)

resized = cv2.resize(image,(200,200))
cv2.imshow("Resized image", resized)
cv2.waitKey(0)

ratio = 300 / w
dim = (300, int(h * ratio))
resized_proper = cv2.resize(image,dim)
cv2.imshow("Resized properly", resized_proper)
cv2.waitKey(0)

center = (w//2, h//2)