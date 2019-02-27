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
M = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated = cv2.warpAffine(image, M, (w,h))
cv2.imshow("Rotated image", rotated)
cv2.waitKey(0)

# ------------------------------------ #

blurred_image = cv2.GaussianBlur(image, (11,11),0)
cv2.imshow("Blurred image", blurred_image)
cv2.waitKey(0)

# ------------------------------------ #

image_copy = image.copy()
cv2.rectangle(image_copy, (320,60), (480,180), (0,0,255), 2)
cv2.imshow("Rectangle", image_copy)
cv2.waitKey(0)

image_copy = image.copy()
cv2.circle(image_copy, (320,60), 40, (0,0,255), -4)
cv2.imshow("Circle", image_copy)
cv2.waitKey(0)

image_copy = image.copy()
cv2.line(image_copy,(320,60), (480,180), (0,0,255), 2)
cv2.imshow("Line", image_copy)
cv2.waitKey(0)

image_copy = image.copy()
cv2.putText(image_copy, "This is text", (500,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
cv2.imshow("Text", image_copy)
cv2.waitKey(0)