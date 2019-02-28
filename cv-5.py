import cv2
import argparse
import numpy as np
import imutils
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Please provide the path for input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
original_image = image.copy()

(h,w, d) = image.shape
ratio = h/500.0
dim = (500,int(w*(500/h)))
image = cv2.resize(image,dim)

gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image,(5,5),0)
edged = cv2.Canny(blurred_image,75,200)

print("Edge Detected image")
cv2.imshow("Original Image", image)
cv2.imshow("Edge Detected Image", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

for c in contours:
    peri = cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if(len(approx)==4):
        screenContour = approx
        break

print("Finding Countours of the image")
cv2.drawContours(image,[screenContour],-1,(0,0,255,2))
cv2.imshow("Outline",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

warped = four_point_transform(original_image, screenContour.reshape(4,2)*ratio)
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped,11,offset=10, method="gaussian")
warped = (warped > T).astype("uint8")*255

print("Final Result")
cv2.imshow("Scanned image", imutils.resize(warped, height=650))
cv2.waitKey(0)