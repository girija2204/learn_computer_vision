import argparse
import cv2
from imutils import grab_contours
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
from imutils import resize
from imutils import contours
import imutils
import numpy as np
from PIL import Image

# ap = argparse.ArgumentParser()
# ap.add_argument("-i","--image",required=True,help="Add the image path")
# args = vars(ap.parse_args())
#
# image = cv2.imread(args["image"])
image = cv2.imread('day-4\\images\\test_05.png')
ANSWER_KEY = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}

original_image = image.copy()
(h,w) = image.shape[:2]
ratio = h / 500
dim = (int(w*(500/h)),500)
resized_image1 = cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
#resized_image2 = resize(image,height=500)

gray_image = cv2.cvtColor(resized_image1,cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image,(5,5),0)
edged = cv2.Canny(blurred_image,75,200)

print("Edge Detected image")
cv2.imshow("Original image", resized_image1)
cv2.imshow("Edge Detected image", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
contours = grab_contours(contours)
contours = sorted(contours,key=cv2.contourArea,reverse=True)[:5]
for cntr in contours:
    perimeter = cv2.arcLength(cntr,True)
    approx = cv2.approxPolyDP(cntr,0.02*perimeter,True)

    if len(approx) == 4:
        screenContour = approx
        break

cv2.drawContours(resized_image1,[screenContour],-1,(0,0,255,2))
cv2.imshow("Outline",resized_image1)
cv2.waitKey(0)
cv2.destroyAllWindows()


warped = four_point_transform(original_image, screenContour.reshape(4,2)*ratio)
warped_rgb = warped.copy()
warped = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
T = threshold_local(warped,11,offset=10,method="gaussian")
warped = (warped > T).astype("uint8")*255

print("Final result")
cv2.imshow("Scanned image",resize(warped_rgb,height=500))
cv2.imshow("Scanned image - GrayScale",resize(warped,height=500))
cv2.waitKey(0)
cv2.destroyAllWindows()

thresholded_image = cv2.threshold(warped,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

print("Thresholded image")
cv2.imshow("TI",thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cnts = cv2.findContours(thresholded_image.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = grab_contours(cnts)
question_cnts = []

for cnt in cnts:
    (x,y,w,h) = cv2.boundingRect(cnt)
    ar = w / float(h)

    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        question_cnts.append(cnt)

print(len(question_cnts))

cv2.drawContours(warped_rgb,question_cnts,-1,(0,0,255,2))
cv2.imshow("Outline of circles",warped_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

question_cnts = imutils.contours.sort_contours(question_cnts,method="top-to-bottom")[0]

for (q,i) in enumerate(np.arange(0,len(question_cnts),5)):
    cnts = imutils.contours.sort_contours(question_cnts[i:i+5])[0]

    bubbled = None
    correct = 0
    for (j, c) in enumerate(cnts):
        mask = np.zeros(thresholded_image.shape, dtype="uint8")
        cv2.drawContours(mask,[c],-1,(255,255,255),-1)

        mask = cv2.bitwise_and(thresholded_image,thresholded_image,mask=mask)
        total = cv2.countNonZero(mask)

        if bubbled is None or total > bubbled[0]:
            bubbled = (total,j)

        color = (0,0,255)
        k = ANSWER_KEY[q]

        if k == bubbled[1]:
            color = (0,255,0)
            correct += 1

        cv2.drawContours(warped_rgb, [cnts[k]],-1,color,3)

score = (correct / 5.0) * 100
print("Correct",correct)
print("Score: {:.2f}%".format(score))
cv2.putText(warped_rgb,"{:.2f}%".format(score),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
cv2.imshow("Original Paper: ", image)
cv2.imshow("Result: ", warped_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()