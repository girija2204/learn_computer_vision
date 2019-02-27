import cv2
import argparse
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())

tetris = cv2.imread(args["image"])
cv2.imshow("Tetris", tetris)
cv2.waitKey(0)

gray_tetris = cv2.cvtColor(tetris,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Tetris", gray_tetris)
cv2.waitKey(0)

edged = cv2.Canny(gray_tetris, 40, 100)
cv2.imshow("Edged", edged)
cv2.waitKey(0)

thresh = cv2.threshold(gray_tetris, 225, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Thresholded image", thresh)
cv2.waitKey(0)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
tetris_copy = tetris.copy()

for c in cnts:
    cv2.drawContours(tetris_copy, [c], -1, (0,0,255), 2)
    cv2.imshow("Contours", tetris_copy)
    cv2.waitKey(0)

cv2.putText(tetris_copy, "I found {} objects".format(len(cnts)), (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
cv2.imshow("Contours", tetris_copy)
cv2.waitKey(0)


thresh_copy = thresh.copy()
eroded = cv2.erode(thresh_copy, None, iterations=5)
cv2.imshow("Eroded", eroded)
cv2.waitKey(0)

thresh_copy = thresh.copy()
dilated = cv2.dilate(thresh_copy, None, iterations=5)
cv2.imshow("Dilated", dilated)
cv2.waitKey(0)

thresh_copy = thresh.copy()
bitwise = cv2.bitwise_and(tetris, tetris, mask=thresh_copy)
cv2.imshow("Output", bitwise)
cv2.waitKey(0)