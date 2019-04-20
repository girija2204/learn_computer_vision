import cv2
import numpy as np
import matplotlib.pyplot as plt

room = cv2.imread("room.jpg")
room_gray = cv2.cvtColor(room,cv2.COLOR_BGR2GRAY)

cv2.imshow("Room",room_gray)
cv2.waitKey(0)

hist = cv2.calcHist([room_gray],channels=[0],mask=None,histSize=[256],ranges=[0,256])

plt.plot(hist)
plt.show()

step = np.zeros(room_gray.shape,dtype="uint8")

for i in  range(room_gray.shape[0]):
    for j in range(room_gray.shape[1]):
        step[i,j] = 255 if(room_gray[i,j] >= 175) else 0

cv2.imshow("step",step)
cv2.waitKey(0)


dig_neg = np.zeros(room_gray.shape,dtype="uint8")

for i in  range(room_gray.shape[0]):
    for j in range(room_gray.shape[1]):
        dig_neg[i,j] = 255-room_gray[i,j]

cv2.imshow("dig_neg",dig_neg)
cv2.waitKey(0)



contrast = np.zeros(room_gray.shape,dtype="uint8")