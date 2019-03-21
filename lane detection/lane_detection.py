import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("test_image.jpg")
original_image = np.copy(image)

image_size = f'image size:: no of rows: {image.shape[0]}, no of cols: {image.shape[1]}, no of channels: {image.shape[2]}'
print(image_size)

cv2.imshow("image",original_image)
cv2.waitKey(0)

def canny(original_image):
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (11, 11), 0)
    canny_image = cv2.Canny(blurred_image, 50, 150)
    return canny_image

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    triangle = np.array([(200,height),(1100,height),(550,250)])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,[triangle],255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

# 1. Convert image to grayscale
# 2. Reduce Noise
# 3. Canny method
canny_image = canny(original_image)

plt.subplot(1,1,1)
plt.imshow(canny_image,cmap="gray")

plt.show()

# 4. Region of interest
# 5. Bitwise_and
masked_image = region_of_interest(canny_image)

plt.subplot(1,1,1)
plt.imshow(masked_image,cmap="gray")

plt.show()

# 6. Hough Transform
lines = cv2.HoughLinesP(masked_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
line_image = display_lines(original_image,lines)

combo_image = cv2.addWeighted(original_image,0.8,line_image,1,1)
cv2.imshow("combo image",combo_image)
cv2.waitKey(0)
