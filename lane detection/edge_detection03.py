import cv2
import numpy as np
from utils import multiple_images
from gaussianNoise import addGaussianNoiseToImg

def non_max_suppression(gradient_mat,theta_mat):
    gradient = np.zeros(gradient_mat.shape,dtype="uint8")
    theta_mat[theta_mat<0] += 180
    pixel1 = -1
    pixel2 = -1

    for i in range(gradient.shape[0]):
        for j in range(gradient.shape[1]):

            try:
                if (0 <= theta_mat[i, j] < 22.5 or 157.5 <= theta_mat[i, j] < 180):
                    pixel1 = gradient_mat[i, j + 1]
                    pixel2 = gradient_mat[i, j - 1]
                elif (22.5 <= theta_mat[i, j] < 67.5):
                    pixel1 = gradient_mat[i - 1, j + 1]
                    pixel2 = gradient_mat[i + 1, j - 1]
                elif (67.5 <= theta_mat[i, j] < 112.5):
                    pixel1 = gradient_mat[i - 1, j]
                    pixel2 = gradient_mat[i + 1, j]
                elif (112.5 <= theta_mat[i, j] < 157.5):
                    pixel1 = gradient_mat[i - 1, j - 1]
                    pixel2 = gradient_mat[i + 1, j + 1]

                if (gradient_mat[i, j] >= pixel1 and gradient_mat[i, j] >= pixel2):
                    gradient[i, j] = gradient_mat[i, j]
                else:
                    gradient[i, j] = 0

            except IndexError as e:
                pass

    return gradient

def threshold(gradient,low_thres=0.02,high_thres=0.5,strong_color=255,weak_color=100):
    low_thres = grad.max()*low_thres
    high_thres = grad.max()*high_thres

    thresholded = np.zeros(gradient.shape,dtype="uint8")

    strong_i,strong_j = np.where(gradient >= high_thres)
    weak_i, weak_j = np.where((low_thres <= gradient) & (gradient < high_thres))
    zeros_i, zeros_j = np.where(gradient < low_thres)

    thresholded[strong_i,strong_j] = 255
    thresholded[weak_i,weak_j] = 100

    return thresholded

def hysteresis(thresholded,strong_color,weak_color):
    for i in range(thresholded.shape[0]):
        for j in range(thresholded.shape[1]):
            if(thresholded[i,j]==100):
                try:
                    if(thresholded[i+1,j] == strong_color or thresholded[i-1,j] == strong_color or thresholded[i,j+1] == strong_color or
                    thresholded[i,j-1] == strong_color or thresholded[i+1,j+1] == strong_color or thresholded[i-1,j-1] == strong_color or
                            thresholded[i-1,j+1] == strong_color or thresholded[i+1,j-1] == strong_color):
                        thresholded[i,j] = strong_color
                    else:
                        thresholded[i,j] = 0
                except IndexError as e:
                    pass

    return thresholded

image = cv2.imread("lenna.png")
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
_,image = addGaussianNoiseToImg(image)


image = cv2.GaussianBlur(image,(5,5),sigmaX=1)
kxx,kyx = cv2.getDerivKernels(dx=1,dy=0,ksize=3)
kxy,kyy = cv2.getDerivKernels(dx=0,dy=1,ksize=3)

der_img_x = cv2.sepFilter2D(image,ddepth=-1,kernelX=kxx,kernelY=kyx)
der_img_y = cv2.sepFilter2D(image,ddepth=-1,kernelX=kxy,kernelY=kyy)

sx = cv2.convertScaleAbs(der_img_x)

sy = cv2.convertScaleAbs(der_img_y)

sxy = cv2.addWeighted( sx, 0.5, sy, 0.5, 0);

gradient = np.hypot(der_img_x,der_img_y)
gradient = gradient/gradient.max()*255
theta = np.arctan2(der_img_y,der_img_x)
deg = theta*180./np.pi

strong_color = 200
weak_color = 170
grad = non_max_suppression(gradient,theta)
thresholded = threshold(grad,strong_color=strong_color,weak_color=weak_color)
thresholded_copy = np.zeros(thresholded.shape,dtype="uint8")
np.copyto(thresholded_copy,thresholded)
final_result = hysteresis(thresholded_copy,strong_color=strong_color,weak_color=weak_color)

# cannied = cv2.Canny(image,threshold1=25,threshold2=200)
cannied = cv2.Canny(image,100,200)

images1 = [np.array([image,sxy,final_result,cannied])]
names1 = [np.array(['Original Image','Sobelled','Final Result','Cannied'])]
multiple_images(images1,names1)
print('hello')

# Note: Show all of them. i.e. canny on the original image and canny on the noisy image too (which does not work).