import numpy as np
import  matplotlib.pyplot as plt
import cv2
from utils import multiple_plots

mat1 = np.full((30,50),fill_value=40,dtype=int)
mat2 = np.full((40,50),fill_value=60,dtype=int)
mat3 = np.full((20,50),fill_value=20,dtype=int)
mat = np.append(mat1,mat2,axis=0)
mat = np.append(mat,mat3,axis=0)

plt.imshow(mat)
plt.axis("off")
plt.show()

k1 = [1,0,-1]
k2 = [1,-2,1]

diff_col_1 = np.convolve(mat[:,3],k1,mode="valid")
diff_col_2 = np.convolve(mat[:,3],k2,mode="valid")

list_of_plots = [np.array([mat[:,3],diff_col_1,diff_col_2])]
list_of_names = [np.array(["Pixel intensities variations on a single column of the image","Differentiation 1st der","Differentiation 2nd der"])]
multiple_plots(list_of_plots,list_of_names)