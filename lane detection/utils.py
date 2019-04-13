import cv2
import numpy as np
import matplotlib.pyplot as plt

def multiple_images(list_of_images = [],list_of_titles = []):
    fig, ax = plt.subplots(nrows=list_of_images.__len__(), ncols=list_of_images[0].shape[0], figsize=(8, 4))

    count = -1
    count_j = -1
    for i, axi in enumerate(ax.flat):
        count = np.floor(i / (list_of_images[0].shape[0])).astype(int)
        count_j = i % (list_of_images[0].shape[0])
        axi.imshow(cv2.cvtColor(list_of_images[count][count_j],cv2.COLOR_BGR2RGB))
        axi.set_title(list_of_titles[count][count_j])

    plt.show()

def multiple_plots(list_of_plots = [],list_of_titles = []):
    fig, ax = plt.subplots(nrows=list_of_plots.__len__(), ncols=list_of_plots[0].shape[0], figsize=(8,4))

    count = -1
    count_j = -1
    for i, axi in enumerate(ax.flat):
        count = np.floor(i/(list_of_plots[0].shape[0])).astype(int)
        count_j = i%(list_of_plots[0].shape[0])
        axi.plot(list_of_plots[count][count_j])
        axi.set_title(list_of_titles[count][count_j])

    plt.show()