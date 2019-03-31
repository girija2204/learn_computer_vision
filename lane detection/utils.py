import cv2
import matplotlib.pyplot as plt

def plot_multiple(array_of_images, name_array):
    plt.figure(figsize=(8,4))
    for i in range(array_of_images.shape[0]):
        plt.subplot(1,array_of_images.shape[0],i+1)
        plt.imshow(cv2.cvtColor(array_of_images[i],cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(name_array[i])
    plt.show()