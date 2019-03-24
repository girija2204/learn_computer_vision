import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import img_as_float, img_as_ubyte
from scipy import stats

image = cv2.imread("lenna.png")

image_float = img_as_float(image) # range of a floating point image is 0.0 to 1.0

mean = 0.0
variance = 0.01
noise = np.random.normal(mean,variance**0.5,image.shape)
noisy_image = image_float + noise

noisy_image = np.clip(noisy_image,0.0,1.0)
noisy_image = img_as_ubyte(noisy_image)

plt.style.use('seaborn') # pretty matplotlib plots
plt.rcParams['figure.figsize'] = (12, 8)

fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(8,4))
ax1.axis("off")
ax1.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
ax2.axis("off")
ax2.imshow(cv2.cvtColor(noisy_image,cv2.COLOR_BGR2RGB))
plt.show()


fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(8,4))
# # major doubt cleared: standard deviation is sigma and it is square root of variance
image_first_channel = image[:,:,0]
count, bins, ignored = ax1.hist(image_first_channel.ravel(), 256, density=True)
ax1.plot(bins, 1 / (np.sqrt(2 * np.pi * np.power(bins.std(), 2))) * \
         (np.power(np.e, -(np.power((bins - bins.mean()), 2) / (2 * np.power(bins.std(), 2))))),linewidth=2, color='r')
ax1.title.set_text("Original Image")
ax1.set_xlabel("pixels range 0-255")
ax1.set_ylabel("Frequency of the pixels")

noise_first_channel = noise[:,:,0]
count, bins, ignored = ax2.hist(noise_first_channel.ravel(), 256, density=True)
ax2.plot(bins, 1 / (np.sqrt(2 * np.pi * np.power(bins.std(), 2))) * \
         (np.power(np.e, -(np.power((bins - bins.mean()), 2) / (2 * np.power(bins.std(), 2))))),linewidth=2, color='r')
ax2.title.set_text("Noise Image")
ax2.set_xlabel("pixels range 0.0-1.0")
ax2.set_ylabel("Frequency of the pixels")

noisy_image_first_channel = noisy_image[:,:,0]
count, bins, ignored = ax3.hist(noisy_image_first_channel.ravel(), 256, density=True)
ax3.plot(bins, 1 / (np.sqrt(2 * np.pi * np.power(bins.std(), 2))) * \
         (np.power(np.e, -(np.power((bins - bins.mean()), 2) / (2 * np.power(bins.std(), 2))))),linewidth=2, color='r')
ax3.title.set_text("Noisy Image")
ax3.set_xlabel("pixels range 0-255")
ax3.set_ylabel("Frequency of the pixels")

plt.show()