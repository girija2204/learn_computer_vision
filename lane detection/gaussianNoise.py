import numpy as np
from skimage import img_as_float, img_as_ubyte

def addGaussianNoiseToImg(image):
    noise = getGaussianNoise(image.shape)
    noisy_image = addNoiseToData(image,noise)
    return noise, noisy_image

def getGaussianNoise(shape):
    mean = 0.0
    variance = 0.001
    noise = np.random.normal(mean,variance**0.5,shape)
    return noise

def getPoissonNoise(shape):
    pass

def addNoiseToData(image,noise):
    image_float = img_as_float(image)  # range of a floating point image is 0.0 to 1.0
    noisy_image = image_float + noise
    noisy_image = np.clip(noisy_image, 0.0, 1.0)
    noisy_image = img_as_ubyte(noisy_image)
    return noisy_image