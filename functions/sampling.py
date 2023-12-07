import numpy as np
import matplotlib.pyplot as plt
import cv2


"""
# Downsampling

Here we implement downsampling approaches to be used in our pipeline. 
We will use these functions to downsample our test images, upsample those 
downsamples, then compare with the original.
"""

"""
Downsamples an image by factor of 2 by throwing out every odd indexed pixel.
Assumes even image shape

"""
def downsample(im: np.ndarray) -> np.ndarray:
    shape = im.shape
    new_shape = (shape[0] // 2, shape[1] // 2, shape[2])

    new_im = np.zeros(new_shape, dtype=float)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if j % 2 == 0 and i % 2 == 0:
                new_im[i//2, j//2] = im[i, j]
                    
    return new_im

def cv2_downsample(im: np.ndarray) -> np.ndarray:
    # https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/#gsc.tab=0
    scale_percent = 50 # percent of original size
    width = int(im.shape[1] * scale_percent / 100)
    height = int(im.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    # resize image
    resized = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
    
    return resized


"""
"Prepped" Upsampling

Here we define an approach to upsampling that "prepares" the upsampled image
prior to performing any interpolation. "Preparation" here refers to generating 
an image with shape equal to the desired upsample, filled with original pixel 
values (from the downsampled image) where they would appear in the upsampled 
image, and empty (black) pixels everywhere else. This approach adds significant 
computation/runtime to the pipeline, but may provide more control than less 
supervised implementations (implemented / to-be implemented further below).
"""


"""
Prepares to upsample and image by factor of "factor" by creating an image 
where every even indexed pixel originates from the provided image and every 
odd indexed pixel is empty.
"""
def prep_upsample(im: np.ndarray, factor: int = 2) -> np.ndarray:
    (a, b, c) = im.shape
    new_shape = (a * factor, b * factor, c)
    new_im = np.zeros(new_shape, dtype=float)
    
    for i in range(a):
        for j in range(b):
            new_im[i*2, j*2] = im[i, j]
                            
    return new_im


"""
Given a prepped image, fills in the "empty" pixels (odd indices) by 
averaging all even indexed pixels up to k steps away.
This implementation only works for images prepped by prep_upsample
"""
def KNN_upsample_prepped(im: np.ndarray, k: int = 1) -> np.ndarray:
    new_im = np.zeros_like(im, dtype=float)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if i % 2 == 1 or j % 2 == 1:
                # Must fill this value by performing a KNN average
                neighbor_pixel_values = []
                for m in range(max(0, i-k), min(i+k+1, im.shape[0]-1)):
                    for n in range(max(0, j-k), min(j+k+1, im.shape[1]-1)):
                        if m % 2 == 0 and n % 2 == 0:
                            neighbor_pixel_values.append(im[m, n])
                            
                to_average = np.array(neighbor_pixel_values)
                avg_value = np.mean(to_average, axis=0)
                new_im[i, j] = avg_value
            else:
                new_im[i, j] = im[i, j]
                
    
    return new_im


"""
# "No-Prep" Upsampling

Here we implement approaches to upsampling that need not generate any intermediary images.
These approaches will only calculate the bare minimum pre-upsampling information required 
to perform the task.
"""

"""
Given an image, upsamples both dimensions by a factor of 2. This is equivilant to the "KNN_upsample_prepped" function
without the necessity for creating an intermediary prepped image.
"""
def KNN_upsample_no_prep(im: np.ndarray, k: int = 1) -> np.ndarray:
    factor = 2
    new_im = np.zeros((im.shape[0] * factor, im.shape[1] * factor, im.shape[2]), dtype=float)
    for i in range(new_im.shape[0]):
        for j in range(new_im.shape[1]):
            if i % factor == 0 and j % factor == 0:
                # This index exists in the original image, so we can just copy its value
                new_im[i, j] = im[i // factor, j // factor]
            
            else:
                # This index doesn't exist in the original image, so we must interpolate its value
                neighbor_pixel_values = []
                # Find all pixels within the original image that are within k steps away from the index we wish to interpolate
                for m in range(max(0, i-k), min(i+k+1, new_im.shape[0]-1)):
                    for n in range(max(0, j-k), min(j+k+1, new_im.shape[1]-1)):
                        if m % factor == 0 and n % factor == 0:
                            neighbor_pixel_values.append(im[m // factor, n // factor])
                            
                to_average = np.array(neighbor_pixel_values)
                avg_value = np.mean(to_average, axis=0)
                new_im[i, j] = avg_value     
    
    return new_im


"""
Given an image, upsamples both dimensions by some factor.
"""
def KNN_upsample_variable_factor(im: np.ndarray, k: int = 1, factor: int = 2) -> np.ndarray:
    new_im = np.zeros((im.shape[0] * factor, im.shape[1] * factor, im.shape[2]), dtype=float)
    for i in range(new_im.shape[0]):
        for j in range(new_im.shape[1]):
            if i % factor == 0 and j % factor == 0:
                # This index exists in the original image, so we can just copy its value
                new_im[i, j] = im[i // factor, j // factor]
            
            else:
                # This index doesn't exist in the original image, so we must interpolate its value
                neighbor_pixel_values = []
                # Find all pixels within the original image that are within k steps away from the index we wish to interpolate
                for m in range(max(0, i-k), min(i+k+1, new_im.shape[0]-1)):
                    for n in range(max(0, j-k), min(j+k+1, new_im.shape[1]-1)):
                        if m % factor == 0 and n % factor == 0:
                            neighbor_pixel_values.append(im[m // factor, n // factor])
                            
                to_average = np.array(neighbor_pixel_values)
                avg_value = np.mean(to_average, axis=0)
                new_im[i, j] = avg_value     
    
    return new_im


"""
Given an image, upsamples each dimension by some (possibly unique) factor.
"""
def KNN_upsample_variable_factors(im: np.ndarray, k: int = 1, factor1: int = 2, factor2: int = 3) -> np.ndarray:
    new_im = np.zeros((im.shape[0] * factor1, im.shape[1] * factor2, im.shape[2]), dtype=float)
    for i in range(new_im.shape[0]):
        for j in range(new_im.shape[1]):
            if i % factor1 == 0 and j % factor2 == 0:
                # This index exists in the original image, so we can just copy its value
                new_im[i, j] = im[i // factor1, j // factor2]
            
            else:
                # This index doesn't exist in the original image, so we must interpolate its value
                neighbor_pixel_values = []
                # Find all pixels within the original image that are within k steps away from the index we wish to interpolate
                for m in range(max(0, i-k), min(i+k+1, new_im.shape[0]-1)):
                    for n in range(max(0, j-k), min(j+k+1, new_im.shape[1]-1)):
                        if m % factor1 == 0 and n % factor2 == 0:
                            neighbor_pixel_values.append(im[m // factor1, n // factor2])
                            
                to_average = np.array(neighbor_pixel_values)
                avg_value = np.mean(to_average, axis=0)
                new_im[i, j] = avg_value     
    
    return new_im
