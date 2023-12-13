import numpy as np
import matplotlib.pyplot as plt
import cv2
from functions.utility import dimcheck


"""
# Downsampling

Here we implement downsampling approaches to be used in our pipeline. 
We will use these functions to downsample our test images, upsample those 
downsamples, then compare with the original.
"""

@dimcheck
def downsample(im: np.ndarray((0, 0, 3))) -> np.ndarray((0, 0, 3)):
    """
    Downsamples an image by factor of 2 by throwing out every odd indexed pixel.
    Assumes even image shape.

    Args:
        im (np.ndarray): Image array of size (w, h, 3).

    Returns:
        np.ndarray: downsampled Image array of size(w/2, h/2, 3).
    """
    shape = im.shape
    new_shape = (shape[0] // 2, shape[1] // 2, shape[0])

    new_im = np.zeros(new_shape, dtype=float)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if j % 2 == 0 and i % 2 == 0:
                new_im[i//2, j//2] = im[i, j]
                    
    return new_im


@dimcheck
def cv2_downsample(im: np.ndarray((0, 0, 3))) -> np.ndarray((0, 0, 3)):
    """
    Downsamples an image by a factor of 2 using CV2 library functions.
    Reference: https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/#gsc.tab=0

    Args:
        im (np.ndarray): Image array of size (w, h, 3).

    Returns:
        np.ndarray: downsampled Image array of size(w/2, h/2, 3).
    """    
    (a, b, c) = im.shape   
    scale_percent = 50 # percent of original size
    width = int(b * scale_percent / 100)
    height = int(a * scale_percent / 100)
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

@dimcheck
def prep_upsample(im: np.ndarray((0, 0, 3)), factor: int = 2) -> np.ndarray((0, 0, 3)):
    """
    Prepares to upsample and image by factor of "factor" by creating an image 
    where every even indexed pixel originates from the provided image and every 
    odd indexed pixel is empty.

    Args:
        im (np.ndarray): Image array of size (w, h, 3).
        factor (int, optional): Factor to upscale the image by. Defaults to 2.

    Returns:
        np.ndarray: Prepped Upsampled Image array of size (factor * w, factor * h, 3).
    """    
    (a, b, c) = im.shape
    new_shape = (a * factor, b * factor, c)
    new_im = np.zeros(new_shape, dtype=float)
    
    for i in range(a):
        for j in range(b):
            new_im[i*2, j*2] = im[i, j]
                            
    return new_im


@dimcheck
def KNN_upsample_prepped(im: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Given a prepped image, fills in the "empty" pixels (odd indices) by 
    averaging all even indexed pixels up to k steps away.
    This implementation only works for images prepped by prep_upsample.

    Args:
        im (np.ndarray): Image array of size (factor * w, factor * h, 3).
        k (int, optional): Number of steps away from a given pixel to use in approximation its color value. Defaults to 1.

    Returns:
        np.ndarray: Upsampled Image array of size (factor * w, factor * h, 3).
    """    
    (a, b, c) = im.shape
    new_im = np.zeros_like(im, dtype=float)
    for i in range(a):
        for j in range(b):
            if i % 2 == 1 or j % 2 == 1:
                # Must fill this value by performing a KNN average
                neighbor_pixel_values = []
                for m in range(max(0, i-k), min(i+k+1, a-1)):
                    for n in range(max(0, j-k), min(j+k+1, b-1)):
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

@dimcheck
def KNN_upsample_no_prep(im: np.ndarray, k: int = 1, factor: int = 2) -> np.ndarray:
    """
    Given an image, upsamples both dimensions by a factor of 2. This is equivilant to the "KNN_upsample_prepped" function
    without the necessity for creating an intermediary prepped image.

    Args:
        im (np.ndarray): Image array of size (w, h, 3).
        k (int, optional): Number of steps away from a given pixel to use in approximation of its color value. Defaults to 1.
        factor (int, optional): Factor to upsample the Image by. Defaults to 2.

    Returns:
        np.ndarray: Upsampled Image array of size (factor * w, factor * h, 3).
    """    
    (a, b, c) = im.shape
    new_im = np.zeros((a * factor, b * factor, c), dtype=np.float64)
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

                avg_value = np.mean(np.array(neighbor_pixel_values), axis=0)
                new_im[i, j] = avg_value     
    
    return new_im


@dimcheck
def KNN_upsample_variable_factor(im: np.ndarray, k: int = 1, factor: int = 2) -> np.ndarray:
    """
    Given an image, upsamples both dimensions by some factor.

    Args:
        im (np.ndarray): Image array of size (w, h, 3).
        k (int, optional): Number of steps away from a given pixel to use in approximation of its color value. Defaults to 1.
        factor (int, optional): Factor to upsample the image in both dimensions by. Defaults to 2.

    Returns:
        np.ndarray: Upsampled Image array of size (factor * w, factor * h, 3).
    """    
    (a, b, c) = im.shape
    new_im = np.zeros((a * factor, b * factor, c), dtype=np.float64)
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
                            
                avg_value = np.mean(np.array(neighbor_pixel_values), axis=0)
                new_im[i, j] = avg_value     
    
    return new_im


@dimcheck
def KNN_upsample_variable_factors(im: np.ndarray, k: int = 1, factor1: int = 2, factor2: int = 2) -> np.ndarray:
    """
    Given an image, upsamples each dimension by some (possibly unique) factor.

    Args:
        im (np.ndarray): Image array of size (w, h, 3).
        k (int, optional): Number of steps away from a given pixel to use in approximation of its color value. Defaults to 1.
        factor (int, optional): Factor to upsample the image in the horizontal dimension by. Defaults to 2.
        factor2 (int, optional): Factor to upsample the image in the vertical dimension by. Defaults to 2.

    Returns:
        np.ndarray: Upsampled Image array of size (factor1 * w, factor2 * h, 3)
    """    
    (a, b, c) = im.shape
    new_im = np.zeros((a * factor1, b * factor2, c), dtype=np.float64)
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


@dimcheck
def edge_detection(im: np.ndarray((0, 0, 3)), a: int = 100, b: int = 200, l2: bool = False) -> np.ndarray((0, 0, 3)):
    return cv2.Canny(im, a, b)