import cv2
import matplotlib.pyplot as plt
"""
Loads an image from path.
"""
def get_im(path):
    im = cv2.imread(path)
    return im

"""
Displays a BGR image.
"""
def display_bgr(im):
    plt.imshow(im[:,:,[2,1,0]])
    plt.show()

"""
Displays images relating to prepped upsampling (original, downsampled, prepped, upsampled).
Assumes a BGR image format.
"""
def display_prepped_upsampling_results(original_im, downsampled_im, prepped_im, upsampled_im):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Results for Prepped Upsampling", fontsize=16)

    # Display the images
    axs[0, 0].imshow(original_im[:,:,[2,1,0]])
    axs[0, 0].set_title("Original")
    
    axs[0, 1].imshow(downsampled_im[:,:,[2,1,0]])
    axs[0, 1].set_title("Downsampled")
    
    axs[1, 0].imshow(prepped_im[:,:,[2,1,0]])
    axs[1, 0].set_title("Prepped")
    
    axs[1, 1].imshow(upsampled_im[:,:,[2,1,0]])
    axs[1, 1].set_title("Upsampled")
    
    plt.show()
    
"""
Displays images relating to no-prep upsampling (original, downsampled, upsampled).
Assumes a BGR image format.
"""
def display_upsampling_results(original_im, upsampled_im):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Results for Upsampling", fontsize=16)

    # Display the images
    axs[0].imshow(original_im[:,:,[2,1,0]])
    axs[0].set_title("Original")
    
    axs[1].imshow(upsampled_im[:,:,[2,1,0]])
    axs[1].set_title("Upsampled")
    
    plt.show()
    
"""
Given an image, removes on row/column from either or both dimensions to ensure image shape is even.
"""
def make_even_shape(im):
    if im.shape[0] % 2 != 0 and im.shape[1] != 0:
        return im[:im.shape[0]-1, :im.shape[1]-1]
    
    elif im.shape[0] % 2 != 0:
        return im[:im.shape[0]-1, :]
    
    elif im.shape[1] % 2 != 0:
        return im[:, im.shape[1]-1]
    
    else:
        return im