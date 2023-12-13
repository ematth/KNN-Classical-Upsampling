import cv2
import matplotlib.pyplot as plt
import numpy as np


"""
Decorator to confirm numpy array dimensions according to annotations.
"""
def dimcheck(func):
    def wrapper(*args, **kwargs):
        annotes = (func.__annotations__)
        b = list(annotes.keys())
        for i, a in enumerate(args):
            if type(a[1]) == (np.ndarray or np.array):
                if (len(a.shape) == 3 and a.shape[2] == (3 or 4)) or (a.shape == None):
                    continue
                else:
                    raise TypeError(f'Array dimensions fail on {b[i]} of shape {a.shape}') 
        return func(*args, **kwargs)
    return wrapper


"""
Loads an image from path. Returns a blank Numpy Array if image is not found
"""
def get_im(path: str) -> np.ndarray((0, 0, 3), float):
    try:
        im = cv2.imread('images/' + path)
        return im.astype(float)
    except:
        return np.zeros((1., 1., 3.))
    


"""
Displays a BGR image.
"""
def display_bgr(im: np.ndarray) -> None:
    plt.imshow(im[:,:,[2,1,0]])
    plt.show()


"""
Displays images relating to prepped upsampling (original, downsampled, prepped, upsampled).
Assumes a BGR image format.
"""
titles: list[str] = ['Original', 'Downsampled', 'Prepped', 'Upsampled']
def display_prepped_upsampling_results(samples: list[np.ndarray], path: str = 'temp.png') -> None:
    [original_im, downsampled_im, prepped_im, upsampled_im] = samples
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    #fig.suptitle("Results for Prepped Upsampling", fontsize=16)
    fig.tight_layout()
    plt.axis('off')

    # Display Images
    for i, a in enumerate(axs.flatten()):
        a.imshow(samples[i][:,:,[2,1,0]])
        a.set_title(titles[i])
        a.axis('off')
        a.get_tightbbox()
    
    #plt.show()
    fig.savefig('results/' + path, bbox_inches='tight')
    

"""
Displays images relating to no-prep upsampling (original, downsampled, upsampled).
Assumes a BGR image format.
"""
def display_upsampling_results(original_im: np.ndarray, upsampled_im: np.ndarray, path: str = 'temp.png') -> None:
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    #fig.suptitle("Results for Upsampling", fontsize=16)
    fig.tight_layout()
    plt.axis('off')

    # Display the images
    axs[0].imshow(original_im[:,:,[2,1,0]])
    axs[0].set_title("Original")
    axs[0].axis('off')
    axs[0].get_tightbbox()
    
    axs[1].imshow(upsampled_im[:,:,[2,1,0]])
    axs[1].set_title("Upsampled")
    axs[1].axis('off')
    axs[1].get_tightbbox()


    #plt.show()
    fig.savefig('results/' + path, bbox_inches='tight')
    

"""
Given an image, removes on row/column from either or both dimensions to ensure image shape is even.
"""
def make_even_shape(im: np.ndarray) -> np.ndarray:
    if im.shape[0] % 2 != 0 and im.shape[1] != 0:
        return im[:im.shape[0]-1, :im.shape[1]-1]
    
    elif im.shape[0] % 2 != 0:
        return im[:im.shape[0]-1, :]
    
    elif im.shape[1] % 2 != 0:
        return im[:, im.shape[1]-1]
    
    else:
        return im