import numpy as np

"""
Calculates the mean squared difference per pixel between two images of the same shape.
Lower MSE indicates more similar images.
With perfect similarity, MSE=0.
"""
def mse(im1, im2):
    error = np.sum((im1.astype(float) - im2.astype(float)) ** 2)
    error /= im1.shape[0] * im1.shape[1]
        
    return error

"""
Calculates the root mean squared difference per pixel between two images of the same shape.
Lower RMSE indicates more similar images.
With perfect similarity, RMSE=0.
"""
def rmse(im1, im2):
    error = np.sqrt(mse(im1, im2))
    
    return error


"""
Calculate the mean absolute difference per pixel between two images of the same shape.
Lower MAE indicates more similar images.
With perfect similarity, MAE=0.
"""
def mae(im1, im2):
    error = np.sum(np.abs(im1.astype(float) - im2.astype(float)))
    error /= im1.shape[0] * im1.shape[1]  
    
    return error

"""
Calculate the peak signal to noise ratio between two images of the same shape.
Higher PSNR indicates higher-quality reconstruction (generally in the range of 30-50Db).
With perfect similarity, PSNR=infinity
"""
def psnr(im1, im2):
    # https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    error = (20 * np.log10(255)) - (10 * np.log10(mse(im1, im2)))
    
    return error
    

"""
Calculate the structural similarity index measure between two images of the same shape.
Higher SSIM indicates more similarity between images (luminence, constrast, structural information), on a 
-1 to 1 scale.
With perfect similarity, SSIM=1.
"""
def ssim(im1, im2):
    # https://en.wikipedia.org/wiki/Structural_similarity
    mean1 = np.mean(im1)
    mean2 = np.mean(im2)
    
    var1 = np.mean((im1 - mean1) ** 2)
    var2 = np.mean((im2 - mean2) ** 2)
    
    cov = np.mean((im1 - mean1) * (im2 - mean2))
    
    L = 255
    k1 = 0.01
    k2 = 0.03
    
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    c3 = c2 / 2
    
    l = ((2 * mean1 * mean2) + c1) / ((mean1 ** 2) + (mean2 ** 2) + c1)
    c = ((2 * var1 * var2) + c2) / ((var1 ** 2) + (var2 ** 2) + c2)
    s = (cov + c3) / ((np.sqrt(var1) * np.sqrt(var2)) + c3)

    error = l * c * s
    
    return error