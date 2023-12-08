import numpy as np
from functions.utility import dimcheck


"""
Calculates the mean squared difference per pixel between two images of the same shape.
Lower MSE indicates more similar images.
With perfect similarity, MSE=0.
"""
dim: int = 0 # placeholder value
@dimcheck
def mse(im1: np.ndarray((dim, dim, 3), dtype=float), im2: np.ndarray((dim, dim, 3), dtype=float)) -> float:
    error: float = np.sum((im1.astype(float) - im2.astype(float)) ** 2)
    error /= im1.shape[0] * im1.shape[1]
    return error


"""
Calculates the root mean squared difference per pixel between two images of the same shape.
Lower RMSE indicates more similar images.
With perfect similarity, RMSE=0.
"""
@dimcheck
def rmse(im1: np.ndarray((dim, dim, 3)), im2: np.ndarray((dim, dim, 3))) -> float:
    error: float = np.sqrt(mse(im1, im2))
    return error


"""
Calculate the mean absolute difference per pixel between two images of the same shape.
Lower MAE indicates more similar images.
With perfect similarity, MAE=0.
"""
@dimcheck
def mae(im1: np.ndarray((dim, dim, 3)), im2: np.ndarray((dim, dim, 3))) -> float:
    error: float = np.sum(np.abs(im1.astype(float) - im2.astype(float)))
    error /= im1.shape[0] * im1.shape[1]  
    return error


"""
Calculate the peak signal to noise ratio between two images of the same shape.
Higher PSNR indicates higher-quality reconstruction (generally in the range of 30-50Db).
With perfect similarity, PSNR=infinity.
Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
"""
@dimcheck
def psnr(im1: np.ndarray((dim, dim, 3)), im2: np.ndarray((dim, dim, 3))) -> float:
    np.seterr(divide='ignore') # TODO: Fix Log10 divide warning!
    error: float = (20 * np.log10(255)) - (10 * np.log10(mse(im1, im2)))
    return error


"""
Calculate the structural similarity index measure between two images of the same shape.
Higher SSIM indicates more similarity between images (luminence, constrast, structural information), on a 
-1 to 1 scale.
With perfect similarity, SSIM=1.
"""
@dimcheck
def ssim(im1: np.ndarray((dim, dim, 3)), im2: np.ndarray((dim, dim, 3))) -> float:
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

    error: float = l * c * s
    return error


"""
Perform all metric calculations (MSE, RMSE, MAE, PSNR, SSIM), printing the results to a table
"""
@dimcheck
def all_metrics(im1: np.ndarray((dim, dim, 3), float), im2: np.ndarray((dim, dim, 3), float)) -> None:
    metric = [mse, rmse, mae, psnr, ssim]
    metric_label = ['MSE', 'RMSE', 'MAE', 'PSNR', 'SSIM']
    for i in range(len(metric)):
        print(f'{metric_label[i]}: {metric[i](im1, im2)}')
    print('\n')
    return