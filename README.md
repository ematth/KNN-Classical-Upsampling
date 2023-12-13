# Nearest Neighbor Classication for Classical Image Upscaling

> ************************[Evan Matthews](https://github.com/ematth) (evanmm3@illinois.edu), [Nicolas Prate](https://github.com/nprate2) (nprate2@illinois.edu)************************
> ************************Department of Computer Science, Grainger College of Engineering, University of Illinois Urbana-Champaign************************

---

## Objective

Given a set of ordered pixel data in the form of an image, our goal is to perform *upsampling* on the data such that: 

- the resulting resolution is improved by some factor,
- the final result passes the “human test,” having added new, believable, and realistic information and detail to the image,
- the time complexity for upscaling is relatively close to that of lossy upscaling implementations.

---

## Collaboration

We are informally collaborating with [Mark Bauer](https://github.com/MarkBauer5) (*markb5@illinois.edu*) and [Quinn Ouyang](https://github.com/quinnouyang) (*qouyang3@illinois.edu*) to accomplish the same problem but with different approaches. In contrast to Mark and Quinn’s learning-based approach to intelligently upsample given prior knowledge, we are taking a more classical approach in which we attempt to upsample given only direct information from the input.

---

## Proposed Problem and Elaborations:

### KNN Interpolation

Our baseline approach is to perform K-Nearest-Neighbors on our image such that nearby pixels are analyzed to make a reasonable guess for each missing pixel that results from upscaling. This is the simplest approach to solve the problem of image upsampling - filling in new pixels with the average of its K-nearest neighboring pixel values. Additionally for this approach, we will support only fixed aspect-ratios between original and upscaled images, which equates the baseline functionality to “deblurring” an image.

Despite being a baseline approach, it will still pose some interesting implementation challenges. Depending on the size of the input image and desired size of the image, there will be a varying number of pixels in various patterns requiring interpolation. For example, if the desired output size is large enough compared to the input image, there may be ‘seas’ of empty pixels that need filling; with too small a K, there may be no original pixels in the neighborhood to consider for interpolation. Additionally, interpolation around the corners and borders of the image provide a smaller neighborhood to work with; there are various approaches to overcome this obstacle and we will have to experiment to determine what works best for this problem.

### Upscaling for Dynamic Aspect Ratios

Given that image upscaling is not strictly limited to the image’s original aspect ratio, an additional challenge to consider would be interpolation with respect to a *dynamic* aspect ratio. This is a generalization of the problem which considers the flexibility of up-scaling as a creative tool, and this also raises a more general issue of which nearest-neighbors to choose for pixel prediction. In a sense, we would ideally not want to rely on a constant aspect ratio when making predictions about neighboring pixels. For example, an image whose height remains constant but width doubles should ideally interpolate only on horizontal nearest-neighbors and stretch the image while maintaining its vertical content. This elaboration would also increase robustness compared to the baseline, allowing ideal upscaling for odd-width or odd-height images and the ability to upscale images to resolutions not accessible to the original aspect ratio.

### Selective Upscaling

In the sense of time complexity, we expect a more rigorous upscaling algorithm to take more time to handle unknown pixels, and this runtime should only get worse for higher-resolution inputs. As such, we would like to consider a means of “selectively upscaling” inputs as to drastically cut down on runtime. For instance, if our input was an image of a face on a perfect, solid-colored background, we would expect selective upscaling to focus most of the runtime on upscaling the face, being able to avoid repeatedly calculating the same pixel value with KNN interpolation across the background. While this elaboration may not be practical for all images, we would expect selective upscaling to work reasonably well on real-life photos and realistic images - inputs where large, solid-color regions or regions with noticeable color patterns are most present.

This, of course, would demand an increase in overhead and complexity to detect regions where selective upscaling could be appropriate in an image, but our belief is that this overhead will be outweighed by the runtime saved in KNN calculations. We may leverage concepts such as edge detection and color-gradient calculations to assist in the distinction between regions with more ‘constant’ pixel values and more detail regions that require our KNN interpolation. Additionally, it will be important to tune the size of region required to trigger our selective upscaling functionality; too small a threshold may result in grainy or patchy results with our KNN interpolation applied nowhere, while too large a threshold may result in the entire image being upscaled with KNN interpolation in addition to the extra overhead we’ve introduced in attempting to detect distinct regions. Our goal is to find some distinction threshold that both decreases runtime on applicable images and maintains the quality produced by our previous approaches.

---

## Metrics and Testing

We will begin with a large set of images. Copies of these images will be downscaled at various stepping factors, creating a cascade of progressively smaller-resolution images for each image in the set. We can then apply our algorithm in order to upsample these images back into their original sizes. Given the original images and their upscaling reconstruction, we can apply various metrics to measure similarity. Such metrics may include: raw pixel comparison, MSE, RMSE, MAE, PSNR, and SSIM. 

- Fundamentally, we will consider raw pixel comparison, which would calculate the RGB differences between the pixels of the original image and our upscaled reconstruction, given that both images have the exact same dimensions. While a theoretically ideal upscaling implementation would return an exact copy of the original image, we only expect our results to *minimize* this metric since a true recovery of lost information is impossible in the classical sense.
- MSE measures the average squared difference between pixels values between the images, MAE measures the average absolute difference, and RMSE measures the square root of MSE. MSE may not align with human perception, MAE is less sensitive to outliers than MSE, and RMSE is in the same unit as the pixel values, perhaps making it more interpretable than MSE. For all MSE, MAE, and RMSE, a lower value indicates a closer similarity between images and thus a better reconstruction.
- PSNR measures the ratio between the maximum value of an image and the power of corrupting noise, with higher values indicating a higher quality of reconstruction.  SSIM measures differences in structural information, texture, and luminance, with higher values indicating a higher quality of reconstruction and the maximum value of ‘1’ representing identical images. Between PSNR and SSIM, SSIM is more likely to align with human perception.

By leveraging multiple metrics, we should be able to gain various perspectives and insights into the nature of differences between the original images and our reconstructions, providing means to better visualize the strengths and weaknesses of our approaches.