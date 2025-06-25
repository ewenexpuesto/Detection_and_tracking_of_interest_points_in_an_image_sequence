import gaussian_blur
import cv2

def compute_difference_of_gaussians(image, kernel_size, sigma1, sigma2):
    """
    Computes the Difference of Gaussians (DoG) by subtracting two blurred versions of the image.

    Parameters:
    - image (np.ndarray): Input image (grayscale).
    - kernel_size (int): Size of the Gaussian kernel (must be odd). It determines how many neighboring pixels are considered when computing the weighted average around each pixel.
    - sigma1 (float): Sigma for the first (smaller blur).
    - sigma2 (float): Sigma for the second (larger blur).

    Returns:
    - np.ndarray: Difference of Gaussians image.
    """
    blur1 = gaussian_blur.apply_gaussian_blur(image, kernel_size, sigma=sigma1)
    blur2 = gaussian_blur.apply_gaussian_blur(image, kernel_size, sigma=sigma2)
    dog = cv2.subtract(blur1, blur2)
    return dog