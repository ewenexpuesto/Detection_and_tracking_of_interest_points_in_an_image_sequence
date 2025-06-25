import cv2

def apply_gaussian_blur(image, kernel_size, sigma):
    """
    Applies Gaussian blur to an image using specified kernel size and sigma.

    Parameters:
    - image (np.ndarray): Input image (grayscale or color).
    - kernel_size (int): Size of the Gaussian kernel (must be odd). It determines how many neighboring pixels are considered when computing the weighted average around each pixel.
    - sigma (float): Standard deviation of the Gaussian.

    Returns:
    - np.ndarray: Blurred image.
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)