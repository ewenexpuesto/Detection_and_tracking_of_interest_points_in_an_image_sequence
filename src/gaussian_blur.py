import cv2
import numpy as np
from numpy.typing import NDArray

def gaussian_blur_array(image: NDArray[np.uint8], kernel_size: int, sigma: float) -> NDArray[np.uint8]:
    """
    Applies Gaussian blur to an image using specified kernel size and sigma.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or color).
    kernel_size : int
        Size of the Gaussian kernel (must be odd). Determines the neighborhood size for the weighted average around each pixel. Higher values result in more blurring with farther neighbours but it becomes more computationally expensive.
    sigma : float
        Standard deviation of the Gaussian kernel. Higher values result in more blurring with farther neighbours.

    Returns
    -------
    blurred_image : np.ndarray
        Blurred image.
    """

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def gaussian_blur_list_of_arrays(video: list[NDArray[np.uint8]], kernel_size: int, sigma: float) -> list[NDArray[np.uint8]]:
    """
    Applies Gaussian blur to each frame of a video.

    Parameters
    ----------
    video : list of np.ndarray
        List of frames. Each frame is a 3D array (height, width, channels).
    kernel_size : int
        Size of the Gaussian kernel (must be odd).
    sigma : float
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    blurred_frames : list of np.ndarray
        List of blurred frames (each frame is a 3D array).
    """

    return [gaussian_blur_array(frame, kernel_size, sigma) for frame in video]
