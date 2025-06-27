import gaussian_blur
import video_processing
import cv2
import numpy as np
from numpy.typing import NDArray

def difference_of_gaussians_array(image: NDArray[np.uint8], kernel_size: int = 5, sigma1: float = 1, sigma2: float = 2) -> NDArray[np.uint8]:
    """
    Computes the Difference of Gaussians (DoG) by subtracting two blurred versions of the image.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale).
    kernel_size : int
        Size of the Gaussian kernel (must be odd). Determines the neighborhood size for the weighted average around each pixel.
    sigma1 : float
        Sigma for the first (smaller) Gaussian blur.
    sigma2 : float
        Sigma for the second (larger) Gaussian blur.

    Returns
    -------
    dog_image : np.ndarray
        Difference of Gaussians image.
    """

    blur1 = gaussian_blur.gaussian_blur_array(image, kernel_size, sigma=sigma1)
    blur2 = gaussian_blur.gaussian_blur_array(image, kernel_size, sigma=sigma2)
    dog = cv2.subtract(blur1, blur2)
    return dog

def difference_of_gaussians_list_of_arrays(video: list[NDArray[np.uint8]], kernel_size: int = 5, sigma1: float = 1, sigma2: float = 2) -> list[NDArray[np.uint8]]:
    """
    Computes the Difference of Gaussians (DoG) for each frame in a list of video frames.

    Parameters
    ----------
    video : list of np.ndarray
        List of video frames. Each frame is a 3D NumPy array (height, width, channels).
    kernel_size : int
        Size of the Gaussian kernel (must be odd).
    sigma1 : float
        Sigma for the first (smaller) Gaussian blur.
    sigma2 : float
        Sigma for the second (larger) Gaussian blur.

    Returns
    -------
    dog_frames : list of np.ndarray
        List of frames with DoG applied (each frame is a 3D array).
    """

    return [difference_of_gaussians_array(frame, kernel_size, sigma1, sigma2) for frame in video]


def difference_of_gaussians_mp4(input_video_path: str, output_folder: str, video_name: str, linear_stretch_colors: bool, extreme_stretch_colors: bool, kernel_size: int = 5, sigma1: float = 1, sigma2: float = 2) -> None:
    """
    Applies Difference of Gaussians to each frame of a video and writes the result to a new video file.

    Parameters
    ----------
    input_video_path : str
        Path to the input video file.
    output_folder : str
        Path to save the processed output folder.
    video_name : str
        Name of the video, ending by mp4.
    linear_stretch_colors : bool
        Put True if you want to accentuate contrast with linear stretch.
    extreme_stretch_colors : bool
        Put True if you want to accentuate contrast to the max.
    kernel_size : int
        Size of the Gaussian kernel (must be odd).
    sigma1 : float
        Sigma for the first (smaller) Gaussian blur.
    sigma2 : float
        Sigma for the second (larger) Gaussian blur.
    """

    # Extract frames and video properties
    frames, fps = video_processing.mp4_to_list_of_arrays(input_video_path)
    # Apply DoG to all frames
    dog_frames = difference_of_gaussians_list_of_arrays(frames, kernel_size, sigma1, sigma2)
    if linear_stretch_colors:
        # Apply linear color stretching to each frame
        dog_frames = [video_processing.linear_stretch_colors(frame) for frame in dog_frames]
    if extreme_stretch_colors:
        # Apply extreme color stretching to each frame
        dog_frames = [video_processing.extreme_stretch_colors(frame) for frame in dog_frames]
    # Write processed frames back to video
    video_processing.list_of_arrays_to_mp4(dog_frames, output_folder, video_name, fps)