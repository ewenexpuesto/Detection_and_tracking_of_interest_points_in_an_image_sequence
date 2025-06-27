import cv2
import numpy as np
from numpy.typing import NDArray
from typing import List
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte


def kadir_brady_saliency_list_of_arrays(frames: List[NDArray[np.uint8]], radius: int = 5, scale: float = 0.25) -> List[NDArray[np.uint8]]:
    """
    Fast approximation of Kadir-Brady saliency using entropy. Downscales to improve speed.

    Parameters
    ----------
    frames : list of np.ndarray
        List of BGR video frames.
    radius : int
        Radius of the disk-shaped neighborhood for entropy.
    scale : float
        Resize factor for faster entropy computation (e.g., 0.25 = 25% size).

    Returns
    -------
    output_frames : list of np.ndarray
        List of saliency-overlaid BGR frames.
    """
    output_frames = []
    total_frames = len(frames)
    selem = disk(radius)

    for i, frame in enumerate(frames):
        # Resize down for faster processing
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Conversion to grayscale in order to compute entropy
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray_ubyte = img_as_ubyte(gray)

        # Compute entropy
        entropy_map = entropy(gray_ubyte, footprint=selem)

        # Resize entropy map back to original size
        entropy_up = cv2.resize(entropy_map, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Normalize and color
        norm = cv2.normalize(entropy_up, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

        blended = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
        output_frames.append(blended)

        # Progress
        progress = (i + 1) / total_frames * 100
        print(f"Kadir-Brady Saliency (downscaled): Frame {i + 1}/{total_frames} ({progress:.1f}%)", end='\r')

    print()
    return output_frames
