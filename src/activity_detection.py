import cv2
import numpy as np
from numpy.typing import NDArray

def saliency_heatmap_list_of_arrays(frames: list[NDArray[np.uint8]]) -> list[NDArray[np.uint8]]:
    """
    Applies static saliency detection to each frame and returns a heatmap overlay.
    """
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    output_frames = []

    total_frames = len(frames)
    for i, frame in enumerate(frames):
        success, saliency_map = saliency.computeSaliency(frame)
        if not success:
            output_frames.append(frame)
            continue

        saliency_map = (saliency_map * 255).astype(np.uint8)
        saliency_colored = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame, 0.6, saliency_colored, 0.4, 0)
        output_frames.append(blended)

        # Print live progress
        progress = (i + 1) / total_frames * 100
        print(f"Saliency Detection: Processing frame {i + 1}/{total_frames} ({progress:.1f}%)", end='\r')

    print()  # Move to the next line after finishing
    return output_frames

def keep_most_red_pixels(frames: list[NDArray[np.uint8]], p: float) -> list[NDArray[np.uint8]]:
    """
    Keeps only the top p proportion of red-dominant pixels in each frame.
    All other pixels are set to black.

    Parameters
    ----------
    frames : list of np.ndarray
        List of BGR frames (3-channel uint8).
    p : float
        Proportion (0 < p <= 1) of most reddish pixels to keep.

    Returns
    -------
    output_frames : list of np.ndarray
        Frames with only top-p red pixels kept.
    """

    if not (0 < p <= 1):
        raise ValueError("p must be between 0 and 1")

    output_frames = []

    for frame in frames:
        # Compute "redness" score: red - max(blue, green)
        blue = frame[:, :, 0].astype(np.int16)
        green = frame[:, :, 1].astype(np.int16)
        red = frame[:, :, 2].astype(np.int16)

        redness = red - np.maximum(blue, green)

        # Flatten and sort to get threshold
        threshold = np.percentile(redness, 100 * (1 - p))

        # Create mask of top-p red pixels
        mask = redness >= threshold
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # Apply mask: keep red pixels, set others to black
        filtered_frame = np.where(mask_3d, frame, 0).astype(np.uint8)
        output_frames.append(filtered_frame)

    return output_frames

def background_subtraction_list_of_arrays(frames: list[NDArray[np.uint8]]) -> list[NDArray[np.uint8]]:
    """
    Applies background subtraction to highlight moving areas in each frame.
    """
    fgbg = cv2.createBackgroundSubtractorMOG2()
    output_frames = []

    total_frames = len(frames)
    for i, frame in enumerate(frames):
        fg_mask = fgbg.apply(frame)
        fg_colored = cv2.applyColorMap(fg_mask, cv2.COLORMAP_HOT)
        blended = cv2.addWeighted(frame, 0.6, fg_colored, 0.4, 0)
        output_frames.append(blended)

        # Print live progress
        progress = (i + 1) / total_frames * 100
        print(f"Background Subtraction: Processing frame {i + 1}/{total_frames} ({progress:.1f}%)", end='\r')

    print()  # Move to the next line after finishing
    return output_frames
