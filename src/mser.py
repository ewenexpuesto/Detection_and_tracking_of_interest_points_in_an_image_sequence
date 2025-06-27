import cv2
import numpy as np
from numpy.typing import NDArray

def mser_list_of_arrays(frames: list[NDArray[np.uint8]]) -> list[NDArray[np.uint8]]:
    """
    Detects MSER (Maximally Stable Extremal Regions) in each frame and overlays them.

    Parameters
    ----------
    frames : list of np.ndarray
        List of BGR video frames.

    Returns
    -------
    output_frames : list of np.ndarray
        Frames with MSER regions drawn.
    """
    output_frames = []
    total_frames = len(frames)
    mser = cv2.MSER_create()

    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect MSER regions
        regions, _ = mser.detectRegions(gray)

        # Create an overlay
        overlay = frame.copy()
        for region in regions:
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            cv2.polylines(overlay, [hull], True, (0, 0, 255), 1)

        # Blend with original image
        blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        output_frames.append(blended)

        # Show progress
        progress = (i + 1) / total_frames * 100
        print(f"MSER Detection: Processing frame {i + 1}/{total_frames} ({progress:.1f}%)", end='\r')

    print()  # Newline after progress
    return output_frames
