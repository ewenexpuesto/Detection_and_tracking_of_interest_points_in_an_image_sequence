import cv2
import numpy as np
from scipy.signal import convolve2d
from numpy.typing import NDArray

def local_color_fill_convolution_list_of_arrays(
    frames: list[NDArray[np.uint8]],
    target_color: tuple[int, int, int] = (255, 0, 0),
    v: int = 5,
    w: int = 5,
    threshold: int = 5
) -> list[NDArray[np.uint8]]:
    output_frames = []
    total_frames = len(frames)
    kernel = np.ones((v, w), dtype=np.uint8)

    target = np.array(target_color, dtype=np.uint8)
    white = np.array([255, 255, 255], dtype=np.uint8)

    for idx, frame in enumerate(frames):
        output = frame.copy()

        # Binary mask of target color pixels
        mask = np.all(frame == target, axis=2).astype(np.uint8)

        # Neighborhood count of target pixels
        neighbor_count = convolve2d(mask, kernel, mode='same')

        # Pixels to keep colored (enough neighbors)
        fill_mask = (neighbor_count >= threshold)

        # Apply the original color where condition is met
        output[fill_mask] = target

        # Apply white to target-colored pixels that don't meet threshold
        white_mask = (mask == 1) & (fill_mask == 0)
        output[white_mask] = white

        output_frames.append(output)

        progress = (idx + 1) / total_frames * 100
        print(f"Fast Local Color Fill w/ White Else: Frame {idx + 1}/{total_frames} ({progress:.1f}%)", end='\r')

    print()
    return output_frames
