import numpy as np
import cv2
from numpy.typing import NDArray

def difference_of_intensity_of_superpixels_list_of_arrays(frames: NDArray[np.uint8], p: float = 0.01, t: float = 0.1) -> NDArray[np.uint8]:
    """
    Process video frames to detect motion using superpixel comparison.

    Parameters
    ----------
    frames: list of np.ndarray
        List of numpy arrays representing video frames.
    p: float
        Fraction of frame size for superpixel dimensions (0 < p < 1).
    t: float
        Threshold percentage for intensity change detection (0 < t < 1).

    Returns
    -------
    processed_frames: list of np.ndarray
        List of processed frames with motion highlighted in bright green.
    """

    if len(frames) < 2:
        return frames
    
    processed_frames = []
    total_frames = len(frames) - 1
    
    for i in range(len(frames) - 1):
        progress = (i + 1) / total_frames * 100
        print(f"Processing frame {i + 1}/{total_frames} ({progress:.1f}%)", end='\r')
        current_frame = frames[i]
        next_frame = frames[i + 1]
        
        # Create output frame
        output_frame = current_frame.copy()
        
        # Get frame dimensions
        height, width = current_frame.shape[:2]
        
        # Calculate superpixel dimensions
        superpixel_height = max(1, int(height * p))
        superpixel_width = max(1, int(width * p))
        
        # Process each superpixel
        for y in range(0, height, superpixel_height):
            for x in range(0, width, superpixel_width):
                # Define superpixel boundaries
                y_end = min(y + superpixel_height, height)
                x_end = min(x + superpixel_width, width)
                
                # Extract superpixel regions from both frames
                current_superpixel = current_frame[y:y_end, x:x_end]
                next_superpixel = next_frame[y:y_end, x:x_end]
                
                # Calculate mean intensities
                if len(current_superpixel.shape) == 3:
                    current_mean = np.mean(current_superpixel)
                    next_mean = np.mean(next_superpixel)
                else:
                    current_mean = np.mean(current_superpixel)
                    next_mean = np.mean(next_superpixel)
                
                # Calculate threshold based on current intensity
                threshold = current_mean * t
                
                # Check if intensity change exceeds threshold
                intensity_diff = abs(next_mean - current_mean)
                
                if intensity_diff > threshold:
                    # Color superpixel in bright green
                    output_frame[y:y_end, x:x_end] = [255, 0, 0]
        
        processed_frames.append(output_frame)
    
    # Add the last frame without processing (no next frame to compare)
    if len(frames[-1].shape) == 2:
        last_frame = cv2.cvtColor(frames[-1], cv2.COLOR_GRAY2RGB)
    else:
        last_frame = frames[-1].copy()
    processed_frames.append(last_frame)
    
    return processed_frames

import numpy as np
import cv2
from numpy.typing import NDArray
from typing import List

def difference_of_intensity_superpixels_matrix(
    frames: List[NDArray[np.uint8]],
    p: float = 0.01,
    t: float = 0.1
) -> List[NDArray[np.uint8]]:
    """
    Vectorized superpixel-based frame difference using matrix ops (no explicit loops over blocks).

    Parameters
    ----------
    frames : list of np.ndarray
        List of BGR video frames.
    p : float
        Superpixel size ratio (0 < p < 1).
    t : float
        Threshold ratio for detecting motion (0 < t < 1). Higher means less detection.

    Returns
    -------
    List of np.ndarray
        Frames with motion regions highlighted in green.
    """
    if len(frames) < 2:
        return frames

    processed_frames = []
    h, w = frames[0].shape[:2]
    sp_h = max(1, int(h * p))
    sp_w = max(1, int(w * p))

    # Generate superpixel ID matrix
    num_blocks_y = (h + sp_h - 1) // sp_h
    num_blocks_x = (w + sp_w - 1) // sp_w
    block_id = np.zeros((h, w), dtype=np.int32)

    block_idx = 0
    for y in range(0, h, sp_h):
        for x in range(0, w, sp_w):
            block_id[y:y+sp_h, x:x+sp_w] = block_idx
            block_idx += 1
    n_blocks = block_idx
    block_id_flat = block_id.reshape(-1)

    for i in range(len(frames) - 1):
        print(f"Processing frame {i+1}/{len(frames)-1}...", end='\r')

        # Convert to grayscale and vectorize
        f1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32).reshape(-1)
        f2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY).astype(np.float32).reshape(-1)

        # Compute mean intensity per superpixel using scatter-add logic
        counts = np.bincount(block_id_flat, minlength=n_blocks).reshape(-1, 1)  # shape: (num_blocks, 1)
        sum1 = np.bincount(block_id_flat, weights=f1, minlength=n_blocks).reshape(-1, 1)
        sum2 = np.bincount(block_id_flat, weights=f2, minlength=n_blocks).reshape(-1, 1)

        mean1 = sum1 / counts
        mean2 = sum2 / counts

        # Compute absolute difference per block
        diff = np.abs(mean2 - mean1)
        threshold = mean1 * t
        changed_blocks = (diff > threshold).flatten()  # shape: (n_blocks,)

        # Create mask per pixel using changed block IDs
        changed_mask = changed_blocks[block_id_flat].reshape(h, w)

        # Generate output frame: red if change detected, else original
        out_frame = frames[i].copy()
        out_frame[changed_mask] = [0, 0, 255]

        processed_frames.append(out_frame)

    # Append the last frame as-is
    last_frame = frames[-1]
    if len(last_frame.shape) == 2:
        last_frame = cv2.cvtColor(last_frame, cv2.COLOR_GRAY2BGR)
    processed_frames.append(last_frame)

    return processed_frames
