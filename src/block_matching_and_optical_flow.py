import cv2
import numpy as np
from numpy.typing import NDArray

def block_matching_list_of_arrays(
    frames: list[NDArray[np.uint8]],
    block_size: int = 16,
    search_radius: int = 8,
    step: int = 16
) -> list[NDArray[np.uint8]]:
    """
    Applies block matching motion estimation between consecutive frames and visualizes motion vectors.

    Parameters
    ----------
    frames : list of np.ndarray
        List of BGR frames.
    block_size : int
        Size of square blocks to match.
    search_radius : int
        Radius (in pixels) to search for block matches.
    step : int
        Step between blocks (set equal to block_size for non-overlapping).

    Returns
    -------
    output_frames : list of np.ndarray
        Original frames with motion vectors drawn.
    """
    output_frames = []
    total_frames = len(frames)

    for i in range(len(frames) - 1):
        frame1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        overlay = frames[i].copy()

        h, w = frame1.shape

        for y in range(0, h - block_size + 1, step):
            for x in range(0, w - block_size + 1, step):
                block = frame1[y:y + block_size, x:x + block_size]

                best_match = (0, 0)
                min_diff = float('inf')

                for dy in range(-search_radius, search_radius + 1):
                    for dx in range(-search_radius, search_radius + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h - block_size + 1 and 0 <= nx < w - block_size + 1:
                            candidate = frame2[ny:ny + block_size, nx:nx + block_size]
                            diff = np.sum(np.abs(block - candidate))
                            if diff < min_diff:
                                min_diff = diff
                                best_match = (dx, dy)

                # Draw motion vector (arrow)
                start = (x + block_size // 2, y + block_size // 2)
                end = (start[0] + best_match[0], start[1] + best_match[1])
                cv2.arrowedLine(overlay, start, end, (0, 0, 255), 1, tipLength=0.3)

        output_frames.append(overlay)

        progress = (i + 1) / (total_frames - 1) * 100
        print(f"Block Matching: Processed frame {i + 1}/{total_frames - 1} ({progress:.1f}%)", end='\r')

    # Append last frame unmodified
    output_frames.append(frames[-1])
    print()
    return output_frames



def optical_flow_farneback_list_of_arrays(frames: list[NDArray[np.uint8]]) -> list[NDArray[np.uint8]]:
    """
    Applies fast dense optical flow (Farneback) to visualize motion between frames.

    Parameters
    ----------
    frames : list of np.ndarray
        List of BGR frames.

    Returns
    -------
    output_frames : list of np.ndarray
        Frames with motion vectors visualized as color overlays.
    """
    output_frames = []
    total_frames = len(frames)

    for i in range(len(frames) - 1):
        prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)

        # Compute dense optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # Convert flow to HSV image for visualization
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(frames[i])
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Blend flow with original frame
        blended = cv2.addWeighted(frames[i], 0.6, flow_rgb, 0.4, 0)
        output_frames.append(blended)

        progress = (i + 1) / (total_frames - 1) * 100
        print(f"Farneback Optical Flow: Frame {i + 1}/{total_frames - 1} ({progress:.1f}%)", end='\r')

    # Add last frame unmodified
    output_frames.append(frames[-1])
    print()
    return output_frames

def optical_flow_farneback_compression_list_of_arrays(frames: list[NDArray[np.uint8]], scale: float = 0.5, mag_threshold: float = 1.0) -> list[NDArray[np.uint8]]:
    """
    Fast and cleaner Farneback optical flow with noise suppression and frame downscaling.

    Parameters
    ----------
    frames : list of BGR np.ndarray
        Input frames.
    scale : float
        Resize factor for faster computation (0.5 = 50% size).
    mag_threshold : float
        Threshold for motion magnitude to suppress noise.

    Returns
    -------
    output_frames : list of np.ndarray
        Frames with flow visualized as color overlays.
    """
    output_frames = []
    total_frames = len(frames)

    for i in range(len(frames) - 1):
        # Resize down
        small1 = cv2.resize(frames[i], (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        small2 = cv2.resize(frames[i + 1], (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        prev_gray = cv2.cvtColor(small1, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise before computing flow
        # prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
        # next_gray = cv2.GaussianBlur(next_gray, (5, 5), 0)

        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # Get magnitude and angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Threshold out small motion
        mask = mag > mag_threshold

        # HSV representation
        hsv = np.zeros_like(small1)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag * mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Resize back to original frame size
        flow_color_up = cv2.resize(flow_color, (frames[i].shape[1], frames[i].shape[0]), interpolation=cv2.INTER_LINEAR)

        # Blend
        blended = cv2.addWeighted(frames[i], 0.6, flow_color_up, 0.4, 0)
        output_frames.append(blended)

        progress = (i + 1) / (total_frames - 1) * 100
        print(f"Optical Flow (fast/clean): Frame {i + 1}/{total_frames - 1} ({progress:.1f}%)", end='\r')

    output_frames.append(frames[-1])
    print()
    return output_frames
