import cv2
import numpy as np
from numpy.typing import NDArray
import video_processing
import difference_of_gaussians

def SIFT_list_of_arrays(frames: list[NDArray[np.uint8]], nfeatures: int = 0, nOctaveLayers: int = 3, contrastThreshold: float = 0.04, edgeThreshold: int = 10, sigma: float = 1.6) -> list[NDArray[np.uint8]]:
    """
    Detects and tracks SIFT keypoints across a list of video frames using optical flow.

    Parameters
    ----------
    frames : list of np.ndarray
        List of BGR frames (NumPy arrays).
    nfeatures : int
        Number of features. Default is 0 (no limit).
    nOctaveLayers : int
        Default is 3.
    contrastThreshold : float
        Controls filtering of low-contrast features. Higher values discard more weak features. Default is 0.04.
    edgeThreshold : float
        Suppresses responses along edges (which are unstable). Higher value allows more edge-like features. Default is 10.
    sigma : float
        How much blur is applied in Gaussian blur.

    Returns
    -------
    output_frames : list of np.ndarray
        Frames with tracked keypoints visualized.
    """

    if len(frames) == 0:
        return []

    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        nOctaveLayers=nOctaveLayers,
        contrastThreshold=contrastThreshold,
        edgeThreshold=edgeThreshold,
        sigma=sigma
    )

    output_frames = []

    # First frame SIFT detection
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    keypoints, _ = sift.detectAndCompute(prev_gray, None)

    if not keypoints:
        # No keypoints found, return original frames
        return frames

    prev_pts = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)

    # Draw first frame keypoints
    first_frame_with_kp = cv2.drawKeypoints(frames[0], keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    output_frames.append(first_frame_with_kp)

    for i in range(1, len(frames)):
        curr_frame = frames[i]
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        if prev_pts is None or len(prev_pts) == 0:
            output_frames.append(curr_frame.copy())
            continue

        # Optical flow tracking
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        if next_pts is None or status is None:
            output_frames.append(curr_frame.copy())
            continue

        good_pts = next_pts[status.flatten() == 1]

        frame_with_kp = curr_frame.copy()
        for pt in good_pts:
            x, y = pt.ravel()
            cv2.circle(frame_with_kp, (int(x), int(y)), 3, (0, 255, 0), -1)

        output_frames.append(frame_with_kp)

        # Prepare for next frame
        prev_gray = curr_gray
        prev_pts = good_pts.reshape(-1, 1, 2)

    return output_frames

def SIFT_difference_of_gaussians_mp4(input_video_path: str, output_video_path: str, video_name: str, kernel_size: int, sigma1: float, sigma2: float) -> None:
    """
    Detects and tracks SIFT keypoints across frames of a video after processing it with difference of gaussians and writes the result to a new video file.

    Parameters
    ----------
    input_video_path : str
        Path to the input video file.
    output_video_path : str
        Path to the folder to save the output video file.
    video_name : str
        Name of the output video file (must include the extension).
    kernel_size : int
        Size of the Gaussian kernel (must be odd).
    sigma1 : float
        Sigma for the first (smaller) Gaussian blur.
    sigma2 : float
        Sigma for the second (larger) Gaussian blur.
    """

    frames, fps = video_processing.mp4_to_list_of_arrays(input_video_path)
    dog_frames = difference_of_gaussians.difference_of_gaussians_list_of_arrays(frames, kernel_size, sigma1, sigma2)
    output_frames = SIFT_list_of_arrays(dog_frames)
    # linear strech colors for each frame
    for i in range(len(output_frames)):
        output_frames[i] = cv2.normalize(output_frames[i], None, 0, 255, cv2.NORM_MINMAX)
    video_processing.list_of_arrays_to_mp4(output_frames, output_video_path, video_name=video_name, fps=fps)
