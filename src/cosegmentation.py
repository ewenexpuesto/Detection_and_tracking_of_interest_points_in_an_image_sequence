import cv2
import numpy as np
from numpy.typing import NDArray

def kmeans_segmentation_list_of_arrays(
    frames: list[NDArray[np.uint8]],
    k: int = 2,
    keep_only_cluster: int = -1  # -1 means keep all clusters
) -> list[NDArray[np.uint8]]:
    """
    Applies K-means color segmentation to each frame.

    Parameters
    ----------
    frames : list of np.ndarray
        List of BGR frames.
    k : int
        Number of color clusters to segment into.
    keep_only_cluster : int
        If >= 0, only keeps this cluster (others turned black).

    Returns
    -------
    output_frames : list of np.ndarray
        List of frames with clustered segmentation.
    """
    output_frames = []
    total_frames = len(frames)

    for i, frame in enumerate(frames):
        Z = frame.reshape((-1, 3)).astype(np.float32)

        # Define criteria and apply KMeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        segmented = centers[labels.flatten()].reshape(frame.shape)

        if keep_only_cluster >= 0 and keep_only_cluster < k:
            # Mask out all other clusters (set to black)
            mask = (labels.flatten() == keep_only_cluster).reshape(frame.shape[:2])
            segmented = np.zeros_like(frame)
            segmented[mask] = centers[keep_only_cluster]

        output_frames.append(segmented)

        progress = (i + 1) / total_frames * 100
        print(f"K-means Segmentation: Frame {i + 1}/{total_frames} ({progress:.1f}%)", end='\r')

    print()
    return output_frames
