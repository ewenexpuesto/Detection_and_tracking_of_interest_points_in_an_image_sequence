import cv2
import os
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

def mp4_to_jpg(video_path: str, output_folder: str) -> int:
    """
    Extracts frames from an MP4 video and saves them as image files.

    Parameters
    ----------
    video_path : str
        Path to the input MP4 video file.
    output_folder : str
        Path to the folder where extracted frames will be saved.

    Returns
    -------
    count : int
        Total number of frames extracted and saved.
    """

    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    frame_count = 0

    while True:
        # Read a frame from the video
        success, frame = video.read()

        # If there are no more frames, stop the loop
        if not success:
            break

        # Construct filename for the output image
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")

        # Save the frame as an image
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    # Release the video object
    video.release()

    print(f"Extracted {frame_count} frames from {video_path}")
    return frame_count

def mp4_to_list_of_arrays(video_path: str) -> Tuple[list[NDArray[np.uint8]], int]:
    """
    Extracts frames from an MP4 video and returns them as a list of NumPy arrays,
    along with the video's frames per second (fps).

    Parameters
    ----------
    video_path : str
        Path to the input MP4 video file.

    Returns
    -------
    frames : list of np.ndarray
        List of video frames. Each frame is a NumPy array of shape (H, W, 3),
        with pixel values in BGR format.
    fps : int
        Frames per second of the video.
    """

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    # Get FPS
    fps = int(video.get(cv2.CAP_PROP_FPS))

    frames = []

    while True:
        # Read a frame from the video
        success, frame = video.read()

        if not success:
            break

        frames.append(frame)

    video.release()

    print(f"Extracted {len(frames)} frames from {video_path} at {fps} fps.")
    return frames, fps


def list_of_arrays_to_mp4(frames: list[NDArray[np.uint8]], folder_path: str, video_name: str = "output.mp4", fps: int = 30) -> str:
    """
    Writes a list of video frames (NumPy arrays) to a video file in the specified folder.

    Parameters
    ----------
    frames : list of np.ndarray
        List of video frames in BGR format.
    folder_path : str
        Path to the folder where the video file will be saved.
    video_name : str, optional
        Name of the output video file (default is "output.mp4").
    fps : int, optional
        Frames per second of the output video (default is 30).

    Returns
    -------
    output_path : str
        Full path to the saved video file.
    """

    if len(frames) == 0:
        raise ValueError("The frame list is empty.")

    # Get frame dimensions from the first frame
    height, width = frames[0].shape[:2]

    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    output_path = os.path.join(folder_path, video_name)

    # Define the codec and create the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i, frame in enumerate(frames):
        if frame.shape[:2] != (height, width):
            raise ValueError(f"Frame at index {i} has different dimensions.")

        # Ensure frame is in uint8 format
        video_writer.write(frame.astype(np.uint8))

    video_writer.release()
    print(f"Video saved to {output_path}")
    return output_path

def list_of_arrays_to_jpgs(frames: list[NDArray[np.uint8]], folder_path: str, prefix: str = "frame") -> list[str]:
    """
    Saves a list of BGR NumPy arrays as individual jpg image files named like 'frame_000.jpg', 'frame_001.jpg', etc., in a specified folder.

    Parameters
    ----------
    frames : list of np.ndarray
        List of image arrays (grayscale or BGR) to be saved.
    folder_path : str
        Path to the output folder where images will be stored.
    prefix : str, optional
        Prefix for the output filenames (default is "frame").

    Returns
    -------
    list of str
        List of full file paths to the saved image files.
    """

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    saved_paths = []
    for i, frame in enumerate(frames):
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        filename = f"{prefix}_{i:03d}.jpg"
        output_path = os.path.join(folder_path, filename)
        success = cv2.imwrite(output_path, frame)
        
        if not success:
            raise IOError(f"Failed to write image to {output_path}")
        
        saved_paths.append(output_path)
        print(f"Saved: {output_path}")
    
    return saved_paths

def linear_stretch_colors(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Linearly stretches pixel intensities in the input image so that the
    minimum value becomes 0 and the maximum becomes 255.

    This enhances contrast by mapping the full input range to [0, 255].

    Parameters
    ----------
    frame : np.ndarray
        Input image array, either 2D (grayscale) or 3D (color).

    Returns
    -------
    stretched_uint8 : np.ndarray
        Output image with pixel values scaled to [0, 255] and dtype uint8.
    """

    # Convert to float for computation to avoid overflow
    frame_float = frame.astype(np.float32)

    # Find min and max values in the frame (across all channels)
    min_val = frame_float.min()
    max_val = frame_float.max()

    # Avoid division by zero if min == max
    if max_val == min_val:
        # If all pixels are the same, return a zeroed array or the original
        return np.zeros_like(frame, dtype=np.uint8)

    # Stretch the values to [0, 255]
    stretched = (frame_float - min_val) * 255.0 / (max_val - min_val)

    # Clip values in case it falls outside of the bonds and convert back to uint8
    stretched_uint8 = np.clip(stretched, 0, 255).astype(np.uint8)

    return stretched_uint8

def extreme_stretch_colors(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Converts an image so all non-zero pixel components become 255 and black remain 0. All pixels are then either pure red, green, blue or any combination of them, including back and white.

    Parameters
    ----------
    frame : NDArray[np.uint8]
        Input 2D image.

    Returns
    -------
    bw_frame : NDArray[np.uint8]
        Binary image with 0 or 255.
    """

    frame_float = frame.astype(np.float32)
    min_val = frame_float.min()
    max_val = frame_float.max()

    if max_val == min_val:
        return np.zeros_like(frame, dtype=np.uint8)

    stretched = (frame_float - min_val) * 255.0 / (max_val - min_val)
    stretched_uint8 = np.clip(stretched, 0, 255).astype(np.uint8)

    # Now, set all non-zero components to 255 (white)
    bw_frame = np.where(stretched_uint8 > 0, 255, 0).astype(np.uint8)

    return bw_frame
