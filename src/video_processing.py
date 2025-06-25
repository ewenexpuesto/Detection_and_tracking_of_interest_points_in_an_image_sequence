import cv2
import os
import numpy as np
from numpy.typing import NDArray

def video_to_frames_write(video_path, output_folder):
    """
    Extracts frames from an MP4 video and saves them as image files.

    Parameters:
    - video_path (str): Path to the input MP4 video file.
    - output_folder (str): Path to the folder where extracted frames will be saved.

    Returns:
    - int: Total number of frames extracted and saved.
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

def video_to_frame_arrays(video_path):
    """
    Extracts frames from an MP4 video and returns them as a list of NumPy arrays.

    Parameters:
    - video_path (str): Path to the input MP4 video file.

    Returns:
    - List[np.ndarray]: List of video frames. Each frame is a NumPy array of shape (H, W, 3),
      with pixel values in BGR format.
    """
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    frames = []

    while True:
        # Read a frame from the video
        success, frame = video.read()

        # If there are no more frames, stop the loop
        if not success:
            break

        # Append the NumPy array directly
        frames.append(frame)

    # Release the video object
    video.release()

    print(f"Extracted {len(frames)} frames from {video_path}")
    return frames

def frames_array_to_video_write(frames, folder_path, video_name="output.mp4", fps=30):
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
    str
        Full path to the saved video file.
    """

    if not frames:
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

def save_frames_array(frames: list[np.ndarray], folder_path: str, prefix: str = "frame"):
    """
    Saves a list of grayscale or BGR image frames as individual image files
    named like 'frame_000.jpg', 'frame_001.jpg', etc., in a specified folder.

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

    # Clip values just in case and convert back to uint8
    stretched_uint8 = np.clip(stretched, 0, 255).astype(np.uint8)

    return stretched_uint8

def extreme_stretch_colors(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Converts an image so all non-zero pixels become 255 and black remain 0. All pixels are then either pure red, green, blue or any combination of them, including back and white.

    Parameters
    ----------
    frame : NDArray[np.uint8]
        Input 2D image.

    Returns
    -------
    bw_frame : NDArray[np.uint8]
        Binary image with 0 or 255.
    """

    # Convert to float for computation to avoid overflow
    frame_float = frame.astype(np.float32)
    min_val = frame_float.min()
    max_val = frame_float.max()

    if max_val == min_val:
        return np.zeros_like(frame, dtype=np.uint8)

    stretched = (frame_float - min_val) * 255.0 / (max_val - min_val)
    stretched_uint8 = np.clip(stretched, 0, 255).astype(np.uint8)

    # Now, set all non-zero pixels to 255 (white)
    bw_frame = np.where(stretched_uint8 > 0, 255, 0).astype(np.uint8)

    return bw_frame
