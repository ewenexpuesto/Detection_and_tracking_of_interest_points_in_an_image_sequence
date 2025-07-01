import cv2
import numpy as np
from numpy.typing import NDArray

def manual_rectangle_tracker(
    frames: list[NDArray[np.uint8]],
    start_pos: tuple[int, int],
    box_size: tuple[int, int] = (120, 70)
) -> list[NDArray[np.uint8]]:
    """
    Manually tracks a rectangle around a user-selected point across frames.

    Parameters
    ----------
    frames : list of np.ndarray
        List of video frames (BGR).
    start_pos : tuple of int
        (x, y) coordinates selected by user in the first frame.
    box_size : tuple
        Size of the rectangle (width, height) to track.

    Returns
    -------
    output_frames : list of np.ndarray
        Frames with tracking rectangle drawn.
    """
    output_frames = []
    x, y = start_pos
    w, h = box_size
    template = frames[0][y:y + h, x:x + w].copy()

    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Run template matching
        res = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)

        # Update rectangle position
        x_new, y_new = max_loc

        # Draw rectangle on a copy
        frame_copy = frame.copy()
        cv2.rectangle(frame_copy, (x_new, y_new), (x_new + w, y_new + h), (0, 255, 0), 2)

        # Update template for next frame (optional for adapting appearance)
        # template = frame[y_new:y_new + h, x_new:x_new + w].copy()

        output_frames.append(frame_copy)

        progress = (i + 1) / len(frames) * 100
        print(f"Tracking: Frame {i + 1}/{len(frames)} ({progress:.1f}%)", end='\r')

    print()
    return output_frames

def CSRT_rectangle_tracker(
    frames: list[NDArray[np.uint8]],
    start_pos: tuple[int, int],
    box_size: tuple[int, int] = (0, 0)
) -> list[NDArray[np.uint8]]:
    """
    Tracks a rectangle across video frames, robust to scale and rotation using CSRT tracker.

    Parameters
    ----------
    frames : list of np.ndarray
        List of video frames (BGR).
    start_pos : tuple of int
        (x, y) coordinates of the top-left corner of the initial rectangle.
    box_size : tuple
        Size of the initial rectangle (width, height).

    Returns
    -------
    output_frames : list of np.ndarray
        List of frames with the tracked rectangle drawn.
    """
    output_frames = []

    # Initial bounding box
    x, y = start_pos
    w, h = box_size
    init_bbox = (x, y, w, h)

    # Create tracker
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frames[0], init_bbox)

    for i, frame in enumerate(frames):
        success, bbox = tracker.update(frame)

        frame_copy = frame.copy()
        if success:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame_copy, "Tracking failure", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        output_frames.append(frame_copy)

        progress = (i + 1) / len(frames) * 100
        print(f"Advanced Tracking: Frame {i + 1}/{len(frames)} ({progress:.1f}%)", end='\r')

    print()
    return output_frames
