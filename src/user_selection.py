import cv2
import numpy as np
from numpy.typing import NDArray

def select_rectangle(frame: NDArray[np.uint8]) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Allows the user to draw a rectangle on a frame using the mouse.
    Returns the top-left position and size of the rectangle.

    Parameters
    ----------
    frame : np.ndarray
        The input image (BGR) to draw on.

    Returns
    -------
    start_pos : tuple of int
        (x, y) of the top-left corner of the selected rectangle.
    box_size : tuple of int
        (width, height) of the selected rectangle.
    """
    clone = frame.copy()
    drawing = False
    ix, iy = -1, -1
    rect = (0, 0, 0, 0)  # x, y, w, h

    def draw_rectangle(event, x, y, flags, param):
        nonlocal drawing, ix, iy, clone, rect

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                clone = frame.copy()
                cv2.rectangle(clone, (ix, iy), (x, y), (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x1, y1 = min(ix, x), min(iy, y)
            x2, y2 = max(ix, x), max(iy, y)
            rect = (x1, y1, x2 - x1, y2 - y1)
            cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.namedWindow("Select Object to Track")
    cv2.setMouseCallback("Select Object to Track", draw_rectangle)

    while True:
        cv2.imshow("Select Object to Track", clone)
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == 32:  # Enter or Space to confirm
            break
        elif key == 27:  # Escape to cancel
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Selection canceled")

    cv2.destroyAllWindows()
    x, y, w, h = rect
    return (x, y), (w, h)
