import cv2
import numpy as np
from ultralytics import YOLO
from difflib import get_close_matches
import manual_retangle_tracker

# Load YOLOv8 pretrained model
model = YOLO("yolov8n.pt")

# Known YOLO classes
YOLO_CLASSES = model.names # dict {id: name}
print("HERE ARE THE CLASSES THAT YOLO CAN GUESS: ", YOLO_CLASSES)

def map_text_to_class(query, class_dict):
    """Find closest YOLO class to the user query."""
    classes = list(class_dict.values())
    match = get_close_matches(query.lower(), classes, n=1, cutoff=0.1)
    return match[0] if match else None

def detect_and_draw(frames, query):
    """
    Detect all objects described by query in the first frame and return bounding boxes.
    
    Parameters
    ----------
    frames : list of np.ndarray
        List of video frames.
    query : str
        Natural language description (e.g., "utility vehicle").
    
    Returns
    -------
    list of tuples
        Each element is a tuple (start_pos, box_size), where:
            start_pos : tuple of int
                (x, y) coordinates of the top-left corner of the rectangle.
            box_size : tuple of int
                (width, height) of the rectangle.
        Returns an empty list if no matching objects are found.
    """
    if not frames:
        raise ValueError("Frame list is empty.")

    # Map query to YOLO class
    target_class = map_text_to_class(query, YOLO_CLASSES)
    if target_class is None:
        raise ValueError(f"No matching YOLO class found for '{query}'")

    # Run detection on first frame
    frame = frames[0]
    results = model(frame)

    boxes = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = YOLO_CLASSES[cls_id]
            if cls_name == target_class:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                start_pos = (x1, y1)
                box_size = (x2 - x1, y2 - y1)
                boxes.append((start_pos, box_size))

    return boxes

def detect_objects(frames, target_classes):
    """
    Detect objects in frames using YOLO, highlight them, and overlay descriptions.

    Parameters
    ----------
    frames : list of np.ndarray
        Video frames (BGR images).
    target_classes : list of str
        Target object categories to detect (e.g., ["car", "person"]).

    Returns
    -------
    list of np.ndarray
        Frames with detections drawn.
    """
    if not frames:
        raise ValueError("Frame list is empty.")

    # Map text queries to YOLO classes
    mapped_classes = []
    for query in target_classes:
        match = map_text_to_class(query, YOLO_CLASSES)
        if match:
            mapped_classes.append(match)
        else:
            print(f"Warning: No YOLO class found for '{query}'")

    output_frames = []
    n_frames = len(frames)

    for i, frame in enumerate(frames, start=1):
        results = model(frame)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = YOLO_CLASSES[cls_id]
                conf = float(box.conf[0])

                if cls_name in mapped_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Add label with confidence
                    label = f"{cls_name} ({conf:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 2)

                    # Add description below the box
                    desc = f"Detected: {cls_name}"
                    cv2.putText(frame, desc, (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 2)

        output_frames.append(frame)

        # Progress bar
        progress = int((i / n_frames) * 50)
        bar = "[" + "#" * progress + "-" * (50 - progress) + "]"
        print(f"\rProcessing frames {i}/{n_frames} {bar}", end="")

    print()  # Newline after progress bar
    return output_frames

def YOLO_with_CSRT(frames, target_class):
    """
    Apply YOLO object detection followed by CSRT tracking on the given frames for all objects of category target_class.

    Parameters
    ----------
    frames : list of np.ndarray
        The input video frames.
    target_class : str
        The class of objects to track (e.g., "car", "person").

    Returns
    -------
    list_of_frames_processed : list of np.ndarray
        A list of processed frames with tracked objects.
    """
    boxes = YOLO.detect_and_draw(frames, target_class)
    list_of_frames_processed = []
    for start_pos, box_size in boxes:
        frames_processed = manual_retangle_tracker.CSRT_rectangle_tracker(frames, start_pos, box_size)
        list_of_frames_processed.append(frames_processed)
    return list_of_frames_processed