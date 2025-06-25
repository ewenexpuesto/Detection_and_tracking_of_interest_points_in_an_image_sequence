import cv2
import numpy as np
import os

def track_objects_with_stif(video_path: str, output_path: str = "tracked_output.avi"):
    cap = cv2.VideoCapture(video_path)

    feature_params = dict(maxCorners=30,
                          qualityLevel=0.5,
                          minDistance=10,
                          blockSize=7)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, old_frame = cap.read()
    if not ret:
        raise ValueError("Failed to read video")

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    brightness_mask = np.zeros_like(old_gray)
    brightness_mask[old_gray > 30] = 255

    edges = cv2.Canny(old_gray, threshold1=50, threshold2=150)
    combined_mask = cv2.bitwise_and(edges, brightness_mask)

    p0 = cv2.goodFeaturesToTrack(old_gray, mask=combined_mask, **feature_params)

    mask = np.zeros_like(old_frame)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    height, width = old_frame.shape[:2]
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(frame_gray, threshold1=50, threshold2=150)  # current frame's edge map

        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            if p1 is not None and st is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                for (new, old) in zip(good_new, good_old):
                    a, b = new.ravel()
                    c, d = old.ravel()

                    # Draw motion trail
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color=(0, 255, 0), thickness=2)

                    # Draw corner point
                    frame = cv2.circle(frame, (int(a), int(b)), 3, color=(0, 0, 255), thickness=-1)

                    # --- NEW: follow edges in local patch ---
                    x, y = int(a), int(b)
                    patch_size = 15
                    half = patch_size // 2
                    x1, y1 = max(x - half, 0), max(y - half, 0)
                    x2, y2 = min(x + half, width), min(y + half, height)

                    edge_patch = edges[y1:y2, x1:x2]
                    edge_coords = cv2.findNonZero(edge_patch)
                    if edge_coords is not None:
                        for pt in edge_coords:
                            ex, ey = pt[0][0] + x1, pt[0][1] + y1
                            frame = cv2.circle(frame, (ex, ey), 1, (255, 255, 0), -1)  # Cyan edge dot

                output = cv2.add(frame, mask)
                p0 = good_new.reshape(-1, 1, 2)
            else:
                output = frame
                p0 = None
        else:
            output = frame
            brightness_mask = np.zeros_like(frame_gray)
            brightness_mask[frame_gray > 30] = 255
            edges = cv2.Canny(frame_gray, 50, 150)
            combined_mask = cv2.bitwise_and(edges, brightness_mask)
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=combined_mask, **feature_params)

        out.write(output)
        old_gray = frame_gray.copy()

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Tracking video saved to {output_path}")

# Example usage
video_path = "output_videos/video_sample_1.mp4"
output_path = "tracked_output.avi"
track_objects_with_stif(video_path, output_path)
