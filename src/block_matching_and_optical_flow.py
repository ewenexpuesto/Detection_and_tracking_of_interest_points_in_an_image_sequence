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

    for i in range(total_frames - 1):
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

                # Draw motion vector
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

import os
import csv

def track_red_pixels_with_optical_flow(
    frames: list[NDArray[np.uint8]],
    fps: float = 30.0,
    log_path: str = "red_optical_flow_log.csv",
    red_bgr: tuple[int, int, int] = (0, 0, 255),
    color_tolerance: int = 35,
    max_corners: int = 30,
    win_size: tuple[int, int] = (20, 20),
    max_level: int = 2 # The more is better
    ) -> list[NDArray[np.uint8]]:
    output_frames = []
    total_frames = len(frames)
    dt = 1 / fps

    lk_params = dict(winSize=win_size, maxLevel=max_level,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    prev_frame = frames[0]
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(prev_frame, np.array(red_bgr, dtype=np.uint8))
    red_mask = (np.linalg.norm(diff, axis=2) <= color_tolerance).astype(np.uint8) * 255

    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=red_mask, maxCorners=max_corners, qualityLevel=0.001, minDistance=5)

    prev_velocities = np.zeros_like(p0) if p0 is not None else None
    trajectories = [[] for _ in range(len(p0))] if p0 is not None else []
    log_rows = [("Frame", "Time (s)", "Point ID", "X", "Y", "Vx", "Vy", "Ax", "Ay")]

    for i in range(1, total_frames):
        time_sec = i * dt
        frame = frames[i]
        # frame_display = frame.copy()
        frame_display = frame # saves RAM
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if p0 is None:
            output_frames.append(frame_display)
            continue

        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
        if p1 is None:
            output_frames.append(frame_display)
            continue

        good_new = p1[st == 1]
        good_old = p0[st == 1]
        old_velocities = prev_velocities[st == 1] if prev_velocities is not None else np.zeros_like(good_new)

        avg_pos = np.array([0.0, 0.0])
        avg_vel = np.array([0.0, 0.0])
        avg_acc = np.array([0.0, 0.0])
        count = 0

        new_trajectories = []

        for j, (new, old, old_vel) in enumerate(zip(good_new, good_old, old_velocities)):
            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()
            vx, vy = (new - old) / dt
            ax, ay = ((vx, vy) - old_vel[0]) / dt

            # Log and stats
            avg_pos += new
            avg_vel += [vx, vy]
            avg_acc += [ax, ay]
            count += 1

            log_rows.append((i, f"{time_sec:.2f}", j, int(x_new), int(y_new),
                             f"{vx:.2f}", f"{vy:.2f}", f"{ax:.2f}", f"{ay:.2f}"))

            # Draw current point
            cv2.circle(frame_display, (int(x_new), int(y_new)), 3, (0, 255, 0), -1)

            # Draw trajectory
            traj = trajectories[j] if j < len(trajectories) else []
            traj.append((int(x_new), int(y_new)))
            for k in range(1, len(traj)):
                cv2.line(frame_display, traj[k-1], traj[k], (0, 180, 255), 1)
            new_trajectories.append(traj)

        # Display average data if any points are tracked
        if count > 0:
            cx, cy = avg_pos / count
            cv2.putText(frame_display, f"Pos: ({int(cx)},{int(cy)})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(frame_display, f"Vel: ({avg_vel[0]/count:.1f},{avg_vel[1]/count:.1f})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(frame_display, f"Acc: ({avg_acc[0]/count:.1f},{avg_acc[1]/count:.1f})", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        output_frames.append(frame_display)

        # Update
        prev_gray = gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        prev_velocities = (good_new - good_old).reshape(-1, 1, 2)
        trajectories = new_trajectories

        progress = (i + 1) / total_frames * 100
        print(f"Tracking Red w/ Optical Flow: Frame {i + 1}/{total_frames} ({progress:.1f}%)", end='\r')

    # Save CSV
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(log_rows)

    print()
    return output_frames


import csv

def optical_flow_rectangle(
    frames: list[NDArray[np.uint8]],
    fps: float = 30.0,
    log_path: str = "optical_flow_green_log.csv",
    target_color: tuple[int, int, int] = (0, 255, 0),
    color_tolerance: int = 20
    ) -> list[NDArray[np.uint8]]:
    """
    Track a colored rectangular object in a sequence of frames using LK optical flow and log its position, velocity, and acceleration.

    Parameters
    ----------
    frames : list of np.ndarray
        Sequence of video frames (BGR images, dtype=uint8).
    fps : float, optional
        Frames per second of the video. Used to compute time, velocity, 
        and acceleration. Default is 30.0.
    log_path : str, optional
        Path to the CSV log file storing position, velocity, and acceleration data. 
        Default is "optical_flow_green_log.csv".
    target_color : tuple of int, optional
        BGR color of the object to track. Default is green ``(0, 255, 0)``.
    color_tolerance : int, optional
        Euclidean distance tolerance in color space for detecting the target 
        object in the first frame. Default is 20.

    Returns
    -------
    list of np.ndarray
        List of processed frames with the trajectory, current position, 
        velocity, and acceleration drawn.
    """
    output_frames = []
    log_rows = [("Time (s)", "X", "Y", "Vx", "Vy", "Ax", "Ay")]
    trajectory_points = []

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(frames[0], np.array(target_color, dtype=np.uint8))
    dist = np.linalg.norm(diff, axis=2)
    green_mask = (dist < color_tolerance).astype(np.uint8) * 255

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No green object detected in first frame.")

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(largest_contour)

    center = np.array([[x + w_box // 2, y + h_box // 2]], dtype=np.float32).reshape(-1, 1, 2)
    prev_center = center[0][0]
    prev_velocity = np.array([0.0, 0.0])

    for i in range(1, len(frames)):
        time_sec = i / fps
        frame = frames[i]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, center, None)

        if status[0][0] == 1:
            new_center = next_pts[0][0]
            dx, dy = new_center - prev_center
            dt = 1 / fps
            velocity = np.array([dx, dy]) / dt
            acceleration = (velocity - prev_velocity) / dt

            cx, cy = int(new_center[0]), int(new_center[1])
            trajectory_points.append((cx, cy))

            # Draw trajectory
            for j in range(1, len(trajectory_points)):
                cv2.line(frame, trajectory_points[j - 1], trajectory_points[j], (0, 255, 255), 2)

            # Draw current center
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            # Overlay position, velocity, acceleration
            cv2.putText(frame, f"Pos: ({cx},{cy})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(frame, f"Vel: ({velocity[0]:.1f},{velocity[1]:.1f})", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(frame, f"Acc: ({acceleration[0]:.1f},{acceleration[1]:.1f})", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            log_rows.append((
                f"{time_sec:.2f}", cx, cy,
                f"{velocity[0]:.2f}", f"{velocity[1]:.2f}",
                f"{acceleration[0]:.2f}", f"{acceleration[1]:.2f}"
            ))

            center = next_pts
            prev_center = new_center
            prev_velocity = velocity
            prev_gray = gray.copy()

        else:
            cv2.putText(frame, "Tracking failure", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        output_frames.append(frame)
        print(f"Optical Flow Tracking: Frame {i+1}/{len(frames)} ({(i+1)/len(frames)*100:.1f}%)", end='\r')

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(log_rows)

    print()
    return output_frames
