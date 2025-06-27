import numpy as np
import cv2 as cv

def process_optical_flow(
    frames_list,
    # Parameters for Shi-Tomasi corner detection
    max_corners=1000,
    quality_level=0.3,
    min_distance=7,
    block_size=7,
    # Parameters for Lucas-Kanade optical flow
    win_size=(15, 15),
    max_level=2,
    lk_epsilon=0.03,
    max_iter=10,
    # Feature refresh parameters
    refresh_interval=5,  # Try to find new features every N frames
    refresh_ratio=0.3,   # Replace up to 30% of features
    new_feature_quality_threshold=0.8,  # Only add very good new features
    min_features_threshold=20,  # Minimum number of features to maintain
):
    """
    Process Lucas-Kanade optical flow on a list of numpy arrays (frames).
    Periodically refreshes features by adding new high-quality corner points.

    Parameters
    ----------
    frames_list : list of np.ndarray
        List of numpy arrays representing video frames (BGR format)
    max_corners : int, optional
        Maximum number of corners to detect
    quality_level : float, optional
        Minimum quality of corners (between 0 and 1)
    min_distance : int, optional
        Minimum distance between corners in pixels
    block_size : int, optional
        Neighborhood size for corner detection
    win_size : tuple, optional
        Search window size for optical flow
    max_level : int, optional
        Pyramid levels for optical flow
    lk_epsilon : float, optional
        Stopping criteria epsilon
    max_iter : int, optional
        Maximum iterations for optical flow
    refresh_interval : int, optional
        Interval (in frames) to attempt feature refresh
    refresh_ratio : float, optional
        Maximum ratio of features to replace (0.0 to 1.0)
    new_feature_quality_threshold : float, optional
        Quality threshold for new features (higher = more selective)
    min_features_threshold : int, optional
        Minimum number of features to maintain before forcing refresh

    Returns
    -------
    output_frames : list of np.ndarray
        List of numpy arrays with optical flow tracks drawn
    """

    if not frames_list or len(frames_list) < 2:
        raise ValueError("Need at least 2 frames to calculate optical flow")

    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size
    )
    
    # Parameters for high-quality new features
    new_feature_params = dict(
        maxCorners=max_corners,
        qualityLevel=max(quality_level, new_feature_quality_threshold),
        minDistance=min_distance,
        blockSize=block_size
    )
    
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(
        winSize=win_size,
        maxLevel=max_level,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, max_iter, lk_epsilon)
    )
    
    # Create some random colors (expand for more features)
    color = np.random.randint(0, 255, (max_corners, 3))
    
    # Take first frame and find corners in it thanks to Shi-Tomasi algorithm
    old_frame = frames_list[0]
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    
    # List to store output frames
    output_frames = []
    
    # Track feature ages and colors
    if p0 is not None:
        feature_ages = np.zeros(len(p0))  # Track how long each feature has been tracked
        feature_colors = color[:len(p0)]  # Assign colors to initial features
    else:
        feature_ages = np.array([])
        feature_colors = np.array([]).reshape(0, 3)
    
    # Process each frame
    for i in range(1, len(frames_list)):
        frame = frames_list[i].copy()  # Copy to avoid modifying original
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Calculate optical flow between old frame and the current frame
        if p0 is not None and len(p0) > 0:
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            
            # Select good points and update ages
            if p1 is not None:
                good_mask = st.flatten() == 1
                good_new = p1[good_mask]
                good_old = p0[good_mask]
                feature_ages = feature_ages[good_mask] + 1  # Increment age of tracked features
                feature_colors = feature_colors[good_mask]
                
                # Try to refresh features periodically or when we have too few
                should_refresh = (i % refresh_interval == 0) or (len(good_new) < min_features_threshold)
                
                if should_refresh and len(good_new) > 0:
                    # Create a mask to avoid detecting features too close to existing ones
                    existing_mask = np.ones(frame_gray.shape[:2], dtype=np.uint8) * 255
                    for point in good_new:
                        x, y = point.ravel().astype(int)
                        cv.circle(existing_mask, (x, y), min_distance * 2, 0, -1)
                    
                    # Find new high-quality features
                    new_features = cv.goodFeaturesToTrack(
                        frame_gray, 
                        mask=existing_mask, 
                        **new_feature_params
                    )
                    
                    if new_features is not None and len(new_features) > 0:
                        # Calculate how many features to replace
                        max_replacements = max(1, int(len(good_new) * refresh_ratio))
                        num_new_features = min(len(new_features), max_replacements)
                        
                        if num_new_features > 0:
                            # Select oldest features to replace
                            oldest_indices = np.argsort(feature_ages)[-num_new_features:]
                            
                            # Replace oldest features with new ones
                            new_points = new_features[:num_new_features].reshape(-1, 2)
                            good_new[oldest_indices] = new_points
                            good_old[oldest_indices] = new_points
                            feature_ages[oldest_indices] = 0  # Reset age for new features
                            
                            # Assign new colors to replaced features
                            available_colors = color[len(feature_colors):len(feature_colors) + num_new_features]
                            if len(available_colors) > 0:
                                feature_colors[oldest_indices] = available_colors
                
                # Draw the tracks
                for j, ((new, old), feat_color) in enumerate(zip(zip(good_new, good_old), feature_colors)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    
                    # Use feature-specific color
                    color_to_use = feat_color.tolist() if len(feat_color) == 3 else color[j % len(color)].tolist()
                    
                    mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color_to_use, 2)
                    frame = cv.circle(frame, (int(a), int(b)), 5, color_to_use, -1)
                
                # Update previous points
                p0 = good_new.reshape(-1, 1, 2)
            else:
                # If no good points found, try to find new features
                p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
                if p0 is not None:
                    feature_ages = np.zeros(len(p0))
                    feature_colors = color[:len(p0)]
        else:
            # If no points to track, find new features
            p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            if p0 is not None:
                feature_ages = np.zeros(len(p0))
                feature_colors = color[:len(p0)]
        
        # Combine frame with tracking mask
        result_frame = cv.add(frame, mask)
        output_frames.append(result_frame)
        
        # Update previous frame
        old_gray = frame_gray.copy()
    
    return output_frames

def get_feature_statistics(frames_list, **kwargs):
    """
    Helper function to analyze feature tracking statistics.
    
    Parameters
    ----------
    frames_list : list of np.ndarray
        List of frames to process
    **kwargs : dict
        Parameters to pass to process_optical_flow
    
    Returns
    -------
    dict : Statistics about feature tracking
    """
    # This is a simplified version that just processes and returns basic stats
    output_frames = process_optical_flow(frames_list, **kwargs)
    
    return {
        'total_frames_processed': len(output_frames),
        'input_frames': len(frames_list),
        'processing_success': len(output_frames) > 0
    }