import video_processing
import gaussian_blur
import difference_of_gaussians
import SIFT
import difference_of_intensity_of_superpixels
import ST_corner_detection_and_LK_optical_flow
import activity_detection
import KB_saliency_detector
import mser
import block_matching_and_optical_flow
import cosegmentation
import local_color_propagation
import manual_retangle_tracker
import user_selection
import YOLO

input_path = "input_videos/20250314_163308.mp4"
output_folder = "output_videos"
video_name = "YOLO_continuous.mp4"

kernel_size = 5
sigma1 = 1
sigma2 = 2
p = 0.002
t = 0.5
v = 10
w = 10
threshold = 70

frames, fps = video_processing.mp4_to_list_of_arrays(input_path)
processed_frames = YOLO.detect_objects(frames, ["car", "motorcycle", "bus", "truck"])
video_processing.list_of_arrays_to_mp4(processed_frames, output_folder, video_name, fps)