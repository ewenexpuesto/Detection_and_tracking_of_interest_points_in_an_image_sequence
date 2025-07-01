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

input_path = "output_videos/4_cut_local_color_propagation_10_10_70_from_difference_of_intensity_of_superpixels_matrix_0.002_0.5.mp4"
output_folder = "output_videos"
video_name = "4_track_red_pixel_with_optical_flow_from_cut_local_color_propagation_10_10_70_from_difference_of_intensity_of_superpixels_matrix_0.002_0.5.mp4"
output_path = output_folder+"/"+video_name
output_log = "log/4_track_red_pixel_with_optical_flow_from_cut_local_color_propagation_10_10_70_from_difference_of_intensity_of_superpixels_matrix_0.002_0.5.csv"

kernel_size = 5
sigma1 = 1
sigma2 = 2
p = 0.002
t = 0.5
v = 10
w = 10
threshold = 70

frames, fps = video_processing.mp4_to_list_of_arrays(input_path)
frames_processed = block_matching_and_optical_flow.track_red_pixels_with_optical_flow(frames, fps, output_log)
video_processing.list_of_arrays_to_mp4(frames_processed, output_folder, video_name, fps)