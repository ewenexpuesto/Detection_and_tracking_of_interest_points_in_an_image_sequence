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

input_path = "output_videos/4_keep_only_color_0_255_0_from_difference_of_intensity_of_superpixels_matrix_0.002_0.35.mp4"
output_folder = "output_videos"
video_name = "4_difference_of_intensity_of_superpixels_matrix_0.004_0.9_from_4_keep_only_color_0_255_0_from_difference_of_intensity_of_superpixels_matrix_0.002_0.35.mp4"
output_path = output_folder+"/"+video_name

kernel_size = 5
sigma1 = 1
sigma2 = 2
p = 0.004
t = 0.9

frames, fps = video_processing.mp4_to_list_of_arrays(input_path)
frames_processed = difference_of_intensity_of_superpixels.difference_of_intensity_superpixels_matrix(frames, p, t)
video_processing.list_of_arrays_to_mp4(frames_processed, output_folder, video_name, fps)
