import video_processing
import gaussian_blur
import difference_of_gaussians
import SIFT
import difference_of_intensity_of_superpixels
import ST_corner_detection_and_LK_optical_flow
import activity_detection

input_path = "input_videos/video_sample_4-1080p.mp4"
output_folder = "output_videos"
video_name = "4_difference_of_intensity_of_superpixels_matrix_0.002_0.35.mp4"
output_path = output_folder+"/"+video_name

kernel_size = 5
sigma1 = 1
sigma2 = 2

# frames, fps = video_processing.mp4_to_list_of_arrays(input_path)
# frames_processed = ST_corner_detection_and_LK_optical_flow.process_optical_flow(frames)
# video_processing.list_of_arrays_to_mp4(frames_processed, output_folder, video_name, fps)

frames, fps = video_processing.mp4_to_list_of_arrays(input_path)
frames_processed = difference_of_intensity_of_superpixels.difference_of_intensity_superpixels_matrix(frames, 0.002, 0.35)
video_processing.list_of_arrays_to_mp4(frames_processed, output_folder, video_name, fps)