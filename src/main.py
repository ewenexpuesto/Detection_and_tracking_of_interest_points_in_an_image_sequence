import video_processing
import gaussian_blur
import difference_of_gaussians
import SIFT

input_path = "input_videos/video_sample_1-uhd_3840_2160_25fps.mp4"
kernel_size = 0
sigma1 = 0.1
sigma2 = 1.0
output_path = "output_videos"
output_path_frames = "output_frames"
video_name = "video_sample_tracked_14_SIFT_and_optical_flow.mp4"

#SIFT.SIFT_difference_of_gaussians_mp4(input_video_path=input_path, output_video_path=output_path, video_name=video_name, kernel_size=kernel_size, sigma1=sigma1, sigma2=sigma2)

frames, fps = video_processing.mp4_to_list_of_arrays(input_path)
SIFT_frames = SIFT.SIFT_list_of_arrays(frames)
video_processing.list_of_arrays_to_mp4(SIFT_frames, output_path, video_name)