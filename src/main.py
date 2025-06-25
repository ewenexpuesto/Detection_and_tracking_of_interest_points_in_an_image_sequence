import video_processing
import gaussian_blur
import difference_of_gaussians

input_path = "sources/video_sample_1-uhd_3840_2160_25fps.mp4"
kernel_size = 0
sigma1 = 0.1
sigma2 = 1.0
output_path = "output_videos"
output_path_frames = "output_frames"

video_array = video_processing.video_to_frame_arrays(input_path)
list_of_frames = []
for i, frame in enumerate(video_array):
    dog_frame = difference_of_gaussians.compute_difference_of_gaussians(frame, kernel_size, sigma1, sigma2)
    dog_frame = video_processing.stretch_colors(dog_frame) # This takes more time
    list_of_frames.append(dog_frame)
video_processing.save_frames_array(list_of_frames, output_path_frames, prefix="frame")
video_processing.frames_array_to_video_write(list_of_frames, output_path, video_name="video_sample_1.mp4", fps=25)