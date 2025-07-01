# Detection and tracking of interest points in an image sequence

## What is inside

- `src` folder : source code
- `report.md` : state of the art on the field with litterature sources
- `task_log.md` : what I did step by step
- `input_videos` folder : the videos studied are stored here
- `output_videos` folder : where the video result is stored after executing the program; some examples are already here to showcase the results
- `log` folder : where position, velocity and acceleration .csv of points of interest detected on videos are stored (the units are in pixels per second or second squared)

## Prerequisites

- Python versions 3.13+
- Download the libraries : NumPy 2.3.1+, OpenCV 4.11.0+ with extra modules `pip install opencv-contrib-python`
- Git 2+

## Step by step guide on how to use

- After cloning the repository, go in the root of the folder through the terminal (in progress)
- Execute the code through the terminal via the command `python src/main.py` (in progress)
- Choose manual of automatic mode; if manual, then follow the instructions appearing on the screen (in progess)
- Choose your algorithm; read `report.md` for more information about each method (in progess)
