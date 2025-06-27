# Task log

## Video processing

Functions to cut an mp4 into a list of NumPy arrays, build the frames back into mp4, store

## Difference of Gaussians

Function to Gaussian blur one array, then multiple

And then compute the Difference of Gaussians

## SIFT

It doesn't work well on static videos where only a small object is moving. Because it uses edges and there are a lot of edges in a video no matter how the objects are moving in it. It is way too generic and not adapted to our problem.

## Difference of intensity of superpixels

Compares each frame : build superpixels, like a grid and computes the average intensity. If the superpixel at the same position has changed of intensity more than a certain threshold in absolute value then this is a feature. It is highlighted in the video.

## ST corner detection with LK optical flow

Works well but doesn't detect when a new feature appears on the screen because it follows the old feature it found at the beginning of the video as soon as the first frame.
