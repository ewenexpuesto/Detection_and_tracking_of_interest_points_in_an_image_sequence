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

## Jeudi soir et vendredi 27/6

Added a matrix-multiplication-based difference of intensity of superpixels

Tested KB saliency detector : heatmap

Tested difference of intensity of superpixels after difference of gaussians and KB saliency detector : less efficient as the original image because the original has all the information

MSER detects edge quite well, but it misses a lot of edeges, so sometimes it misses the moving car, especially when it is distant. Sticking with gaussian difference is also more computationally efficient (faster).

Block matching (naive version) is way too slow

Optical flow is way faster but maybe less efficient (probably not) than the block matching naive. It works well but a lot of noise

Applied two times in a row the difference of intensity of superpixels algorithm : first time as before, then we remove all colors from the video except the one that the algorithm highlighted. This way we get an almost full black video. Then by applying the difference of intensity of superpixels algorithm but with bigger superpixels, we get a bit less noise, but it doesn't seem it is worth the hassle.

## Lundi 30/06

Difference of intensity of superpixels + cosegmentation : grouping superpixels together to layer the whole object and forget noise

Using kmeans after the difference of intensity of superpixels ? Très long et échec avec k=2.

Using a local color propagation algorithm to propagate the highlighted pixels so that they form more of an object rather than independant pixels

Optical flow works better when you tell it to follow a red dot than when you tell it nothing
