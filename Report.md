# Detection and tracking of interest points in an image sequence

## Prerequisite

No human intervention at all

## Object detection, finding points of interest, feature extraction

[Feature (wikipedia)](https://en.wikipedia.org/wiki/Feature_(computer_vision))

SLAM (*simultaneous localization and mapping*) : mapping the environnement relatively to an agent and finding where it is in this environnement. It is used in robot navigation and virtual reality using motion sensors. SLAM encompasses different algorithms. **Here we only have images and videos** and we don't study the robot's position, only the eternal world. *Library : Mobile Robot Programming Toolkit in C++*.

SLAM with DATMO is a model which tracks moving objects in a similar way to the agent itself.

A robot has idiothetic (himself) sources of position and motion knowledge as well as allothetic (others) sources. **We are only studying allothetic images sources**. Our program takes as input a video. **The goal is to follow the objects present on the video**.

### Edge detection

#### Derivative methods

##### Gaussian blurr

To detect edges (borders) of images, we can blur the image and substract it to the original image. Applying a Gaussian blur is the same as convolving (convolution, "multiplication") with a Gaussian function. Since the Fourier Transform of a Gaussian is another Gaussian, applying a Gaussian blur has the effect of reducing the image's high-frequency components; a Gaussian blur is thus a low-pass-filter. In two dimensions, the Gaussian function is the product of two such Gaussian functions, one in each dimension (symetry) : $G(x,y)=\frac{1}{2\pi \sigma^2}e^{-\frac{x^2+y^2}{2\sigma^2}}$ (insert a picture of the function) where $x$ is the distance from the origin in the horizontal axis, $y$ is the distance from the origin in the vertical axis and $\sigma$ is the standard deviation of the Gaussian distribution. When applied in two dimensions, this formula produces a surface whose contours are concentric circles with a Gaussian distribution from the center point. Values from this distribution are used to build a convolution matrix which is applied to the original image. This convolution process is illustrated visually in the figure on the right. Each pixel's new
value is set to a weighted average of that pixel's neighborhood as we can exclude all points further than $3\sigma$. The original pixel's value receives the heaviest weight (having the highest Gaussian value) and neighboring pixels receive smaller weights as their distance to the original pixel increases.

How much $\sigma$ will impact the blurred image ? A higher $\sigma$ will fade the image.

Why use Gaussian and no other function ? Because it is circles and counts each contribution of each pixel according to its distance

We use this method before applying each of the following methods.

##### Laplacian of Gaussian

We then get an image $L(x,y)=g(x,y)*f(x,y)$ that is the convolution of the Gaussian with the original image. To compute the edge, you compute the Laplacian of the new image. This allows to find at which rate pixels are changing.

Why blurring is necessary ? Raw images contain noise, and taking second derivatives (like the Laplacian does) can amplify that noise.

Why not use Jacobian (first derivative of speed) ? The Laplacian detects where the gradient peaks or changes sign (i.e., zero-crossings), which often correspond to actual edges. First derivatives tell you "there's change here." Second derivatives tell you "there's a peak, valley, or blob here." For feature detection (like in SIFT, which is about finding the "global leading" gradient, Jacobian here), you want interest points, not just edges.

Why ? An edge is a boundary where the image intensity changes sharply, for instance the border of an object, but edges are full of many indistinctive points, and we want later to be able to follow remarquable points between images, for instance corners, which are the intersection of two edges

- Edges are not unique enough because on an edge, there's no information in one direction. Imagine trying to identify a location in a desert where the only feature is a straight fence : it's easy to say "I'm next to the fence", but hard to say where along the fence. No landmark to help localize.
- Edges are too sensitive to small noise. Edges are detected using first derivatives (gradients, Jacobian), which are sensitive to tiny intensity changes. Even a small variation in brightness, noise, or image compression can change where or whether an edge is detected. While the second derivative (Laplacian) is more stable and detects corners by definition.
- Matching edges or corners could cause ambiguity : a point on one image's edge could match many different points along the corresponding edge in another image.
- Laplacian-based methods are more sensitive to these structures. The Laplacian is isotropic (same in all directions), so it's more suitable for rotation-invariant detection. Is it because there are circles so the second derivative is juste a constant ?

##### Difference of Gaussians

The method here is to subtract a Gaussian-blurred image from the original image. Blurring an image using a Gaussian kernel (that is how the function is called) is a low-passing filter, which means low intensity pixels pass more than higher ones. Indeed, the Gaussian function is an ascending function and then descending function

##### Determinant of Hessian

Similar idea to the Laplacian idea. It has the advantage do be rotation-invariant.

Then computing the determinant of the Hessian matrix for each point gives if it is an optimum and which optimum it is :

- If $\det(H)>0$ then the point is a local extremum
- If $\det(H)<0$ the point is a saddle point
- If $\det(H)=0$, the surface is flat, or ambiguous

| Aspect                       | **Laplacian (LoG)**                                         | **Hessian (Determinant)**                                                  |
| ---------------------------- | ----------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Definition**         | Sum of second derivatives:$f_{xx} + f_{yy}$                     | Matrix of second derivatives; use determinant:$f_{xx} f_{yy} - f_{xy}^2$       |
| **Blob measure**       | Zero-crossings or extrema of the LoG response                     | Maxima of the determinant of the Hessian                                         |
| **Isotropy**           | Isotropic (rotation invariant by default)                         | Determinant is also rotation invariant, but captures anisotropy in structure     |
| **Sensitivity**        | Responds to intensity changes (e.g. edges and blobs)              | Captures structure with consistent curvature in multiple directions (blobs only) |
| **Computational Cost** | LoG is computationally simpler, especially with DoG approximation | Requires computing all second derivatives and determinant                        |
| **Popular Use**        | Used in SIFT (via DoG), Marr-Hildreth edge detector               | Used in SURF, Hessian-based detectors (e.g. Hessian-Laplace, Hessian-Affine)     |

##### The hybrid Laplacian and determinant of the Hessian operator (Hessian-Laplace)

#### Kadir–Brady saliency detector

The image must first be turned into a grayscale image. The algorithm then computes the Shannon entropy around each pixel, given the size of the box around inside which it must compute.

First build a histogram of pixel intensities in this box. Then normalize it to get probabilities. Then compute the entropy. The entropy measures how diverse/uncertain the region is, that is to say how much information there is : $-\sum_i p_i \log_2 p_i$ with $p_i$ being the proportion of the $i$ pixel intensity. $\log_2 p_i$ measures the information content (or surprise) of seeing $i$ and the entropy is thus the expected information (average information).

#### Maximally stable extremal regions : accentuate contrast

Every pixel is put in a class according to if it fits in a certain threshold. This way the image is cut in 2 groups. It is very lightweight. If $n$ is the number of pixels, the process might take a different amount of time. Using binary sort, it would take $O(n)$.

For some use cases, it is possible to divide the image in a grid and then compute local thresholding

Note that for colors and not for grayscale, we use agglomerative clustering using colour gradients. Often referred to as a "bottom-up" approach, begins with each data point as an individual cluster. At each step, the algorithm merges the two most similar clusters based on a chosen distance metric (e.g., Euclidean distance) and linkage criterion (e.g., single-linkage,
complete-linkage). This process continues until all data points are combined into a single cluster or a stopping criterion is met. Agglomerative methods are more commonly used due to their simplicity and computational efficiency for small to medium-sized datasets. In constrast, there is a divisive clustering, known as a "top-down" approach, starts with all
data points in a single cluster and recursively splits the cluster into smaller ones. It is a greedy algorithm : it makes a series of locally optimal choices without reconsidering previous steps.

It is possible to try other types of clustering methods like k-means, BIRCH or CURE.

#### Difference between bilinear and nearest neighbour resampling

Substracting the bilinear-rescaled image from the nearest-neighbour-resampled image can outline the borders. See for [proof](https://github.com/ewenexpuesto/Image-manipulation-and-filters-with-C/blob/main/final_project/images/Lenna_color_difference.ppm)

But it is less efficient than Gaussian difference

### Corner detection

The goal is to connect corners to edges.

Detecting corners : corners are the intersection of edges

### Blob detection

Locating and tracking the target object successfully is dependent on the algorithm. For example, using blob tracking is useful for identifying human movement because a person's profile changes dynamically.

Laplacian of Gaussian, Difference of Gaussians and Determinant of Hessian allow to detect blobs

### Kanade-Lucas-Tomasi (KLT) Tracker

[Lucas-Kanade feature tracker](https://rpg.ifi.uzh.ch/docs/teaching/2020/11_tracking.pdf)

[Lucas-Kanade feature tracker wikipedia](https://en.wikipedia.org/wiki/Kanade%E2%80%93Lucas%E2%80%93Tomasi_feature_tracker)

#### Tomasi-Kanade

Method for choosing the best feature (image patch) for tracking

#### Lucas-Kanade

Method for aligning (tracking) an image patch

[Lucas-Kanade](https://www.cs.cmu.edu/~16385/s17/Slides/15.1_Tracking__KLT.pdf)

### Convolutional neural networks, YOLO

The goal is to optimize a kernel in a learning process. In our case, we must take a pre-trained one.

#### Region-based convolutional neural networks

Why not combine both worlds ? Also use image segmentation ? Compare all methods and do benchmarks

## Motion detection, or how to establish correspondence between images

The objective of video tracking is to associate target objects in consecutive video frames. The association can be especially difficult when the objects are moving fast relative to the frame rate. Another situation that increases the complexity of the problem is when the tracked object changes orientation over time. For these situations video tracking systems usually employ a motion model which describes how the image of the target might change for different possible motions of the object. To perform video tracking an algorithm analyzes sequential video frames and outputs the movement of targets between the frames. There are a variety of algorithms, each having strengths and weaknesses. Considering the intended use is important when choosing which algorithm to use. There are two major components of a visual tracking system: target representation and localization, as well as filtering and data association.

Structure from motion is a photogrammetric range imaging (photo and depth) technique for estimating three-dimensional structures from two-dimensional image sequences that may be coupled with local motion signals (vectors). Biological creatures use motion parallax from 2D images. This is more what we are able to do.

There are two basic ways to find the correspondences between two images :

- Correlation-based – checking if one location in one image looks/seems like another in another image.
- Feature-based – finding features in the image and seeing if the layout of a subset of features is similar in the two images. To avoid the aperture problem a good feature should have local variation in two directions.
- Multi-scale-approach - Scaling the image down to reduce the search space, then correct the coarse approximations on smaller windows. Solving the correspondence problem over a small search spaces is easily trained on a convolutional neural network.

### How to solve the aperture problem ?

[Aperture problem](https://www.sciencedirect.com/topics/computer-science/aperture-problem)

### Correlation-based approach : direct methods

The goal is to deal with optical flow, which is the apparent motion of object when moving comparatively to them, even if the objects are not moving relatively to the Earth for instance.

#### Block-matching algorithm

The goal is to minimize an error function that corresponds the move of blocks between consecutive frame which matches the movement of the classes (dectected objects)

### Feature-based approach : indirect methods

#### Scale-invariant feature transform

To achieve motion detection, we need to find correspondence between images, which refers to the problem of ascertaining which parts of one image correspond to which parts of another image, where differences are due to **movement of the camera**, the **elapse of time**, and/or **movement of objects in the photos**. We use SIFT (scale-invariant feature transform). The first step in SIFT is finding a dominant gradient direction of the points of interest we found. SIFT builds a histogram of gradient directions weighted by their magnitude. The peak(s) in this histogram represent the dominant orientation(s) of that keypoint’s local patch. To make it rotation-invariant (if the camera rotates), you can describe keypoints relative to their dominant orientation: once the dominant gradient direction is found, SIFT rotates the coordinate system of the descriptor to align with this orientation. This means the feature descriptor is always “normalized” to this orientation before comparing with others.

Why is it scale-invariant ? Because if we zoom it works the same ?

#### Speeded up robust features

#### Condensation (conditional density propagation) algorithm

It is a probabilistic algorithm. Each pixel is not studied. Instead pixels are chosen randomly. The algorithm’s creation was inspired by the inability of Kalman filtering to perform object tracking well in the presence of significant background clutter.  The presence of clutter tends to produce probability distributions for the object state which are multi-modal and therefore poorly modeled by the Kalman filter.  The condensation algorithm in its most general form requires no assumptions about the probability distributions of the object or measurements.

### Multi-scale-approach

### Kalman filter to correct noise in addition to other methods

It answers "Given some noisy measurements, what is the most likely state of a system?"

The algorithm works via a two-phase process: a prediction phase and an update phase. In the prediction phase, the Kalman filter produces estimates of the current state variables,
 including their uncertainties. Once the outcome of the next measurement (necessarily corrupted with some error, including random noise) is observed, these estimates are updated using a weighted average, with more weight given to estimates with greater certainty. The algorithm is recursive. It can operate in real time, using only the present input measurements and the state calculated previously and its uncertainty matrix; no additional past information is required.

Prediction step:

- You use the old frame parameters (position and speed of a point of interest in 2D) to estimate where the car should be next.
- This gives you a predicted state and a confidence level (uncertainty).

Update step:

- You take a new measurement
- You combine the predicted position and the noisy measurement, weighting each by how confident you are in them.
- The result is a better estimate than either one alone.

This comes on top of motion detection solution and corrects it

### Co-segmentation

Co-segmentation differs from segmentation in which it lasts throughout a video and not just a frame.

It is often challenging to extract segmentation masks of a target/object from a noisy collection of images or video frames, which involves object discovery coupled with segmentation. A noisy collection implies that the object/target is present sporadically in a set of images or the object/target disappears intermittently throughout the video of interest. Early methods typically involve mid-level representations such as object proposals.

First is to segment the image through superpixels. The number of clusters must not be specified, but ideally optimized. In the beginning there are a certain number of clusters which form a grid ([Grid of clusters](https://www.youtube.com/watch?v=zx1CthO5FEk)) and then it grows

Then to ensure consistency between frames, labels must stay consistent. For that, each class from each image must be added to a graph that according to its form/colours/overall appearance so that we have a graph with cliques. [Graph and cliques](https://www.youtube.com/watch?v=TdRYcZ2xUSM)

## Limits

We are at risk to segment all small parts of an image, for instance the windows of a car, as well as the handle, the wheels etc as they are all different colours

Depending if the image is full of colour or not, we may want to give a growing importance to the colour difference between pixels to define edges, so that to detect a green leaf you must regroup all green-ish pixel (even the dark ones) in a single class

Main possible issues : rotating, scaling, lighting
