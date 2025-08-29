# Detection and tracking of interest points in an image sequence

## Objectives

Develop two approaches for object detection and tracking:

* **Manual tracking** : A human operator selects a specific point of interest (ideally a well-defined corner), and the system continuously tracks it throughout the video.
* **Automatic tracking** : The system autonomously detects keypoints or features of interest and tracks them over time.

## Feature Detection and Object Tracking : edge, corner and blob detection

We focus on tracking visual features across video frames to analyze object motion. This involves detecting and extracting meaningful points of interest in the image, such as corners or textured regions.

In image processing, **edges** are regions where the intensity changes sharply, typically marking object boundaries. **Corners** are points where two edges meet and intensity varies in multiple directions, making them ideal for tracking. **Blobs** are areas that differ in texture or brightness from their surroundings, often representing compact regions of interest. Together, edges, corners, and blobs are key features used to detect and understand structures in images.

### SLAM

Although this task relates conceptually to **SLAM** ( *Simultaneous Localization and Mapping* ), our scope is narrower. SLAM is typically used in robotics and augmented reality to map an environment and localize an agent within it using data from sensors and cameras. A popular SLAM implementation is the **Mobile Robot Programming Toolkit (MRPT)** in C++.

However, in our case:

* We do **not** analyze the agent's position or use motion sensors.
* We **only** work with video input (i.e., external visual data).
* Our objective is to **track external objects** in the video frames, not to localize the observer.

A related concept is **SLAM with DATMO** (Detection and Tracking of Moving Objects), which combines environment mapping and object tracking. While inspired by this, our system focuses purely on **tracking visible objects in a video using visual data alone** —without incorporating ego-motion estimation or sensor fusion.

### Gaussian blurr

To detect the edges (or borders) in images, a common approach is to blur the image and then subtract the result from the original image. This enhances areas of rapid intensity change, which correspond to edges. Blurring is typically done using a **Gaussian blur**, which is equivalent to **convolving** (i.e., applying a weighted sum operation) the image with a **Gaussian function**. In signal processing terms, this is a **low-pass filter**, because the **Fourier Transform of a Gaussian is another Gaussian**, and the convolution suppresses high-frequency components (fine details or noise) in the image.

In two dimensions, the Gaussian function is **symmetric** and defined as:

$$
G(x, y) = \frac{1}{2\pi \sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}
$$

where $x$ and $y$ are the horizontal and vertical distances from the center (or the coordinates in the convolution matrix), and $\sigma$ is the **standard deviation** that controls the spread (or "width") of the blur.

This function generates a smooth, bell-shaped surface with concentric circular contours centered around the origin. The values of this Gaussian surface form a **convolution kernel**, which is applied to the image thus getting a final image $L(x,y)=g(x,y)*f(x,y)$ with $g$ being the convolution kernel and $f$ the original image in 2D. The kernel assigns higher weights to pixels near the center and progressively smaller weights to those farther away, with values beyond $3\sigma$ typically ignored due to their negligible contribution.

During convolution, each pixel in the image is replaced by a **weighted average** of its neighborhood, where the weights come from the Gaussian function. The pixel at the center of the kernel gets the highest weight, and surrounding pixels contribute less depending on their distance.

The parameter $\sigma$ plays a crucial role:

* A **larger $\sigma$** produces a **more blurred** image by averaging over a wider area.
* A **smaller $\sigma$** preserves more detail.

Why choose the Gaussian specifically? Because it creates a smooth, **radially symmetric** (circular) weight distribution and considers the influence of neighboring pixels based on their **distance** — a natural, efficient, and mathematically elegant approach.

This Gaussian blur step is used as a **preprocessing stage** before applying the feature detection methods described next.

### Difference of Gaussians

This method consists in subtracting a **Gaussian-blurred image** from the original image. The blur is applied using a **Gaussian kernel**, which gives more weight to nearby pixels and less to distant ones.

Blurring acts as a **low-pass filter** : it smooths the image by preserving low-frequency details and removing sharp changes. Subtracting the blurred version from the original keeps only the **high-frequency components**, such as edges and small details, this is called **Difference of Gaussians (DoG)**.

A higher $\sigma$ increases the blur, removing more detail. We use this method because the Gaussian is smooth, symmetric, and gives a natural weighting based on distance. It is applied before other processing steps.

### Derivative methods

#### Laplacian of Gaussian

After applying Gaussian blur, we obtain an image $L(x,y)=g(x,y)*f(x,y)$, which is the **convolution** of a Gaussian kernel with the original image. To compute edges, we then apply the **Laplacian** to this blurred image. The Laplacian measures how rapidly pixel intensities change, making it suitable for edge detection.

**Why is blurring necessary?**

Raw images contain noise, and the Laplacian involves taking **second derivatives**, which can strongly amplify that noise. Blurring reduces noise and smooths the image before applying the Laplacian.

**Why not use the Jacobian (first derivative of intensity)?**

The Laplacian detects locations where the **gradient changes sharply or crosses zero**, which often indicates **edges or blobs**. First derivatives (Jacobian) show where change is occurring ("there is variation here"), but second derivatives highlight more significant structures like **peaks, valleys, and corners** which are key for interest point detection).

**Why are edges not enough?**

An **edge** marks a sharp intensity change like the boundary of an object but:

* Edges are **not unique** . Along a straight edge, there's no way to localize position in the direction **along the edge** . It's like standing next to a straight fence in a desert: you're next to the fence, but can't say where along it.
* Edges are **sensitive to noise** , since they're detected using **first derivatives** , which react strongly to small changes in brightness or compression artifacts.
* Edges are **ambiguous for matching** : a point on one edge in an image can match many possible points on the same edge in another image.
* The **Laplacian**, being a second derivative, is more stable for detecting **corners** , which lie at the **intersection of two edges**. It is also **isotropic** (responds equally in all directions), making it suitable for detecting features **independently of orientation**.

*Is this due to the Laplacian involving circles, making the second derivative constant?*

Indeed, the Laplacian operator in 2D involves circular symmetry. In the case of a **Gaussian**, the second derivative yields a **Laplacian of Gaussian (LoG)** that is radially symmetric—leading to consistent detection regardless of direction.

Note: this approach is gives similar results to difference of Gaussians.

#### Determinant of Hessian

This approach is similar in spirit to using the Laplacian and shares its main advantage : **it is rotation-invariant** .

Here, we compute the **determinant of the Hessian matrix** at each point to determine whether the point is an extremum and what kind of extremum it is:

* If $\det(H)>0$ then the point is a **local extremum** (either a minimum or a maximum).
* If $\det(H)<0$ the point is a **saddle point.**
* If $\det(H)=0$, the surface is **flat or ambiguous**, meaning no clear curvature is detected at that point.

| Aspect                       | **Laplacian (LoG)**                                         | **Hessian (Determinant)**                                                  |
| ---------------------------- | ----------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Definition**         | Sum of second derivatives:$f_{xx} + f_{yy}$                     | Matrix of second derivatives; use determinant:$f_{xx} f_{yy} - f_{xy}^2$       |
| **Blob measure**       | Zero-crossings or extrema of the LoG response                     | Maxima of the determinant of the Hessian                                         |
| **Isotropy**           | Isotropic (rotation invariant by default)                         | Determinant is also rotation invariant, but captures anisotropy in structure     |
| **Sensitivity**        | Responds to intensity changes (e.g. edges and blobs)              | Captures structure with consistent curvature in multiple directions (blobs only) |
| **Computational Cost** | LoG is computationally simpler, especially with DoG approximation | Requires computing all second derivatives and determinant                        |
| **Popular Use**        | Used in SIFT (via DoG), Marr-Hildreth edge detector               | Used in SURF, Hessian-based detectors (e.g. Hessian-Laplace, Hessian-Affine)     |

### Kadir–Brady saliency detector

The image must first be converted to **grayscale**. The algorithm then computes the **Shannon entropy** around each pixel, using a defined box size to determine the neighborhood for the calculation.

To do this:

1. A **histogram** of pixel intensities is built within this box.
2. The histogram is **normalized** to obtain the **probability distribution** of intensities.
3. The **entropy** is then computed as:

   $$
   -\sum_i p_i \log_2 p_i
   $$

   where $p_i$ is the proportion of pixels with intensity $i$.

This entropy value reflects how **diverse or uncertain** the region is, that is to say how much **information** it contains.

The term $\log_2 p_i$ measures the **information content** (or **surprise**) of observing intensity $i$, and the entropy itself is the **expected value** of that information (i.e., the **average information** in the region).

### Maximally stable extremal regions: accentuate contrast

Each pixel is assigned to a class based on whether it falls within a certain **threshold**, effectively dividing the image into **two groups**. This method is **lightweight** and efficient. If $n$ is the number of pixels, the process can run in $O(n)$ time using a **binary thresholding** approach.

In some cases, the image can be **divided into a grid**, and the thresholding can be computed **locally** in each cell. This allows for **adaptive thresholding**, which is more effective when lighting conditions vary across the image.

### Shi-Tomasi corner detector: feature selection

The Shi-Tomasi corner detector, is based on analyzing the local gradient structure around each pixel using the **structure tensor** (also called the second moment matrix), after applying an edge detection algorithm like difference of Gaussians to the image. For a given image $I$, at each pixel, the gradients in the $x$ and $y$ directions, $I_x$ and $I_y$, are computed. These gradients capture how image intensity changes in each direction.

From these gradients, the algorithm constructs the **structure tensor matrix** MM for a local window (usually a square neighborhood around the pixel):

$$
\begin{bmatrix}
\sum I_x^2 & \sum I_x I_y \\
\sum I_x I_y & \sum I_y^2
\end{bmatrix}
$$

where the sums are over the pixels in the local window. This matrix $M$ encodes the gradient distribution in the neighborhood: its eigenvalues $\lambda_1$ and $\lambda_2$ represent the intensity variation along two orthogonal directions.

This Shi-Tomasi corner detector differs from computing the Jacobian across the image or other derivative methods as seen before, as the structure tensor goes deeper into analyzing local variations by combining and averaging gradient information over a neighborhood. This allows it to capture the dominant directions and strength of intensity changes, making it more robust for detecting features like corners.

The Harris corner detector computes a corner response function based on $\lambda_1$ and $\lambda_2$ as:

$$
R = \det(M) - k \cdot \text{trace}(M)^2 = \lambda_1 \lambda_2 - k(\lambda_1 + \lambda_2)^2
$$

where $k$ is a sensitivity parameter.

Note : Shi and Tomasi is an improvement over the Harris corner detector. Shi and Tomasi observed that this cornerness measure is not ideal for tracking purposes. Instead, they proposed to use the **minimum eigenvalue** of $M$ directly as the corner score:

$$
R_{ST} = \min(\lambda_1, \lambda_2)
$$

The reasoning is that a good corner has large gradients in both directions, so the smallest eigenvalue reflects the weakest direction's gradient strength. If this minimum eigenvalue is above a threshold, the point is considered a strong corner.

This approach is advantageous because:

* It avoids tuning the parameter $k$ required in Harris.
* It better predicts how well a feature can be tracked, since features with a low minimum eigenvalue are unstable.

Finally, the algorithm selects points with high $R_{ST}$ values and applies non-maximum suppression to avoid clustering too many features too close together.

## Motion detection, or how to establish correspondence between images

The objective of video tracking is to associate target objects in consecutive video frames. The association can be especially difficult when the objects are moving fast relative to the frame rate. Another situation that increases the complexity of the problem is when the tracked object changes orientation over time. For these situations video tracking systems usually employ a motion model which describes how the image of the target might change for different possible motions of the object. To perform video tracking an algorithm analyzes sequential video frames and outputs the movement of targets between the frames. There are a variety of algorithms, each having strengths and weaknesses. Considering the intended use is important when choosing which algorithm to use. There are two major components of a visual tracking system: target representation and localization, as well as filtering and data association.

Structure from motion is a photogrammetric range imaging (photo and depth) technique for estimating three-dimensional structures from two-dimensional image sequences that may be coupled with local motion signals (vectors). Biological creatures use motion parallax from 2D images. This is more what we are able to do.

There are two basic ways to find the correspondences between two images :

* Correlation-based – checking if one location in one image looks/seems like another in another image.
* Feature-based – finding features in the image and seeing if the layout of a subset of features is similar in the two images. To avoid the aperture problem a good feature should have local variation in two directions.
* Multi-scale-approach - Scaling the image down to reduce the search space, then correct the coarse approximations on smaller windows. Solving the correspondence problem over a small search spaces is easily trained on a convolutional neural network.

### Lucas-Kanade optical flow

The **Lucas-Kanade** method is used to estimate **optical flow** which is the apparent motion of objects between two consecutive frames in a video. It assumes that pixel intensities of small patches remain consistent between frames and tries to compute how these patches have moved.

The method assumes that the **brightness of a pixel remains constant** between two frames:

$$
I(x, y, t) = I(x + u, y + v, t + 1)
$$

where:

* $(x, y$) is the pixel location,
* $I(x, y, t)$ is the intensity at time tt,
* $(u, v)$ is the optical flow (motion) vector we want to find.

By applying a first-order Taylor expansion on the right-hand side and dropping higher-order terms:

$$
I(x + u, y + v, t + 1) \approx I(x, y, t) + I_x u + I_y v + I_t
$$

Subtracting both sides gives:

$$
I_x u + I_y v = -I_t
$$

This is called the **optical flow constraint equation**.

This equation provides only **one equation with two unknowns ($u$ and $v$)** per pixel, which makes it **underdetermined**. To solve this, the Lucas-Kanade method assumes that **neighboring pixels** in a small window (typically 3×3 or 5×5, it is a parameters in the program) move in the same way. So it collects this equation over all pixels in a neighborhood of size $N\times N$ (with $N$ being usually $3$ or $5$):

$$
\begin{bmatrix}
I_{x1} & I_{y1} \\
I_{x2} & I_{y2} \\
\vdots & \vdots \\
I_{xN} & I_{yN}
\end{bmatrix}
\begin{bmatrix}
u \\
v
\end{bmatrix}
=
-\begin{bmatrix}
I_{t1} \\
I_{t2} \\
\vdots \\
I_{tN}
\end{bmatrix}
$$

This is an overdetermined linear system, solved via least squares method. We want to solve $A \vec{x} = \vec{b}$. Where :

* $A$ is $\begin{bmatrix}I_{x1} & I_{y1} \\I_{x2} & I_{y2} \\\vdots & \vdots \\I_{xN} & I_{yN}\end{bmatrix}$
* $\vec{x}$ is $\begin{bmatrix} u \\ v \end{bmatrix}$
* $\vec{b} = -\begin{bmatrix}I_{t1} \\I_{t2} \\\vdots \\I_{tN}\end{bmatrix}$

To find the best approximation of $\vec{v}$, minimize the squared error: $\min_{\vec{x}} \| A \vec{x} - \vec{b} \|^2$. Thus it is equivalent trying to minimize $(A\vec{x}-\vec{b})^T(A\vec{x}-\vec{b})$. 

The gradient of this expression is: $\nabla_{\vec{x}}(A\vec{x}-\vec{b})^T(A\vec{x}-\vec{b})=\nabla_{\vec x}(\vec{x}^TA^TA\vec{x}-\vec{x}^TA^T\vec{b}-\vec{b}^TA\vec{x}+\vec{b}^T\vec{b})=\nabla_{\vec x}(\vec{x}^TA^TA\vec{x}-2\vec{b}^TA\vec{x}+\vec{b}^T\vec{b})$ because as scalars, $\vec{x}^TA^T\vec{b}=\vec{b}^TA\vec{x}=(\vec{x}^TA^T\vec{b})^T$. 

Thus $\nabla_{\vec{x}}(A\vec{x}-\vec{b})^T(A\vec{x}-\vec{b})=\nabla_{\vec{x}}(\vec{x}^TA^TA\vec{x})-\nabla_{\vec{x}}(2\vec{b}^TA\vec{x})=2A^TA\vec{x}-2\vec{b}^TA$. So $\min_{\vec{x}} \| A \vec{x} - \vec{b} \|^2\iff A^TA\vec{x}=\vec{b}^TA=A^T\vec{b}$

This leads to the **normal equations**: $A^\top A \vec{x} = A^\top \vec{b}$. $I_x$ being $\begin{bmatrix}I_{x1}\\ I_{x2}\\ \vdots \\ I_{xN}\end{bmatrix}$ and $I_y$ being $\begin{bmatrix}I_{y1}\\ I_{y2}\\ \vdots\\ I_{xN}\end{bmatrix}$, we get :

$$
\begin{bmatrix}I_x^2&I_xI_y\\ I_xI_y&I_y^2\end{bmatrix}\begin{bmatrix}u\\ v\end{bmatrix}=\begin{bmatrix}-I_xI_t\\ -I_yI_t\end{bmatrix}
$$

Thus solving gives $\vec{x} = (A^\top A)^{-1} A^\top \vec{b}$

The matrix on the left is known as the **structure tensor** (same as in Shi-Tomasi), and must be invertible to compute motion.

### Correlation-based approach : direct methods

This approach addresses **optical flow**, which is the apparent motion of objects in a visual scene due to relative motion between the observer (camera) and the objects. Note that this motion can occur **even if the objects are stationary in the world** : they appear to move because the camera moves.

#### Block-matching algorithm (template matching)

The **block-matching algorithm** works by dividing each frame of a video into small rectangular regions (blocks), and then, for each block in one frame, **searching for the best match** in the next frame within a predefined search area.

The algorithm compares the block with candidate blocks in the next frame using an **error function**, such as:

* Sum of Absolute Differences (SAD)
* Sum of Squared Differences (SSD)
* Normalized Cross-Correlation (NCC)

The best match (i.e., the one that minimizes the error) gives the **displacement vector**, which estimates the motion of the block between frames. This motion reflects the movement of the **underlying object or region**, assuming brightness consistency and small temporal changes.

Template matching is sensitive to changes in **scale, rotation, and lighting**.

#### CSRT template matching

The **CSRT tracker** (*Discriminative Correlation Filter with Channel and Spatial Reliability*) is an advanced visual object tracking algorithm that builds upon the framework of **Discriminative Correlation Filters (DCFs)**, with specific modifications to improve robustness to **deformation, rotation, and partial occlusion** .

At the core of CSRT is the idea of  **correlation filtering**. A DCF learns a filter $f$ that, when convolved with the image, gives a strong peak where the object is located and low response elsewhere. This is typically done in the **Fourier domain** to speed up the convolution operation (since convolution becomes multiplication in the frequency domain).

Unlike basic DCFs that use raw grayscale pixels, CSRT computes features from **multiple image channels** :

* HoG (Histogram of Oriented Gradients)
* Color Names (quantized color labels)
* Intensity values (grayscale)

Each channel contributes to the final response map, increasing the richness of the representation and improving robustness to lighting changes and background noise.

A key innovation in CSRT is the introduction of a **spatial reliability map**. Not all pixels in the template are equally helpful for tracking (e.g., pixels near the object's edge or background can be misleading). CSRT computes a binary or continuous-valued spatial mask that weights the importance of each spatial location during training of the correlation filter.

This means the filter focuses more on **reliable, central, object-specific regions**, and less on uncertain or background areas, leading to fewer tracking failures when the background changes.

### Feature-based approach : indirect methods

#### Scale-invariant feature transform

To detect motion, we need to find correspondences between images, that is, to determine which parts of one image match parts of another. This is necessary because differences between images can result from the movement of the **camera**, the  **passage of time**, or the **movement of objects**. One common method to achieve this is the **Scale-Invariant Feature Transform (SIFT)**.

The first step in SIFT is to detect points of interest and compute their  **dominant gradient direction** . Around each keypoint, a histogram of gradient orientations is built, where each direction is weighted by its magnitude. The most prominent peak in this histogram represents the **dominant orientation** of the local patch. To ensure  **rotation invariance** , SIFT then aligns the descriptor to this dominant direction before comparison, so the keypoint is always described relative to its own orientation.

SIFT is also  **scale-invariant** , meaning it works correctly even if the image is zoomed in or out. However, the algorithm requires detecting many features in each image, and it doesn't directly compare images to find changes. Instead, it processes each image independently, focusing on the local features that can later be matched across frames.

[Source originale](https://www.ipol.im/pub/art/2014/82/?utm_source=doi)

#### Speeded up robust features

#### Condensation (conditional density propagation) algorithm

It is a probabilistic algorithm. Each pixel is not studied. Instead pixels are chosen randomly. The algorithm's creation was inspired by the inability of Kalman filtering to perform object tracking well in the presence of significant background clutter. The presence of clutter tends to produce probability distributions for the object state which are multi-modal and therefore poorly modeled by the Kalman filter. The condensation algorithm in its most general form requires no assumptions about the probability distributions of the object or measurements.

### Multi-scale-approach

### Kalman filter to correct noise in addition to other methods

It answers "Given some noisy measurements, what is the most likely state of a system?"

The algorithm works via a two-phase process: a prediction phase and an update phase. In the prediction phase, the Kalman filter produces estimates of the current state variables,
 including their uncertainties. Once the outcome of the next measurement (necessarily corrupted with some error, including random noise) is observed, these estimates are updated using a weighted average, with more weight given to estimates with greater certainty. The algorithm is recursive. It can operate in real time, using only the present input measurements and the state calculated previously and its uncertainty matrix; no additional past information is required.

Prediction step:

* You use the old frame parameters (position and speed of a point of interest in 2D) to estimate where the car should be next.
* This gives you a predicted state and a confidence level (uncertainty).

Update step:

* You take a new measurement
* You combine the predicted position and the noisy measurement, weighting each by how confident you are in them.
* The result is a better estimate than either one alone.

This comes on top of motion detection solution and corrects it

### Co-segmentation

Co-segmentation differs from segmentation in which it lasts throughout a video and not just a frame.

It is often challenging to extract segmentation masks of a target/object from a noisy collection of images or video frames, which involves object discovery coupled with segmentation. A noisy collection implies that the object/target is present sporadically in a set of images or the object/target disappears intermittently throughout the video of interest. Early methods typically involve mid-level representations such as object proposals.

First is to segment the image through superpixels. The number of clusters must not be specified, but ideally optimized. In the beginning there are a certain number of clusters which form a grid ([Grid of clusters](https://www.youtube.com/watch?v=zx1CthO5FEk)) and then it grows

Then to ensure consistency between frames, labels must stay consistent. For that, each class from each image must be added to a graph that according to its form/colours/overall appearance so that we have a graph with cliques. [Graph and cliques](https://www.youtube.com/watch?v=TdRYcZ2xUSM)

For **color images** (unlike grayscale), segmentation often uses **agglomerative clustering** based on  **color gradients** . This "bottom-up" method starts with each pixel as its own cluster and iteratively merges the most similar clusters, using a distance metric like **Euclidean distance** and a linkage criterion such as **single-linkage** or  **complete-linkage** . It continues until a single cluster remains or another **stopping condition** is reached.

Agglomerative methods are widely used in image segmentation because they are simple and effective for small to medium datasets. By considering both **spatial proximity** and  **color similarity** , they group pixels into coherent regions.

In contrast, **divisive clustering** (a "top-down" approach) starts with the entire image as one cluster and recursively splits it. This approach is more **computationally intensive** and often  **greedy** , as it makes locally optimal decisions at each step without backtracking.

## Focus

Where to focus attention where it is the most important ?

## Limits

We are at risk to segment all small parts of an image, for instance the windows of a car, as well as the handle, the wheels etc as they are all different colours

Depending if the image is full of colour or not, we may want to give a growing importance to the colour difference between pixels to define edges, so that to detect a green leaf you must regroup all green-ish pixel (even the dark ones) in a single class

Main possible issues : rotating, scaling, lighting
