# Image Filtering & Feature Matching

Two computer-vision notebooks exploring local image filters and correspondence between photographs using OpenCV, NumPy, and Matplotlib.

[Image filtering](Assignment-task1COMP338.ipynb) · [Feature matching](Assignment-task2COMP338.ipynb) · [Assignment document](Assighment1-338%20-PDF%20file.pdf)

## What the notebooks explore

| Notebook | Focus | Saved visual output |
| --- | --- | --- |
| `Assignment-task1COMP338.ipynb` | Custom 3 × 3 filtering loops for edge detection, blur, sharpening, and embossing | Four original/filtered image comparisons |
| `Assignment-task2COMP338.ipynb` | ORB keypoint detection and SIFT/ORB descriptor matching with a ratio test | Keypoint visualizations and side-by-side match diagrams |

The filtering notebook makes the per-pixel operation explicit. The matching notebook moves from detecting distinctive points to drawing candidate correspondences between `victoria1.jpg` and `victoria2.jpg`. Its commentary also discusses SURF; SURF is not implemented here.

## View the existing results

![Saved SIFT and ORB correspondence diagrams between two building photographs](assets/feature-matches.png)

*Original saved output from the matching notebook. These are exploratory matches, not verified ground-truth correspondences.*

Open the notebooks on GitHub to inspect their saved image outputs without executing them. The matching notebook records 14 accepted SIFT matches and 6 accepted ORB matches for its image pair and settings. These counts describe that saved run and are not a general performance comparison.

## Run locally

Install the notebook tools and the libraries imported by the source:

```bash
python -m pip install jupyter numpy matplotlib opencv-python
```

Place `victoria1.jpg` and `victoria2.jpg` beside the notebooks. **The original JPEG files are not included in this repository.** You can use your own image pair by updating the filenames in the cells, but the resulting figures and match counts will differ.

Open the notebooks with Jupyter. The second notebook assumes imports already exist in the kernel; add this cell before running it independently:

```python
import cv2
import matplotlib.pyplot as plt
```

Dependency versions are not pinned, and these installation instructions have not been validated by rerunning the notebooks.

## Implementation notes

- The custom filter leaves border pixels at zero, does not flip the kernel, and stores output in the input image's integer dtype. It illustrates local filtering but is not a general, numerically robust convolution implementation.
- The matching code uses one default `BFMatcher` for both methods. ORB's binary descriptors should be evaluated with a Hamming-distance matcher before drawing comparative conclusions.
- The image-resizing code assigns `shape[:2]` to variables named width and height in reversed order; check image dimensions before using this as a reusable pipeline.

This is an exploratory coursework implementation with saved visual examples. Source and existing outputs were inspected for this documentation update; the image-processing code was not rerun.
