"""Small, independently testable operations used by the coursework notebooks."""
from pathlib import Path
import cv2
import numpy as np


def read_image(path, grayscale=False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Image not found: {path}. Supply your own image or the original coursework photograph.')
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot decode image: {path}')
    return image


def resize_pair(image1, image2):
    """Match widths without enlarging either image or distorting its aspect ratio."""
    width = min(image1.shape[1], image2.shape[1])
    return tuple(cv2.resize(image, (width, max(1, round(image.shape[0] * width / image.shape[1]))))
                 for image in (image1, image2))


def ratio_matches(descriptors1, descriptors2, binary=False, ratio=0.8):
    if descriptors1 is None or descriptors2 is None or len(descriptors1) == 0 or len(descriptors2) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING if binary else cv2.NORM_L2)
    pairs = matcher.knnMatch(descriptors1, descriptors2, k=2)
    return [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]


def custom_convolution(image, kernel):
    """True 2D convolution, with zero padding and signed floating-point output."""
    image = np.asarray(image, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    if image.ndim != 2 or image.size == 0 or kernel.ndim != 2 or kernel.size == 0:
        raise ValueError('A nonempty grayscale image and 2D kernel are required')
    if any(size % 2 == 0 for size in kernel.shape):
        raise ValueError('Kernel dimensions must be odd')
    if not np.isfinite(image).all() or not np.isfinite(kernel).all():
        raise ValueError('Image and kernel must be finite')
    ph, pw = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)))
    flipped = np.flip(kernel)
    output = np.empty_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = np.sum(padded[i:i + kernel.shape[0], j:j + kernel.shape[1]] * flipped)
    return output
