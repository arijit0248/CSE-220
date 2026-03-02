import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from discrete_framework import DiscreteSignal, FastFourierTransform, BluesteinFFT


def best_shared_row(img1, img2):
    var1 = np.var(img1, axis=1)
    var2 = np.var(img2, axis=1)
    return int(np.argmax(var1 * var2))  # high variance in BOTH


def best_shared_col(img1, img2):
    var1 = np.var(img1, axis=0)
    var2 = np.var(img2, axis=0)
    return int(np.argmax(var1 * var2))


def to_grayscale(img):
    """Convert an image array (H,W), (H,W,3), or (H,W,4) to grayscale float64 (H,W)."""
    arr = np.array(img, dtype=np.float64)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] >= 3:
        # RGB -> luminance
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        return 0.299 * r + 0.587 * g + 0.114 * b
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def findShift1d(signal1, signal2):
    N = len(signal1)

    s1 = DiscreteSignal(signal1)
    s2 = DiscreteSignal(signal2)

    analyzer = BluesteinFFT()
    A = analyzer.compute_dft(s1)
    B = analyzer.compute_dft(s2)

    C = A * np.conj(B)

    c = analyzer.compute_idft(C)

    shift = int(np.argmax(np.abs(c)))

    if shift > N//2:
        shift -= N

    return shift


def findImageShift(original, shifted):
    rowIndex = best_shared_row(original, shifted)
    colIndex = best_shared_col(original, shifted)

    horizontalShift = findShift1d(original[rowIndex, :], shifted[rowIndex, :])
    verticalShift = findShift1d(original[:, colIndex], shifted[:, colIndex])

    return horizontalShift, verticalShift


def reallignImage(shiftedImage, dx, dy):
    realligned = np.roll(shiftedImage, -dy, axis=0)
    realligned = np.roll(realligned, -dx, axis=1)
    return realligned


# implement the necessary functions here

image = plt.imread("image.png")
shifted_image = plt.imread("shifted_image.png")
imageGray = to_grayscale(image)
shiftedGray = to_grayscale(shifted_image)

dx, dy = findImageShift(imageGray, shiftedGray)
reversed_shifted_image = reallignImage(shiftedGray, dx, dy)

diff = np.abs(imageGray-reversed_shifted_image)

plt.figure(figsize=(12, 8))

# Original Image
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Shifted Image
plt.subplot(2, 3, 2)
plt.imshow(shifted_image, cmap='gray')
plt.title(f"Shifted Image")
plt.axis('off')


# Reversed Shifted Image
plt.subplot(2, 3, 3)
plt.imshow(reversed_shifted_image, cmap='gray')
plt.title("Reversed Shifted Image")
plt.axis('off')

plt.tight_layout()
plt.show()
