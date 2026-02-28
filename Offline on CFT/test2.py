import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread

# =====================================================
# Continuous Image Class
# =====================================================


class ContinuousImage:
    """
    Represents an image as a continuous 2D signal.
    """

    def __init__(self, image_path):
        self.image = imread(image_path, mode='L')
        self.image = self.image / np.max(self.image)

        # Continuous spatial axes
        self.x = np.linspace(-1, 1, self.image.shape[1])
        self.y = np.linspace(-1, 1, self.image.shape[0])

    def show(self, title="Image"):
        plt.imshow(self.image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()


# =====================================================
# 2D Continuous Fourier Transform Class
# =====================================================

class CFT2D:
    """
    Computes 2D Continuous Fourier Transform
    using numerical integration.
    """

    def __init__(self, image_obj: ContinuousImage):
        self.I = image_obj.image
        self.x = image_obj.x
        self.y = image_obj.y

    def compute_cft(self):
        rows, cols = self.I.shape

        # Continuous frequency axes (chosen manually)
        self.u = np.linspace(-10, 10, cols)
        self.v = np.linspace(-10, 10, rows)

        real = np.zeros((rows, cols))
        imaginary = np.zeros((rows, cols))

        X, Y = np.meshgrid(self.x, self.y)

        for i in range(rows):
            for j in range(cols):
                phase = 2 * np.pi * (self.u[j] * X + self.v[i] * Y)

                cos_term = self.I * np.cos(phase)
                sin_term = self.I * np.sin(phase)

                y_cos = np.trapezoid(cos_term, self.y, axis=0)
                y_sin = np.trapezoid(sin_term, self.y, axis=0)

                real[i, j] = np.trapezoid(y_cos, self.x)
                imaginary[i, j] = -np.trapezoid(y_sin, self.x)

        self.real = real
        self.imaginary = imaginary
        return real, imaginary

    def plot_magnitude(self):
        magnitude = np.sqrt(self.real**2 + self.imaginary**2)
        log_mag = np.log(1 + magnitude)

        # Center DC component safely (works for odd/even sizes)
        shifted = np.roll(log_mag, log_mag.shape[0]//2, axis=0)
        shifted = np.roll(shifted, log_mag.shape[1]//2, axis=1)

        plt.figure(figsize=(8, 8))
        plt.imshow(shifted, cmap='hot')
        plt.title("2D CFT Magnitude Spectrum")
        plt.axis('off')
        plt.show()


# =====================================================
# Frequency Filtering
# =====================================================

class FrequencyFilter:
    def low_pass(self, real, imag, cutoff):
        rows, cols = real.shape
        cx, cy = rows // 2, cols // 2

        for i in range(rows):
            for j in range(cols):
                if (i - cx)**2 + (j - cy)**2 > cutoff**2:
                    real[i, j] = 0
                    imag[i, j] = 0

        return real, imag


# =====================================================
# Inverse 2D Continuous Fourier Transform
# =====================================================

class InverseCFT2D:
    """
    Reconstructs image from frequency domain.
    """

    def __init__(self, real, imag, x, y):
        self.real = real
        self.imag = imag
        self.x = x
        self.y = y

    def reconstruct(self):
        rows, cols = self.real.shape

        u = np.linspace(-10, 10, cols)
        v = np.linspace(-10, 10, rows)

        reconstructed = np.zeros((len(self.y), len(self.x)))

        U, V = np.meshgrid(u, v)

        for i in range(len(self.y)):
            for j in range(len(self.x)):
                phase = 2 * np.pi * (U * self.x[j] + V * self.y[i])

                cos_term = np.cos(phase)
                sin_term = np.sin(phase)

                integrand = self.real * cos_term - self.imag * sin_term

                v_int = np.trapezoid(integrand, v, axis=0)
                reconstructed[i, j] = np.trapezoid(v_int, u)

        # Normalize
        reconstructed -= reconstructed.min()
        reconstructed /= reconstructed.max() + 1e-10

        return reconstructed


# =====================================================
# Main Execution (UNCHANGED)
# =====================================================

img = ContinuousImage("noisy_image.png")
img.show("Original Image")

cft2d = CFT2D(img)
real, imag = cft2d.compute_cft()
cft2d.plot_magnitude()

filt = FrequencyFilter()
real_f, imag_f = filt.low_pass(real, imag, cutoff=40)

icft2d = InverseCFT2D(real_f, imag_f, img.x, img.y)
denoised = icft2d.reconstruct()

plt.imshow(denoised, cmap='gray')
plt.title("Reconstructed (Denoised) Image")
plt.axis('off')
plt.show()
