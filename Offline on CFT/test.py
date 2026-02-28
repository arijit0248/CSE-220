from os import path
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread

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

        # Define continuous spatial axes
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
    using separability and numerical integration.
    """

    def __init__(self, image_obj: ContinuousImage):
        self.I = image_obj.image
        self.x = image_obj.x
        self.y = image_obj.y

    def compute_cft(self):
        """
        Compute real and imaginary parts of 2D CFT.
        Simplified using vectorized operations.
        """
        rows, cols = self.I.shape
        real = np.zeros((cols, rows))
        img = np.zeros((cols, rows))

        # Create meshgrids for spatial coordinates
        X, Y = np.meshgrid(self.x, self.y, indexing='xy')

        for i, ui in enumerate(self.x):
            for j, vj in enumerate(self.y):
                print(f"u: {i}/{cols} v: {j}/{rows}")

                # Compute phase for all spatial points at once
                phase = 2 * np.pi * (ui * X + vj * Y)

                # Compute integrands
                cos_integrand = self.I * np.cos(phase)
                sin_integrand = self.I * np.sin(phase)

                # Double integration using trapezoid rule
                real[i, j] = np.trapezoid(np.trapezoid(
                    cos_integrand, self.y, axis=0), self.x)
                img[i, j] = - \
                    np.trapezoid(np.trapezoid(
                        sin_integrand, self.y, axis=0), self.x)

        return real, img

    def plot_magnitude(self):
        """
        Plot log-scaled magnitude spectrum.
        Simplified to compute magnitude directly.
        """
        real, img = self.compute_cft()
        magnitude = np.sqrt(real**2 + img**2)

        plt.imshow(np.log(1 + magnitude), cmap='magma',
                   extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]])
        plt.title("2D Magnitude Spectrum")
        plt.axis('off')
        plt.show()


# =====================================================
# Frequency Filtering
# =====================================================
class FrequencyFilter:
    def low_pass(self, real, imag, cutoff):
        rows, cols = real.shape
        cx, cy = rows//2, cols//2

        for i in range(rows):
            for j in range(cols):
                if np.sqrt((i-cx)**2 + (j-cy)**2) > cutoff:
                    real[i, j] = 0
                    imag[i, j] = 0
        return real, imag

# =====================================================
# Inverse 2D Continuous Fourier Transform
# =====================================================


class InverseCFT2D:
    """
    Reconstructs image from 2D frequency spectrum.
    """

    def __init__(self, real, imag, x, y):
        self.real = real
        self.imag = imag
        self.x = x
        self.y = y

    def reconstruct(self):
        """
        Perform inverse 2D CFT using numerical integration.
        Simplified using vectorized operations.
        """
        cols, rows = self.real.shape
        reconstructed = np.zeros((cols, rows))

        # Create meshgrid for frequency coordinates
        U, V = np.meshgrid(self.x, self.y, indexing='xy')

        for i, xi in enumerate(self.x):
            for j, yj in enumerate(self.y):
                print(f"x: {i}/{cols} y: {j}/{rows}")

                # Compute phase for all frequency points at once
                phase = 2 * np.pi * (U * xi + V * yj)

                # Compute integrand
                integrand = self.real * \
                    np.cos(phase) - self.imag * np.sin(phase)

                # Double integration
                reconstructed[i, j] = np.trapezoid(
                    np.trapezoid(integrand, self.y, axis=0), self.x)

        return reconstructed


# =====================================================
# Main Execution (Task 2)
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
