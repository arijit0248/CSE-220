import numpy as np
import matplotlib.pyplot as plt
from discrete_framework import DiscreteSignal, BluesteinFFT


def compute_cross_correlation_1d(signal1, signal2):
    """
    Compute cross-correlation of signal1 with signal2.
    Result peak at index m means: signal2 = roll(signal1, m)
    i.e. signal2[n] = signal1[n - m]  =>  signal1 is shifted RIGHT by m to get signal2
    """
    fft = BluesteinFFT()
    s1 = DiscreteSignal(signal1.astype(np.float64))
    s2 = DiscreteSignal(signal2.astype(np.float64))
    F1 = fft.compute_dft(s1)
    F2 = fft.compute_dft(s2)
    # Cross-correlation: corr(m) = sum_n s1[n] * s2[n+m]
    # In freq domain: IFFT(conj(F1) * F2)
    cross_spectrum = np.conj(F1) * F2
    corr = fft.compute_idft(cross_spectrum)
    return np.real(corr)


def detect_shift(corr, N):
    """
    Peak at index m means shifted_signal = roll(original, m).
    To reverse: roll by -m.
    Wrap-around: m > N//2 means negative shift.
    """
    peak_idx = int(np.argmax(corr))
    if peak_idx > N // 2:
        return peak_idx - N
    return peak_idx


def best_row_idx(img1, img2):
    """Row with high variance in BOTH images."""
    combined = np.minimum(np.var(img1, axis=1), np.var(img2, axis=1))
    return int(np.argmax(combined))


def best_col_idx(img1, img2):
    """Column with high variance in BOTH images."""
    combined = np.minimum(np.var(img1, axis=0), np.var(img2, axis=0))
    return int(np.argmax(combined))


def to_gray(img):
    if img.ndim == 3:
        return np.mean(img[:, :, :3], axis=2)
    return img.copy()


# ── Load images ────────────────────────────────────────────────────────────────
image = plt.imread("image.png")
shifted_image = plt.imread("shifted_image.png")

image_gray = to_gray(image)
shifted_gray = to_gray(shifted_image)
rows, cols = image_gray.shape

# ── Select best rows/columns from BOTH images ──────────────────────────────────
row_idx = best_row_idx(image_gray, shifted_gray)
col_idx = best_col_idx(image_gray, shifted_gray)
print(
    f"Using row {row_idx} for horizontal shift, col {col_idx} for vertical shift")

# ── Detect shifts ──────────────────────────────────────────────────────────────
# corr(m): peak at m means shifted = roll(original, m)
corr_h = compute_cross_correlation_1d(
    image_gray[row_idx, :], shifted_gray[row_idx, :])
col_shift = detect_shift(corr_h, cols)

corr_v = compute_cross_correlation_1d(
    image_gray[:, col_idx], shifted_gray[:, col_idx])
row_shift = detect_shift(corr_v, rows)

print(f"Detected horizontal shift (columns): {col_shift}")
print(f"Detected vertical shift   (rows):    {row_shift}")

# ── Reverse shift ──────────────────────────────────────────────────────────────
# shifted = roll(original, shift)  =>  reversed = roll(shifted, -shift)
reversed_shifted_image = np.roll(shifted_gray, -row_shift, axis=0)
reversed_shifted_image = np.roll(reversed_shifted_image, -col_shift, axis=1)

# ── Plot ───────────────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(image_gray, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(shifted_gray, cmap='gray')
plt.title("Shifted Image")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(reversed_shifted_image, cmap='gray')
plt.title("Reversed Shifted Image")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.plot(corr_h)
plt.axvline(np.argmax(corr_h), color='r', linestyle='--')
plt.title(f"Horizontal Cross-Correlation\ncol_shift={col_shift}")
plt.xlabel("Lag")

plt.subplot(2, 3, 5)
plt.plot(corr_v)
plt.axvline(np.argmax(corr_v), color='r', linestyle='--')
plt.title(f"Vertical Cross-Correlation\nrow_shift={row_shift}")
plt.xlabel("Lag")

plt.subplot(2, 3, 6)
diff = np.abs(image_gray.astype(np.float64) -
              reversed_shifted_image.astype(np.float64))
plt.imshow(diff, cmap='hot')
plt.title(f"Difference (Original vs Reversed)\nMax diff: {diff.max():.4f}")
plt.axis('off')
plt.colorbar()

plt.tight_layout()
plt.savefig("result.png", dpi=150)
plt.show()
