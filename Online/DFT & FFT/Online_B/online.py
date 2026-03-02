import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from discrete_framework import DiscreteSignal, DFTAnalyzer


def decryptImage(encrypyted_image):
    rows, cols = encrypted_image.shape

    keyIndex = np.argmin(np.sum(encrypted_image, axis=1))
    keyRow = encrypted_image[keyIndex]

    keySignal = DiscreteSignal(keyRow)
    analyzer = DFTAnalyzer()

    KEY = analyzer.compute_dft(keySignal)

    epsilon = 1e-10
    KEY[np.abs(KEY) < epsilon] = epsilon

    decryptedImage = np.zeros_like(encrypted_image, dtype=np.float64)

    for i in range(rows):
        if i == keyIndex:
            decryptedImage[i] = keyRow
            continue

        rowSignal = DiscreteSignal(encrypted_image[i])
        Y = analyzer.compute_dft(rowSignal)

        X = Y / KEY

        xRecovered = analyzer.compute_idft(X)

        decryptedImage[i] = np.real(xRecovered)

    return decryptedImage


image = Image.open("encrypted_image.tiff")

# Convert the image to a NumPy array
encrypted_image = np.array(image)

decryptedImage = decryptImage(encrypted_image)


plt.figure(figsize=(8, 6))

# Encrypted image
plt.subplot(1, 2, 1)
plt.imshow(encrypted_image, cmap='gray')
plt.title("Encrypted Image")
plt.axis('off')

# Decrypted image
plt.subplot(1, 2, 2)
plt.imshow(decryptedImage, cmap='gray')
plt.title("Decrypted Image")
plt.axis('off')

plt.show()
