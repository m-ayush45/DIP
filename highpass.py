import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
image = cv2.imread('/Users/ayushgautam/Downloads/WhatsApp Image 2025-03-30 at 16.34.37.jpeg', cv2.IMREAD_GRAYSCALE)

# 3x3 High-Pass Kernel
kernel_3x3 = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)

# 5x5 High-Pass Kernel (stronger)
kernel_5x5 = np.array([
    [-1, -1, -1, -1, -1],
    [-1, 1,  2,  1, -1],
    [-1, 2,  4,  2, -1],
    [-1, 1,  2,  1, -1],
    [-1, -1, -1, -1, -1]
], dtype=np.float32)

# Apply convolution with both kernels
filtered_3x3 = cv2.filter2D(image, -1, kernel_3x3)
filtered_5x5 = cv2.filter2D(image, -1, kernel_5x5)

# Display results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Grayscale")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(filtered_3x3, cmap='gray')
plt.title("High-Pass Filter (3x3)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(filtered_5x5, cmap='gray')
plt.title("High-Pass Filter (5x5)")
plt.axis("off")

plt.tight_layout()
plt.show()
