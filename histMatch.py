import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms

# Load source and reference images (grayscale)
source = cv2.imread('/Users/ayushgautam/Downloads/WhatsApp Image 2025-03-30 at 16.34.37.jpeg', cv2.IMREAD_GRAYSCALE)
reference = cv2.imread('/Users/ayushgautam/Downloads/WhatsApp Image 2025-03-30 at 16.34.37.jpeg', cv2.IMREAD_GRAYSCALE)

# Perform histogram matching
matched = match_histograms(source, reference, channel_axis=None)

# Function to plot histograms
def plot_histogram(image, title):
    hist = cv2.calcHist([image.astype('uint8')], [0], None, [256], [0, 256])
    plt.plot(hist, label=title)

# Plot images
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.imshow(source, cmap='gray')
plt.title("Source Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(reference, cmap='gray')
plt.title("Reference Image")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(matched, cmap='gray')
plt.title("Matched Image")
plt.axis("off")

# Plot histograms
plt.subplot(2, 3, 4)
plot_histogram(source, "Source Histogram")
plot_histogram(reference, "Reference Histogram")
plot_histogram(matched, "Matched Histogram")
plt.title("Histogram Comparison")
plt.legend()

plt.tight_layout()
plt.show()
