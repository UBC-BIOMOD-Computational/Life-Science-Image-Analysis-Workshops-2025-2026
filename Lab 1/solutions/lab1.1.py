### Loading and Querying Images
"""
Created by:  Charity Grey (2025)
Modified by:  [Your Name] (2025)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image (color and grayscale)
image_color = cv2.imread('your_image.jpg')  # Replace with your image path
image_gray = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)

# Display shape and datatype
print("Color Image Shape:", image_color.shape)  # (Height, Width, Channels)
print("Grayscale Image Shape:", image_gray.shape)
print("Data Type:", image_color.dtype)

# Convert BGR to RGB for proper display
image_rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)

# Show images
plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
plt.title('Color Image')
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Grayscale Image')
plt.imshow(image_gray, cmap='gray')
plt.axis('off')
plt.show()
