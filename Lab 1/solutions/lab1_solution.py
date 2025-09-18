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

# Resize image
resized = cv2.resize(image_rgb, (300, 300))

plt.title("Resized Image")
plt.imshow(resized)
plt.axis('off')
plt.show()

# Rotate image
(h, w) = image_gray.shape
center = (w // 2, h // 2)

# Rotate 45 degrees
M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(image_gray, M, (w, h))

plt.title("Rotated Image")
plt.imshow(rotated, cmap='gray')
plt.axis('off')
plt.show()

# Transform Image

# Move image right by 50, down by 30 pixels
M = np.float32([[1, 0, 50], [0, 1, 30]])
translated = cv2.warpAffine(image_gray, M, (w, h))

plt.title("Translated Image")
plt.imshow(translated, cmap='gray')
plt.axis('off')
plt.show()

# Flip image horizontally
flipped_horizontal = cv2.flip(image_rgb, 1)
plt.title("Flipped Image (Horizontal)")
plt.imshow(flipped_horizontal)
plt.axis('off')
plt.show()

# Flip image vertically
flipped_vertical = cv2.flip(image_rgb, 0)
plt.title("Flipped Image (Vertical)")
plt.imshow(flipped_vertical)
plt.axis('off')
plt.show()

# Crop image (e.g., top-left 100x100 region)
cropped = image_rgb[0:100, 0:100]
plt.title("Cropped Image (100x100)")
plt.imshow(cropped)
plt.axis('off')
plt.show()

# Adjust brightness and contrast
# brightness: -127 to 127, contrast: -127 to 127
brightness = 40
contrast = 40
img_bc = image_rgb.astype(np.int16)
img_bc = img_bc * (contrast / 127 + 1) - contrast + brightness
img_bc = np.clip(img_bc, 0, 255).astype(np.uint8)
plt.title("Brightness & Contrast Adjusted")
plt.imshow(img_bc)
plt.axis('off')
plt.show()

# Chain transformations: rotate → resize → translate.
# 1. Rotate image_rgb by 30 degrees
(h, w) = image_rgb.shape[:2]
center = (w // 2, h // 2)
M_rotate = cv2.getRotationMatrix2D(center, 30, 1.0)
rotated_rgb = cv2.warpAffine(image_rgb, M_rotate, (w, h))

# 2. Resize rotated image to 200x200
resized_rotated = cv2.resize(rotated_rgb, (200, 200))

# 3. Translate resized image: right by 40, down by 20 pixels
M_translate = np.float32([[1, 0, 40], [0, 1, 20]])
translated_final = cv2.warpAffine(resized_rotated, M_translate, (200, 200))

plt.title("Chained: Rotated → Resized → Translated")
plt.imshow(translated_final)
plt.axis('off')
plt.show()