### Geometric Transformations
"""
Created by:  Charity Grey (2025)
Modified by:  [Your Name] (2025)
"""

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
