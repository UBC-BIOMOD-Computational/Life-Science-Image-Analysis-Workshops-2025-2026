### Histogram Processing and Equalization
"""
Created by:  Charity Grey (2025)
Modified by:  [Your Name] (2025)
"""


# Histogram of grayscale image
hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])

plt.figure()
plt.title("Grayscale Histogram")
plt.plot(hist, color='black')
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.grid()
plt.show()


# Improve contrast using histogram equalization
equalized = cv2.equalizeHist(image_gray)

# Show before and after
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title('Original')
plt.imshow(image_gray, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title('Equalized')
plt.imshow(equalized, cmap='gray')
plt.axis('off')
plt.show()
