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



# Apply Gaussian Blur
blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)

plt.title("Gaussian Blurred")
plt.imshow(blurred, cmap='gray')
plt.axis('off')
plt.show()


# Sobel X and Y gradients
sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)

# Convert to absolute
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

# Combine
edges = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

plt.title("Sobel Edge Detection")
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.show()
