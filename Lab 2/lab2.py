### Histogram Processing and Equalization
"""
Created by:  Charity Grey (2025)
Modified by:  [Your Name] (2025)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image (grayscale)
image_gray = '../data/fruit.png' # TODO: read in grayscale

# --- Histogram of Grayscale Image ---
# A histogram shows the distribution of pixel intensities in the image.
# It helps visualize contrast, brightness, and intensity spread.
hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])

plt.figure()
plt.title("Grayscale Histogram")
plt.plot(hist, color='black')
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.grid()
plt.show()

# --- Histogram Equalization ---
# Histogram equalization improves the global contrast of images.
# It spreads out the most frequent intensity values, making features more distinguishable.
equalized = cv2.equalizeHist(image_gray)

# Show before and after equalization
# TODO show original and equalized images side by side


# After watching this first 2min30secs of the below video, answer the questions: 
#  https://www.youtube.com/watch?v=yb2tPt0QVPY

# What is a convolution
# Ans: TODO

# What is a kernel
# Ans: TODO

'''
If you had:
- an input slice of 10x10
- a 3x3 kernel
- stride of 1
What would be the output size after applying the kernel to the input slice using valid padding (no padding)?
'''
# Ans: TODO


# --- Kernel ---
# A kernel (or filter) is a small matrix used to apply effects like blurring, sharpening, or edge detection.
# It is convolved with the image to produce the desired effect.
example_kernel = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]]) / 9  # Simple averaging kernel
print("Kernel:\n", example_kernel)


# manually using Numpy to apply the kernel for stride of 1
def apply_kernel(image, kernel):
    # TODO: annotate each line of this function, note! there is a bug :)
    (h, w) = image.shape[:2]  
    kh, kw = kernel.shape     

    # Output dimensions
    out_h = h - kh      
    out_w = w - kw    
    output = np.zeros((out_h, out_w), dtype="float32")  

    for y in range(out_h):    
        for x in range(out_w): 

            region = image[y:y + kh, x:x + kw]  

            output[y, x] = np.sum(region * kernel)  

            output[y, x] = np.clip(output[y, x], 0, 255) 

    return output

# Apply the kernel to the image using our function
filtered_manual = apply_kernel(image_gray, example_kernel)

# --- Kernel application with cv2 ---
# Apply the kernel to the image using filter2D
filtered_cv2 = cv2.filter2D(image_gray, -1, example_kernel)

# compare filtered manual with cv2's filter2D function
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Filtered Image (Manual with custom kernel)")
#TODO

plt.subplot(1,2,2)
plt.title("Filtered Image - example_kernel")
plt.imshow(filtered_cv2, cmap='gray')
#TODO


# --- Gaussian Blur ---
# Watch this 2 minute video demonstrating Gaussian blurs: https://www.youtube.com/watch?v=-AuwMJAqjJc
# Gaussian blur smooths the image by averaging pixels with their neighbors, weighted by a Gaussian kernel.
# It is useful for noise reduction and pre-processing before edge detection.
blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)

# Compare Gaussian blur image with normal filter image
#TODO


# --- Sobel Edge Detection ---
# The Sobel operator detects edges by calculating gradients in the X and Y directions.
# It highlights regions with high spatial derivatives, i.e., edges.
sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)

# Convert to absolute values for display
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

# Combine X and Y gradients to get overall edge magnitude
edges = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)


#TODO Show Sobel edge detection result


# Explain how histogram equalization is different from convolution with a kernel (spatial filtering)
# Ans: TODO