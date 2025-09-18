### Loading and Querying Images
"""
Created by:  Charity Grey (2025)
Modified by:  [Your Name] (2025)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def image_properties(image_color, image_gray):
    # Display shape and datatype
    print("Color Image Shape:", image_color.shape)  # (Height, Width, Channels)
    print("Grayscale Image Shape:", image_gray.shape)
    print("Data Type:", image_color.dtype)

def show_image(img, title='', cmap=None):
    plt.title(title)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()

def resize_image(image, size=(300, 300)):
    return cv2.resize(image, size)

def rotate_image(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, M, (w, h))

def translate_image(image, tx, ty):
    (h, w) = image.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, M, (w, h))

def flip_image(image, mode='horizontal'):
    if mode == 'horizontal':
        return cv2.flip(image, 1)
    elif mode == 'vertical':
        return cv2.flip(image, 0)
    else:
        raise ValueError("mode must be 'horizontal' or 'vertical'")

def crop_image(image, x1, x2, y1, y2):
    return image[y1:y2, x1:x2]

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    img = image.astype(np.int16)
    img = img * (contrast / 127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def chain_transformations(image):
    # 1. Rotate by 30 degrees
    rotated = rotate_image(image, 30)
    # 2. Resize to 200x200
    resized = resize_image(rotated, (200, 200))
    # 3. Translate right by 40, down by 20
    translated = translate_image(resized, 40, 20)
    return translated

### Main Execution Block ###
if __name__ == "__main__":
    # Load image (color and grayscale)
    image_color = cv2.imread('../data/your_image.png')  # Replace with your image path
    image_gray = cv2.imread('../data/your_image.png', cv2.IMREAD_GRAYSCALE)

    image_properties(image_color, image_gray)

    # Convert BGR to RGB for proper display
    image_rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)

    # Show images
    show_image(image_rgb, "Color Image")
    show_image(image_gray, "Grayscale Image", cmap='gray')

    # Resize
    resized = resize_image(image_rgb)
    show_image(resized, "Resized Image")

    # Rotate grayscale
    rotated = rotate_image(image_gray, 45)
    show_image(rotated, "Rotated Image", cmap='gray')

    # Translate grayscale
    translated = translate_image(image_gray, 50, 30)
    show_image(translated, "Translated Image", cmap='gray')

    # Flip horizontal
    flipped_horizontal = flip_image(image_rgb, 'horizontal')
    show_image(flipped_horizontal, "Flipped Image (Horizontal)")

    # Flip vertical
    flipped_vertical = flip_image(image_rgb, 'vertical')
    show_image(flipped_vertical, "Flipped Image (Vertical)")

    # Crop
    cropped = crop_image(image_rgb, 400, 664, 64, 360)
    show_image(cropped, "Cropped Image")

    # Adjust brightness and contrast
    img_bc = adjust_brightness_contrast(image_rgb, brightness=40, contrast=40)
    show_image(img_bc, "Brightness & Contrast Adjusted")

    # Chain transformations
    chained = chain_transformations(image_rgb)
    show_image(chained, "Chained: Rotated → Resized → Translated")