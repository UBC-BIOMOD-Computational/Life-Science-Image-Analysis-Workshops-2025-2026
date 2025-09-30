## 2. Manipulating Imaging Arrays in NumPy and Torch

import torch

# Convert SimpleITK image to NumPy array
image_np = sitk.GetArrayFromImage(image)  # shape: (z, y, x) for 3D, (y, x) for 2D
print('NumPy array shape:', image_np.shape)

# Manipulate with NumPy (e.g., normalize)
image_np_norm = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np))
print('Normalized min/max:', image_np_norm.min(), image_np_norm.max())

# Convert to Torch tensor
image_tensor = torch.from_numpy(image_np_norm).float()
print('Torch tensor shape:', image_tensor.shape)


#data i/o
import torchio as tio

# TorchIO expects images in (C, Z, Y, X) or (C, Y, X) format
if image_tensor.ndim == 3:
    image_tensor = image_tensor.unsqueeze(0)  # Add channel dimension
elif image_tensor.ndim == 2:
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

# Create a TorchIO Subject
subject = tio.Subject(
    image=tio.ScalarImage(tensor=image_tensor)
)

# Define a data augmentation pipeline
transform = tio.Compose([
    tio.RandomFlip(axes=(0, 1)),
    tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
    tio.RandomNoise(mean=0, std=0.1),
    tio.RandomBiasField(coefficients=0.5)
])

# Apply augmentation
augmented = transform(subject)
aug_image = augmented.image.data
print('Augmented image shape:', aug_image.shape)

## 4. Visualize Original and Augmented Images (2D slice example)

import matplotlib.pyplot as plt

# Show a middle slice for 3D, or the image for 2D
def show_slice(tensor, title):
    arr = tensor.squeeze().cpu().numpy()
    if arr.ndim == 3:
        idx = arr.shape[0] // 2
        arr = arr[idx]
    plt.imshow(arr, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

show_slice(image_tensor, 'Original Image')
show_slice(aug_image, 'Augmented Image')

## 5. NumPy to PyTorch Tensors: A Comparison Guide

# For those familiar with NumPy arrays, here are the key differences and similarities with PyTorch tensors.import torch
import numpy as np

# 1. Creating arrays/tensors
print("=== Creating Arrays/Tensors ===")

# NumPy
np_array = np.array([1, 2, 3, 4, 5])
np_zeros = np.zeros((3, 4))
np_ones = np.ones((2, 3))
np_random = np.random.rand(2, 3)

# PyTorch (similar syntax)
torch_tensor = torch.tensor([1, 2, 3, 4, 5])
torch_zeros = torch.zeros(3, 4)
torch_ones = torch.ones(2, 3)
torch_random = torch.rand(2, 3)

print("NumPy array:", np_array)
print("PyTorch tensor:", torch_tensor)
print("Types:", type(np_array), type(torch_tensor))

# 2. Converting between NumPy and PyTorch
print("\n=== Converting Between NumPy and PyTorch ===")

# NumPy to PyTorch
np_to_torch = torch.from_numpy(np_array)
print("NumPy to PyTorch:", np_to_torch)

# PyTorch to NumPy
torch_to_np = torch_tensor.numpy()
print("PyTorch to NumPy:", torch_to_np)

# Note: These share memory! Changes in one affect the other
np_array[0] = 999
print("After changing np_array[0]:", np_to_torch)  # Also changed!

# To avoid shared memory, use .clone() or .copy()
safe_torch = torch.from_numpy(np_array.copy())
safe_np = torch_tensor.clone().numpy()

# 3. Basic operations - very similar syntax!
print("\n=== Basic Operations ===")

np_a = np.array([[1, 2], [3, 4]])
np_b = np.array([[5, 6], [7, 8]])

torch_a = torch.tensor([[1, 2], [3, 4]])
torch_b = torch.tensor([[5, 6], [7, 8]])

# Addition
print("NumPy addition:", np_a + np_b)
print("PyTorch addition:", torch_a + torch_b)

# Matrix multiplication
print("NumPy matmul:", np.matmul(np_a, np_b))
print("PyTorch matmul:", torch.matmul(torch_a, torch_b))
# or simply: torch_a @ torch_b

# Reshaping
print("NumPy reshape:", np_a.reshape(-1))
print("PyTorch reshape:", torch_a.reshape(-1))
# or: torch_a.view(-1)

# 4. Key differences: Device and gradients
print("\n=== Key Differences ===")

# Device (CPU vs GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Available device:", device)

torch_cpu = torch.ones(2, 3)
print("Tensor device:", torch_cpu.device)

# Move to GPU (if available)
if torch.cuda.is_available():
    torch_gpu = torch_cpu.to('cuda')
    print("GPU tensor device:", torch_gpu.device)

# Gradients (for automatic differentiation)
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()  # Compute gradients
print("Gradient of x^2 at x=2:", x.grad)  # Should be 4

# 5. Common tensor operations for image processing
print("\n=== Image Processing Operations ===")

# Create a sample "image" tensor (batch_size, channels, height, width)
image_tensor = torch.rand(1, 3, 64, 64)  # 1 RGB image, 64x64
print("Image tensor shape:", image_tensor.shape)

# Indexing (similar to NumPy)
red_channel = image_tensor[0, 0, :, :]  # First batch, red channel
print("Red channel shape:", red_channel.shape)

# Permute dimensions (like np.transpose)
# Change from NCHW to NHWC format
image_hwc = image_tensor.permute(0, 2, 3, 1)
print("NHWC format shape:", image_hwc.shape)

# Squeeze/unsqueeze (like np.squeeze/np.expand_dims)
squeezed = image_tensor.squeeze(0)  # Remove batch dimension
print("Squeezed shape:", squeezed.shape)

unsqueezed = squeezed.unsqueeze(0)  # Add batch dimension back
print("Unsqueezed shape:", unsqueezed.shape)

# 6. Working with our medical image example
print("\n=== Medical Image Tensor Operations ===")

# Our image_tensor from earlier
print("Original image tensor shape:", image_tensor.shape)
print("Data type:", image_tensor.dtype)

# Common operations you might do:

# 1. Normalize (similar to NumPy)
normalized = (image_tensor - image_tensor.mean()) / image_tensor.std()

# 2. Add batch dimension if needed
if image_tensor.dim() == 3:  # (C, H, W)
    batched = image_tensor.unsqueeze(0)  # -> (1, C, H, W)
    print("Added batch dimension:", batched.shape)

# 3. Convert data type
float_tensor = image_tensor.float()  # Ensure float32
print("Float tensor dtype:", float_tensor.dtype)

# 4. Clone for safety (like np.copy())
image_copy = image_tensor.clone()

# 5. Move to device for processing
image_on_device = image_tensor.to(device)
print("Image on device:", image_on_device.device)

## Summary: NumPy vs PyTorch Tensors

**Similarities:**
- Very similar syntax for basic operations
- Indexing, slicing, reshaping work almost identically
- Mathematical operations are nearly the same

**Key Differences:**
- **Device support**: Tensors can live on GPU for faster computation
- **Automatic differentiation**: Tensors can track gradients for deep learning
- **Memory sharing**: Converting between NumPy and PyTorch shares memory by default
- **Method names**: Some differences (e.g., `view()` vs `reshape()`, `permute()` vs `transpose()`)

**For Medical Imaging:**
- Use tensors when you need GPU acceleration
- Use tensors for deep learning models
- Convert to NumPy for visualization with matplotlib
- Be aware of dimension ordering (NCHW vs NHWC)