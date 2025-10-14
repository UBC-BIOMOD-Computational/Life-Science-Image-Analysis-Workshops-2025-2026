# Life-Science-Image-Analysis-Workshops-2025-2026

- Created by Charity Grey and Chloe Nguyen for BIOMOD Computational Team 2025-2026

**Pre-requisites:**

- Basic GitHub skills
- Basic Python skills
- Basic Python library management and virtual env creation [workshop](https://www.youtube.com/watch?v=Y21OR1OPC9A)
  - Each of the labs will have a `requirements.txt` file, please install the required packages ahead in your venv

**Pre-requisite Practice:**

- Download Python
- Find your python version
- Create your first virtual environment
- Download some libraries
- Activate your environment
- Deactivate your enviroment
- Delete your environment

## Lab 1: Image Processing Fundamentals in Python
**Worktime: Sept 24 - Oct 8**

**DISCUSSION LEADER on Oct 8:** Jack Xu
****
- use cv2 to load and show images in rgb and grayscale
- see image properties
- understand how you can use image properties to create transformations across image

## Lab 2:
**Worktime: Oct 1 - Oct 15**

**DISCUSSION LEADER on Oct 15:** todo sign up here
- histogram equalization
- Convolutions and Kernels
- Name common kernels
- Analyze how kernels affect the image

### ASIDE
- getting jupyter lab together 
- opening lab 3 in jupyter lab

## Lab 3: Miscellaneous
**Worktime: Oct 15 - Oct 22**

**DISCUSSION LEADER on Oct 22:** todo sign up here
- DICOM
- APIs (briefly)
- Loading images with SimpleITK 
- Using scikit-image for thresholding + masking

## Lab 4: Image Segmentation  
**Worktime: Oct 22 - Oct 29**

**DISCUSSION LEADER on Oct 29:** todo sign up here
 - what is image segmentation even
   
    Object detection vs instance segmentation vs semantic segmentation
    Exploring tools used to annotate images for label generation
    Learning traditional methods: Atlas-based, thresholding-based, and watershed-based methods

## Lab 5: Image Segmentation  
**Worktime: Oct 30 - Nov 5 **

**DISCUSSION LEADER on Nov 5:** todo sign up here

-  Segmentation with deep learning

#### unknown
- Manipulating imaging arrays in NumPy/Torch
- Data augmentation on 2D/3D images using TorchIO


# Credits:

We would like to thank Bioinformatics.ca for providing a comphrehensive course outline for us to develop training material on 
- https://bioinformatics.ca/workshops-all/2024-machine-learning-for-medical-imaging-analysis-pixels-to-predictions-toronto-on/#course-outline


```python
# for Charity to duplicate notebooks 
jupyter nbconvert --ClearOutputPreprocessor.enabled=True \
  --to notebook --output=lab3 lab3_solutions.ipynb
```