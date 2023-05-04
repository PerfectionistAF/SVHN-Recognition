# SVHN-Recognition
Street View House Number Recognition using computer vision and perception techniques
No machine learning/data methods were used

## Project Phases
### Phase one: 
   ### Segmentation and Boundary Creation
   ### Stages of Image Processing:
   ### 1) Spatial Improvement of Images:
   def SpatialImprovement(imagePath, invBinary):
   ```
   imagePath: path of images
   invBinary: bool, whether image threshold to invBinary or not
   ```
   1) Median blur
   2) Gaussian blur
   3) Sharpening Box filter
   4) Grayscale
   5) Identify background/baseline/foreground and remove backgorund
   6) Sobel filteration
   7) Laplacian Approximation
   8) Normalisation
   
   ### 2) Thresholding of Images:
  def ThresholdTrial(imagePath, lowerAreaThreshold, upperAreaThreshold, thresholdType, invBinary):
  ```
  imagePath: path of images
  lowerAreaThreshold: lower threshold
  upperAreaThreshold:upper threshold
  thresholdType: string, type of threshold used
  invBinary: bool, whether image threshold to invBinary or not
  ```
  1) Denoising coloured image
  2) Grayscale
  3) Gaussian Blur
  4) Adaptive Mean Thresholding
  5) Adaptive Gaussian Thresholding
  6) Otsu Thresholding
  7) Triangle Thresholding
  
   ### 3) Edge Detection of Images:
  def EdgeDetection(imagePath, Canny):
  ```
  imagePath: path of images
  Canny: bool, whether or not Canny edge detection was used
  ```
  1) Denoising of coloured image
  2) Grayscale
  3) Sobel filter
  4) Laplacian filter
  5) High bost filter
  6) Super high boost filter
  7) Prewitt filter
  8) Scharr filter
  9) Increase Contrast 
  10) Canny edge detection

  
### Phase two: 
    Identification
# Dependencies
```python
from __future__ import print_function
import numpy as np
import cv2, os, scipy.io
from PIL import Image
from PIL import ImageEnhance
import statistics
import cv2
from numpy import asarray
import imutils
```

docs.opencv.org
answers.opencv.org
https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html?loclr=blogmap

