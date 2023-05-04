# SVHN-Recognition
Street View House Number Recognition using computer vision and perception techniques
No machine learning/data methods were used

# Phase 1 Digit Localization:
   ## Dependencies
   ```python
   import sys
   import math
   import cv2
   import os
   import json
   import numpy as np
   import copy
   from numpy import asarray
   from PIL import ImageEnhance, Image
   ```
   ## Implemented Functionality
   ```
   • Spatial Improvement of Images
   • Thresholding of Images
   • Edge Detection of Images
   • Region Detection of Images
   • Finding a Rectangle that Encompasses all contours
   • Cropping Images to Region Of Interest
   • Corner Harris Detection of Images
   • Estimating Digit Area Based on Image Dimensions
   • Contouring and Digit Localisation in Images
   • Computing Percentage of Intersection using IOU
   ```
   
   ## Running The Code 'EdgeDetectionLocalization.py'
   #### Replace The following paths in LocalizeDir Function with the path to the folder with your test images
   ```python
   total = len(os.listdir('testImages'))
   for filename in os.listdir("testImages")
   f = os.path.join("testImages", filename)
   ```
   #### When HarshAccuracy is set to true the image contours accuracy gets affected by any extra contours that are not digits
   ```python
   HarshAccuracy = False
   ```
   #### When showSteps is set to true the images with contours displayed is shown
   ```python
   percentages = LocalizeDir(dataSet, showSteps=False)
   ```
   ## Refrences
   ```
   docs.opencv.org
   answers.opencv.org
   https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
   https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html?loclr=blogmap
   ```

# Phase 2 Digit Classification:
