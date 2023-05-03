import numpy as np
import cv2, os, scipy.io
from PIL import Image
from PIL import ImageEnhance
import statistics
import cv2
from numpy import asarray


def SpatialImprovement(imagePath, invBinary):
    img = cv2.imread(imagePath)
    img = cv2.medianBlur(img,3)
    reducenoise = cv2.GaussianBlur(img, (5, 5), 0)
    sharpen_filter=np.array([[-1,-1,-1],
                 [-1,9,-1],
                [-1,-1,-1]])
    sharp_image=cv2.filter2D(reducenoise,-1,sharpen_filter)
    gray = cv2.cvtColor(sharp_image, cv2.COLOR_BGR2GRAY)
    
    ret,baseline = cv2.threshold(gray,130,255,cv2.THRESH_TRUNC)
    if(invBinary):
        ret,background = cv2.threshold(baseline,125,255,cv2.THRESH_BINARY_INV)
        ret,foreground = cv2.threshold(baseline,125,255,cv2.THRESH_BINARY)
    
    else:
        ret,background = cv2.threshold(baseline,125,255,cv2.THRESH_BINARY)
        ret,foreground = cv2.threshold(baseline,125,255,cv2.THRESH_BINARY_INV) 
 
    foreground = cv2.bitwise_and(img,img, mask=foreground)  # Update foreground with bitwise_and to extract real foreground
 
    # Convert black and white back into 3 channel greyscale
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
 
    # Combine the background and foreground to obtain our final image
    finalimage = background+foreground
    sobelx = cv2.Sobel(finalimage,cv2.CV_64F,1,0)#,ksize=5)     
    sobely = cv2.Sobel(finalimage,cv2.CV_64F,0,1)#,ksize=5)
    laplacianapproximation = cv2.Laplacian(finalimage,cv2.CV_64F)
    display1 = cv2.convertScaleAbs(finalimage)
    display2 = cv2.normalize(display1, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    cv2.imshow('Sobel Transform', display2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ThresholdTrial(imagePath, lowerAreaThreshold, upperAreaThreshold, thresholdType, invBinary):
    img = cv2.imread(imagePath)

    dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5,5), 0)

    if thresholdType == "adaptiveMean":
        if invBinary:
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        else:
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    elif thresholdType == "adaptiveGaussian":
        if invBinary:
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        else:
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    elif thresholdType == "Otsu":
         if invBinary:
             _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
         else:
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        if invBinary:
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.THRESH_TRIANGLE, cv2.THRESH_BINARY_INV, 11, 2)
        else:
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.THRESH_TRIANGLE, cv2.THRESH_BINARY, 11, 2)

    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #thresh = cv2.erode(thresh, kernel, iterations=1)
    #thresh = cv2.dilate(thresh, kernel, iterations=1)
    #dim = (200, 149)
    #resized = cv2.resize(thresh, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow('Threshold Trials', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def EdgeDetection(imagePath, Canny):
    img = cv2.imread(imagePath)
    dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    sobel_filter_vertical=np.array([[-1,0,+1],
                 [-2,0,+2],
                [-1,0,+1]])
    sobel_filter_horizontal=np.array([[1,2,+1],
                 [0,0,0],
                [-1,-2,-1]])
    sharp_image_opt=cv2.filter2D(gray,-1,sobel_filter_vertical)
    sharp_image_opt=cv2.filter2D(sharp_image_opt,-1,sobel_filter_vertical)
    laplacian_filter = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])
    laplacian_filter_strong = np.array([[-1, -1, -1],
                                 [-1, 8, -1],
                                 [-1, -1, -1]])
    hbf = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])
    shbf = np.array([[-1, -1, -1],
                    [-1, 9, -1],
                    [-1, -1, -1]])
    prewitt_horizontal = np.array([[-1, 0, 1],
                                   [-1, 0, 1],
                                   [-1, 0, 1]])
    prewitt_vertical = np.array([[-1, -1, -1],
                                [0,  0,  0],
                                [1,  1,  1]])
    scharr =np.array([[-3, 0, 3],
                      [-10, 0, 10],
                      [-3, 0, 3]])
    if(Canny):
        sharp_image_opt=cv2.filter2D(gray,-1,sobel_filter_horizontal)
        sharp_image_opt=cv2.filter2D(sharp_image_opt,-1,sobel_filter_vertical)
        sharp_image_opt =cv2.filter2D(gray,-1,prewitt_horizontal)
        sharp_image_opt =cv2.filter2D(sharp_image_opt,-1,prewitt_vertical)                             
        sharp_image_opt=cv2.filter2D(sharp_image_opt,-1,laplacian_filter_strong)
        #print(sharp_image_opt)
        sharp_image_opt = Image.fromarray(sharp_image_opt)
        enhancer = ImageEnhance.Contrast(sharp_image_opt)
        factor = 3 
        sharp_image_opt = enhancer.enhance(factor)
        sharp_image_opt = asarray(sharp_image_opt)
        #sharp_image_opt =cv2.filter2D(sharp_image_opt,-1,scharr)
        sharp_image_opt=cv2.Canny(gray,50,100)

    else:
        sharp_image_opt =cv2.filter2D(gray,-1,prewitt_horizontal)
        sharp_image_opt =cv2.filter2D(gray,-1,prewitt_vertical)                             
        sharp_image_opt=cv2.filter2D(gray,-1,laplacian_filter_strong)
        #print(sharp_image_opt)
        sharp_image_opt = Image.fromarray(sharp_image_opt)
        enhancer = ImageEnhance.Contrast(sharp_image_opt)
        factor = 3 
        sharp_image_opt = enhancer.enhance(factor)
        sharp_image_opt = asarray(sharp_image_opt)
        sharp_image_opt =cv2.filter2D(gray,-1,scharr)
        sharp_image_opt=cv2.filter2D(gray,-1,shbf)

    cv2.imshow('Edge Detection', sharp_image_opt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#def CornerDetection():
#def BoundaryGeneration():

#def ROI():
#def MSER():

# images = scipy.io.loadmat("train_32x32.mat").get("X")
# for i in range(7, 14):
#     img = images[:, :, :, i]
#     img = Image.fromarray(img, 'RGB')
#     img.show()


for filename in os.listdir("PROJECT FILES/train/train"):
#for filename in range(10):
    f = os.path.join("PROJECT FILES/train/train", filename)
    #SpatialImprovement(f, False)
    #SpatialImprovement(f, True)
    #bgremove2(f)
    #medianFilter(SpatialImprovement(f))
    #ThresholdTrial(f, 30, 5000, "adaptiveMean", False)
    #ThresholdTrial(f, 30, 5000, "adaptiveMean", True)
    #ThresholdTrial(f, 30, 5000, "adaptiveGaussian", False)
    #ThresholdTrial(f, 30, 5000, "adaptiveGaussian", True)
    #ThresholdTrial(f, 30, 5000, "Otsu", False)
    #ThresholdTrial(f, 30, 5000, "Otsu", True)
    #ThresholdTrial(f, 30, 5000, "A", False)
    #ThresholdTrial(f, 30, 5000, "A", True)
    EdgeDetection(f, True)
    EdgeDetection(f, False)
