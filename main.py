import numpy as np
import cv2, os, scipy.io
from PIL import Image
import statistics
import cv2

def SpatialImprovement(imagePath, invBinary):
    img = cv2.imread(imagePath)
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


def LocalizeDigits(imagePath, lowerAreaThreshold, upperAreaThreshold, thresholdType, invBinary):
    img = cv2.imread(imagePath)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

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
    else:
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if lowerAreaThreshold < area < upperAreaThreshold:
            digit_contours.append(contour)

    for contour in digit_contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Localized Digits', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# images = scipy.io.loadmat("train_32x32.mat").get("X")
# for i in range(7, 14):
#     img = images[:, :, :, i]
#     img = Image.fromarray(img, 'RGB')
#     img.show()

#LocalizeDigits("13.png", 100, 1000, "adaptiveGaussian", False)
#LocalizeDigits("12.png", 50, 1000, "adaptiveMean", True)

for filename in os.listdir("PROJECT FILES/train/train"):
#for filename in range(10):
    f = os.path.join("PROJECT FILES/train/train", filename)
    SpatialImprovement(f, False)
    SpatialImprovement(f, True)
    #bgremove2(f)
    #medianFilter(SpatialImprovement(f))
    #LocalizeDigits(f, 30, 5000, "adaptiveMean", False)
    #LocalizeDigits(f, 30, 5000, "adaptiveMean", True)'''