import cv2
import os
import json
import numpy as np


def loadDataSet(file_path: str):
    f = open(file_path, 'r')
    data = json.load(f)
    return data


def SpatialImprovement(img, invBinary):
    reduceNoise = cv2.GaussianBlur(img, (5, 5), 0)
    sharpen_filter = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
    sharp_image = cv2.filter2D(reduceNoise, -1, sharpen_filter)
    gray = cv2.cvtColor(sharp_image, cv2.COLOR_BGR2GRAY)

    ret, baseline = cv2.threshold(gray, 130, 255, cv2.THRESH_TRUNC)
    if invBinary:
        ret, background = cv2.threshold(baseline, 125, 255, cv2.THRESH_BINARY_INV)
        ret, foreground = cv2.threshold(baseline, 125, 255, cv2.THRESH_BINARY)

    else:
        ret, background = cv2.threshold(baseline, 125, 255, cv2.THRESH_BINARY)
        ret, foreground = cv2.threshold(baseline, 125, 255, cv2.THRESH_BINARY_INV)

    # Update foreground with bitwise_and to extract real foreground
    foreground = cv2.bitwise_and(img, img, mask=foreground)

    # Convert black and white back into 3 channel greyscale
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

    # Combine the background and foreground to obtain our final image
    finalImage = background + foreground
    display1 = cv2.convertScaleAbs(finalImage)
    display2 = cv2.normalize(display1, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    cv2.imshow('SpatialImprovement', display2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ThresholdImage(img, thresholdType, invBinary, showSteps=False):

    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    if showSteps:
        cv2.imshow('Noise Reduction', dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    if showSteps:
        cv2.imshow('GrayScale', gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if thresholdType == "adaptiveMean":
        if invBinary:
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        else:
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    elif thresholdType == "adaptiveGaussian":
        if invBinary:
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        else:
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    elif thresholdType == "Otsu":
        if invBinary:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        if invBinary:
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.THRESH_TRIANGLE, cv2.THRESH_BINARY_INV, 11, 2)
        else:
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.THRESH_TRIANGLE, cv2.THRESH_BINARY, 11, 2)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # thresh = cv2.erode(thresh, kernel, iterations=1)
    # thresh = cv2.dilate(thresh, kernel, iterations=1)

    if showSteps:
        cv2.imshow('ThresholdImage', thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return thresh


def LocalizeDigits(img, lowerAreaThreshold, upperAreaThreshold):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if lowerAreaThreshold < area < upperAreaThreshold:
            digit_contours.append(contour)

    finalContours = []
    for contour in digit_contours:
        finalContours.append(cv2.boundingRect(contour))

    return finalContours


def getIntersectionPercentage(myOutput, realOutput):
    img1 = cv2.imread('black.png')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    _, img1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img2 = cv2.imread('black.png')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    _, img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for (x, y, w, h) in realOutput:
        cv2.rectangle(img1, (x, y), (x + w, y + h), (255), 2)
    for (x, y, w, h) in myOutput:
        cv2.rectangle(img2, (x, y), (x + w, y + h), (255), 3)
    interSection = cv2.bitwise_and(img1, img2)

    percentage = (np.sum(interSection == 255) /
                  (np.sum(img1 == 255) + np.sum(img2 == 255) - np.sum(interSection == 255))) * 100
    return percentage


def LocalizeDir(dataset, thresholdType, optimize, showSteps):
    if thresholdType not in ['adaptiveMean', 'adaptiveGaussian', 'Otsu', 'Other']:
        print('Please Enter Valid Threshold Type\nEx: adaptiveMean, adaptiveGaussian, Otsu, Other')
        return

    for filename in os.listdir("testImage"):
        f = os.path.join("testImage", filename)
        imgReal = cv2.imread(f)

        myOutputBin = LocalizeDigits(
            ThresholdImage(imgReal, thresholdType, False, showSteps=showSteps), 0, 100000)
        myOutputInv = LocalizeDigits(
            ThresholdImage(imgReal, thresholdType, True, showSteps=showSteps), 0, 100000)
        realOutput = []

        for box in dataset[int(filename.split(".")[0]) - 1]['boxes']:
            realOutput.append((int(box['left']), int(box['top']), int(box['width']), int(box['height'])))

        binPer = getIntersectionPercentage(myOutputBin, realOutput)
        invPer = getIntersectionPercentage(myOutputInv, realOutput)

        if binPer > invPer:
            if optimize:
                i = 0
                LocalizeDigits(
                    ThresholdImage(imgReal, thresholdType, False, False), i + 5, 10000)
                while getIntersectionPercentage(LocalizeDigits(
                        ThresholdImage(imgReal, thresholdType, False, False), i + 5, 10000), realOutput) >= binPer:
                    i += 5

                j = 10000
                while getIntersectionPercentage(LocalizeDigits(
                        ThresholdImage(imgReal, thresholdType, False, False), i, j - 100), realOutput) >= binPer:
                    j -= 100

                myOutputBin = LocalizeDigits(
                    ThresholdImage(imgReal, thresholdType, False, False), i, j)

            for (x, y, w, h) in myOutputBin:
                cv2.rectangle(imgReal, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            if optimize:
                i = 0
                while getIntersectionPercentage(LocalizeDigits(
                        ThresholdImage(imgReal, thresholdType, True, False), i + 5, 10000), realOutput) >= invPer:
                    i += 5

                j = 10000
                while getIntersectionPercentage(LocalizeDigits(
                        ThresholdImage(imgReal, thresholdType, True, False), i, j - 100), realOutput) >= invPer:
                    j -= 100

                myOutputInv = LocalizeDigits(
                    ThresholdImage(imgReal, thresholdType, True, False), i, j)

            for (x, y, w, h) in myOutputInv:
                cv2.rectangle(imgReal, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Localized Digits', imgReal)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


dataSet = loadDataSet('training.json')
LocalizeDir(dataSet, 'adaptiveMean', optimize=False, showSteps=True)
