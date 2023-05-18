import math
import cv2
import os
import sys
import numpy as np
import json


def loadDataSet(file_path: str):
    # Open the file in read-only mode
    f = open(file_path, 'r')
    # Load the contents of the file as JSON data
    data = json.load(f)
    # Return the loaded data
    return data


def NormalizeImage(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgbPlanes = cv2.split(gray)
    normalizedPlanes = []
    for plane in rgbPlanes:
        dilatedImage = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        blurredImage = cv2.medianBlur(dilatedImage, 21)
        planeDifferenceImage = 255 - cv2.absdiff(plane, blurredImage)
        normalizedImage = cv2.normalize(planeDifferenceImage, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                        dtype=cv2.CV_8UC1)
        normalizedPlanes.append(normalizedImage)
    normalizedResult = cv2.merge(normalizedPlanes)
    return normalizedResult


def ExtractFeatures(img):
    sift = cv2.SIFT_create()
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        keyPoints, descriptors = sift.detectAndCompute(img, None)
        return keyPoints, descriptors
    keyPoints, descriptors = sift.detectAndCompute(gray, None)
    return keyPoints, descriptors


def MatchFeatures(img1, img2):
    try:
        bruteForceMatcher = cv2.BFMatcher()

        keyPoints1, descriptors1 = ExtractFeatures(img1)
        keyPoints2, descriptors2 = ExtractFeatures(img2)

        matches = bruteForceMatcher.knnMatch(descriptors1, descriptors2, k=2)

        optimizedMatches = []
        for firstImageMatch, secondImageMatch in matches:
            if firstImageMatch.distance < 1 * secondImageMatch.distance:
                optimizedMatches.append(firstImageMatch)

        similarity_scores = [match.distance for match in optimizedMatches]
        max_distance = max(similarity_scores)
        min_distance = min(similarity_scores)
        normalized_scores = [(max_distance - score) / ((max_distance - min_distance) + 0.0000001) for score in similarity_scores]

        matched_image = cv2.drawMatches(img1, keyPoints1, img2, keyPoints2, optimizedMatches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv2.imshow('Digit', matched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return sum(normalized_scores)/len(normalized_scores)
    except:
        return math.inf


def Estimate_digit_area(image_size):
    # Estimate the maximum and minimum sizes of the digits based on the image size
    max_digit_height = int(image_size[0] * 0.8)  # assume maximum digit height is 80% of the image height
    aspect_ratio = [0.38, 0.51, 0.54, 0.53, 0.55, 0.58, 0.53, 0.47, 0.57, 0.52]  # aspect ratio of digits 0-9
    # Assume maximum digit width is 90% of the image width, adjusted by the maximum aspect ratio
    max_digit_width = int(image_size[1] * 0.9 * max(aspect_ratio))
    # Assume minimum digit height is 10% of the image height
    min_digit_height = int(image_size[0] * 0.1)
    # Assume minimum digit width is 10% of the image width, adjusted by the minimum aspect ratio
    min_digit_width = int(image_size[1] * 0.1 * min(aspect_ratio))

    # Calculate the approximate maximum and minimum area of the digit contours based on the estimated sizes
    max_digit_area = (max_digit_height * max_digit_width)
    min_digit_area = (min_digit_height * min_digit_width)

    return min_digit_area, max_digit_area


def CannyEdge(img):
    # Reduce noise using bilateral filter
    dst = cv2.bilateralFilter(img, 9, 75, 75)

    # Apply Canny edge detection algorithm
    canny = cv2.Canny(dst, 50, 100)

    # Return the output image
    return canny


def LocalizeDigits(img):
    # Find contours of the input image with external retrieval mode.
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Estimate the minimum and maximum area of digit contours based on the input image size using a separate function.
    minArea, maxArea = Estimate_digit_area(img.shape)

    # Create an empty list to store final contours.
    finalContours = []

    # Iterate through all contours found earlier and keep only those that have an area between minArea and maxArea.
    # Append the bounding rectangle of each selected contour to the finalContours list.
    for contour in contours:
        area = cv2.contourArea(contour)
        if minArea < area < maxArea:
            finalContours.append(cv2.boundingRect(contour))

    # Return the list of bounding rectangles for localized digits.
    return finalContours


def testImages(v=False):
    print("Loading DataSet File..")
    dataSet = loadDataSet('training.json')
    print("DataSet File Loaded!!\n")
    accuracy = []
    total = len(os.listdir("testImages"))
    i = 0
    for filename in os.listdir("testImages"):
        i += 1
        loadPercent = (i / total) * 100
        if not v:
            sys.stdout.write(
                f"\rComputing Accuracy: [{'=' * math.floor(loadPercent / 10)}{' ' * (10 - math.floor(loadPercent / 10))}]"
                f"{round(loadPercent, 1)}%")
        imgReal = cv2.imread(os.path.join("testImages", filename))
        boxes = dataSet[int(filename.split(".")[0]) - 1]['boxes']

        normalizedImage = NormalizeImage(imgReal)

        for idxBox, box in enumerate(boxes):
            (x, y, w, h) = int(box['left']), int(box['top']), int(box['width']), int(box['height'])
            img = imgReal[y:y+h, x:x+w]
            label = str(box['label']).split(".")[0]
            digit = ""
            score = math.inf
            for idx, digitFilename in enumerate(os.listdir("digitTemplates")):
                template = os.path.join("digitTemplates", digitFilename)
                digitTemplate = cv2.imread(template)

                desired_height = img.shape[0]
                aspect_ratio = digitTemplate.shape[1] / digitTemplate.shape[0]
                desired_width = int(desired_height * aspect_ratio)
                resized_image = cv2.resize(digitTemplate, (desired_width, desired_height))

                sim = MatchFeatures(resized_image, normalizedImage[y:y+h, x:x+w])
                if sim < score:
                    score = sim
                    digit = digitFilename.split(".")[0]

            if v:
                image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                aspect_ratio = image.shape[1] / image.shape[0]
                image = cv2.resize(image, (int(500 * aspect_ratio), 500))
                image = cv2.putText(image, digit, (image.shape[1] // 2, image.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                                    3, (0, 255, 0), 5, cv2.LINE_AA)
                cv2.imshow('Digit', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                print(f"\nImage {filename.split('.')[0]}, Box:{idxBox}: Label = {label},"
                      f" Predicted Outcome = {digit}")
            accuracy.append(digit == label)

    if not v:
        sys.stdout.write(f"\rComputing Accuracy: [{'=' * 10}] 100%")
    return sum(accuracy)/len(accuracy)


acc = testImages(False)
print(f"\n\nAccuracy: {round(acc * 100, 1)}%")
