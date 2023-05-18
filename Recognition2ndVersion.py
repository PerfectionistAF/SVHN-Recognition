import math
import cv2
import os
import sys
import numpy as np
import scipy.io


def ThresholdImage(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb_planes = cv2.split(img_gray)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                 dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)

    return result_norm


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
    print("Loading Mat File..")
    mat = scipy.io.loadmat('train_32x32.mat')
    X = mat['X']
    Labels = mat['y']
    num_images = X.shape[3]
    accuracy = []
    print("Mat File Loaded!!\n")

    for i in range(1000):
        loadPercent = (i / 1000) * 100
        sys.stdout.write(
            f"\rComputing Accuracy: [{'=' * math.floor(loadPercent / 10)}{' ' * (10 - math.floor(loadPercent / 10))}] "
            f"{round(loadPercent, 1)}%")
        imgReal = X[:, :, :, i]
        label = str(Labels[i][0])

        digit = ""
        score = math.inf

        thresh = ThresholdImage(imgReal)

        predictions = []
        for idxBox, (x, y, w, h) in enumerate(LocalizeDigits(CannyEdge(thresh))):

            img = imgReal[y:y+h, x:x+w]

            for idx, filename in enumerate(os.listdir("digitTemplates")):
                template = os.path.join("digitTemplates", filename)
                digitTemplate = cv2.imread(template)

                desired_height = img.shape[0]
                aspect_ratio = digitTemplate.shape[1] / digitTemplate.shape[0]
                desired_width = int(desired_height * aspect_ratio)
                resized_image = cv2.resize(digitTemplate, (desired_width, desired_height))

                sim = MatchFeatures(resized_image, thresh[y:y+h, x:x+w])

                if sim < score:
                    score = sim
                    digit = filename.split(".")[0]
            predictions.append(digit)

        res = label in predictions
        if v: print(f"\nImage {i + 1}: Label = {label}, Predicted Outcome = {res}")
        accuracy.append(res)

    sys.stdout.write(
        f"\rComputing Accuracy: [{'=' * 10}] "
        f"{100}%")
    return sum(accuracy)/len(accuracy)


acc = testImages(False)
print(f"\n\nAccuracy: {round(acc * 100, 1)}%")
