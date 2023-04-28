import cv2, os, scipy.io
from PIL import Image


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

# LocalizeDigits("13.png", 100, 1000, "adaptiveGaussian", False)
# LocalizeDigits("12.png", 50, 1000, "adaptiveMean", True)

for filename in os.listdir("testImage"):
    f = os.path.join("testImage", filename)
    LocalizeDigits(f, 30, 5000, "adaptiveMean", False)
    LocalizeDigits(f, 30, 5000, "adaptiveMean", True)