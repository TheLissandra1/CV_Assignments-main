import cv2
import numpy as np

# apply thresholding with threshold value of 128
threshold_value = 25
max_value = 255


def get_contour(img, min_c, flag=cv2.RETR_EXTERNAL):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # Get contours
    contours = cv2.findContours(img, flag, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # noise points
    small_contours = []
    for con in contours:
        area = cv2.contourArea(con)
        if 0 < area < min_c:
            small_contours.append(con)

    # max contour: foreground
    max_contour = max(contours, key=cv2.contourArea)

    return img, contours, max_contour, small_contours


# increase contrast and use cv2.threshold to set white and black area
def get_thresh(img, black_thresh, white_thresh):
    ret, thresh = cv2.threshold(img, black_thresh, 255, type=cv2.THRESH_TOZERO)

    # increase contrast
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(10, 10))
    thresh = clahe.apply(thresh)

    for i in range(thresh.shape[0]):
        for j in range(thresh.shape[1]):
            if thresh[i, j] > white_thresh:
                thresh[i, j] = 255

    return thresh


def clean_foreground(img, small_contours):
    # delete white noise area
    cv2.drawContours(img, small_contours, -1, (0, 0, 0), cv2.FILLED)
    # fill black holes in foreground
    img, cos, max_c, small_cs = get_contour((255 - img), 1000, flag=cv2.RETR_TREE)
    img = (255 - img)
    cv2.drawContours(img, small_cs, -1, (255, 255, 255), cv2.FILLED, maxLevel=6)

    return img


def auto_threshold(img, step, goal=15000, max_goal=60000, max_thresh=255):
    mean, std = cv2.meanStdDev(img)
    black = int(mean + 3)
    white = int(mean * 10 + 10)
    temp_thresh = None
    while True:
        thresh = get_thresh(img, black, white)
        thresh = get_thresh(thresh, black + 1, white)
        thresh = get_thresh(thresh, black + 2, white)

        ret, thresh = cv2.threshold(thresh, 0, max_thresh, type=cv2.THRESH_OTSU)

        im, contours, max_contour, small_contours = get_contour(thresh, 2000)

        max_area = cv2.contourArea(max_contour)
        if max_area >= goal:
            if max_area >= max_goal:
                if temp_thresh is None:
                    temp_thresh = thresh
                white = white + step
                thresh = temp_thresh
                im, contours, max_contour, small_contours = get_contour(thresh, 2000)

            im = clean_foreground(im, small_contours)
            break
        else:
            white = white - step
            temp_thresh = thresh

    return im, white


def threshold(hsv):
    i = 0
    outputs = []
    cv2.namedWindow('Thresholded Image', 0)

    for file in hsv:
        # load input image in grayscale
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, [1000, 1000])

        thresh, t = auto_threshold(img, 1)
        print(t)

        # display thresholded image
        cv2.resizeWindow('Thresholded Image', 500, 500)
        cv2.imshow('Thresholded Image', thresh)
        cv2.waitKey(0)
        # cv2.imwrite("step2\cam1\diff\Diff_threshold" + "_" + str(i) + ".png", thresh)
        # release resources and close windows
        outputs.append(thresh)
        i = i + 1

    t1 = cv2.bitwise_and(outputs[0], outputs[1])
    t2 = cv2.bitwise_and(outputs[0], outputs[2])
    t3 = cv2.bitwise_and(outputs[1], outputs[2])

    result = cv2.bitwise_or(t1, t2)
    result = cv2.bitwise_or(result, t3)

    # apply morphology open then close
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=3)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=2)

    result, cos, max_c, small_cs = get_contour(result, 6000)
    result = clean_foreground(result, small_cs)

    cv2.imshow('Thresholded Image', result)
    cv2.imwrite("step2\cam4\diff\Diff_threshold.png", result)
    cv2.destroyAllWindows()
    return result


Diff_H = 'step2\cam4\diff\Diff_H.png'
Diff_S = 'step2\cam4\diff\Diff_S.png'
Diff_V = 'step2\cam4\diff\Diff_V.png'
hsv_images = [Diff_H, Diff_S, Diff_V]
thresh_V = threshold(hsv_images)

'''
# cam2
cam2_Diff_V = 'step2\cam2\diff\Diff_V.png'
thresh_V = threshold(img=cam2_Diff_V, method='threshold', type=cv2.THRESH_BINARY)
thresh_V = threshold(img=cam2_Diff_V, method='threshold', type=cv2.THRESH_OTSU)
opening = cv2.morphologyEx(thresh_V, cv2.MORPH_OPEN, kernel, iterations=2)
cv2.imshow("opening", opening)
cv2.imwrite("step2\cam2\diff\Diff_V_threshold.png", opening)
# closing = cv2.morphologyEx(thresh_V,cv2.MORPH_CLOSE, kernel, iterations=2 )
# cv2.imshow("closing", closing)

# cam3
cam3_Diff_V = 'step2\cam3\diff\Diff_V.png'
thresh_V = threshold(img=cam3_Diff_V, method='threshold', type=cv2.THRESH_BINARY)
thresh_V = threshold(img=cam3_Diff_V, method='threshold', type=cv2.THRESH_OTSU)
opening = cv2.morphologyEx(thresh_V, cv2.MORPH_OPEN, kernel, iterations=2)
cv2.imshow("opening", opening)
cv2.imwrite("Assignment2\step2\cam3\diff\Diff_V_threshold.png", opening)
# closing = cv2.morphologyEx(thresh_V,cv2.MORPH_CLOSE, kernel, iterations=2 )
# cv2.imshow("closing", closing)


# # cam4
cam4_Diff_V = 'Assignment2\step2\cam4\diff\Diff_V.png'
thresh_V = threshold(img=cam4_Diff_V, method='threshold', type=cv2.THRESH_BINARY)
thresh_V = threshold(img=cam4_Diff_V, method='threshold', type=cv2.THRESH_OTSU)

# closing = cv2.morphologyEx(thresh_V,cv2.MORPH_CLOSE, kernel, iterations=2 )
# cv2.imshow("closing", closing)
opening = cv2.morphologyEx(thresh_V, cv2.MORPH_OPEN, kernel, iterations=1)
cv2.imshow("opening", opening)
cv2.imwrite("Assignment2\step2\cam4\diff\Diff_V_threshold.png", opening)
# eroding = cv2.erode(opening,  kernel)
# cv2.imshow("eroding", eroding)


cv2.waitKey(0)
# release resources and close windows
cv2.destroyAllWindows()

# generally Diff_V channel is the best channel to threshold, thresh value could be set between 40-55
# erosion or dilation is needed to remove small white areas.

# opening operation


# combine channel images together

'''
