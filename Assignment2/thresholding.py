import cv2
import numpy as np

# apply thresholding with threshold value of 128
threshold_value = 25
max_value = 255


def auto_threshold(img, start_thresh, step, goal=18000, max_goal=20000, max_thresh=255, flag=cv2.THRESH_BINARY):
    threshold = start_thresh
    while True:
        ret, closing = cv2.threshold(img, threshold, max_thresh, type=flag)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
        #closing = cv2.dilate(closing, kernel)

        # Get contours
        contours = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        # noise points
        small_contours = []
        for con in contours:
            area = cv2.contourArea(con)
            if 0 < area < 9000:
                small_contours.append(con)

        # max contour: foreground
        max_contour = max(contours, key=cv2.contourArea)
        max_area = cv2.contourArea(max_contour)
        if max_area >= goal:
            if max_area >= max_goal:
                threshold = threshold + step
                ret, closing = cv2.threshold(img, threshold, max_thresh, type=flag)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
            cv2.drawContours(closing, small_contours, -1, (0, 0, 0), cv2.FILLED)
            break
        else:
            threshold = threshold - step

    return closing, threshold


def threshold(hsv, flag=cv2.THRESH_BINARY):
    i = 0
    outputs = []
    for file in hsv:
        # load input image in grayscale
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        # increase contrast
        ret, contrast = cv2.threshold(img, 10, 255, type=cv2.THRESH_TOZERO)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(10, 10))
        cl1 = clahe.apply(contrast)
        blurred = cv2.GaussianBlur(cl1, (9, 9), 0.8, 0.8)

        thresh, t = auto_threshold(blurred, 100, 2)
        print(t)
        #ret, thresh = cv2.threshold(blurred, 10, max_value, type=flag)

        # display thresholded image
        cv2.imshow('Thresholded Image', thresh)
        cv2.waitKey(0)
        #cv2.imwrite("step2\cam1\diff\Diff_threshold" + "_" + str(i) + ".png", thresh)
        # release resources and close windows
        outputs.append(thresh)
        i = i + 1
    a = cv2.bitwise_and(outputs[0], outputs[1])
    a = cv2.bitwise_and(outputs[2], a)
    # apply morphology open then close
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, kernel, iterations=4)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    a = cv2.morphologyEx(a, cv2.MORPH_OPEN, kernel, iterations=2)

    # Get contours
    contours = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    # max contour: foreground
    max_contour = max(contours, key=cv2.contourArea)

    cv2.drawContours(a, contours, -1, (0, 0, 0), cv2.FILLED)
    cv2.drawContours(a, [max_contour], -1, (255, 255, 255), cv2.FILLED)

    cv2.imshow('Thresholded Image', a)
    cv2.imwrite("step2\cam4\diff\Diff_threshold.png", a)
    cv2.destroyAllWindows()
    return a

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
