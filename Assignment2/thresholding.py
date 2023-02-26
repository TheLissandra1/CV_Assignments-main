import cv2
import numpy as np

# apply thresholding with threshold value of 128
threshold_value = 50
max_value = 255
# for threshold()
# type = cv2.THRESH_BINARY, cv2.THRESH_OTSU
# THRESH_BINARY = 0
# THRESH_BINARY_INV = 1
# THRESH_MASK = 7
# THRESH_OTSU = 8
# THRESH_TOZERO = 3
# THRESH_TOZERO_INV = 4
# THRESH_TRIANGLE = 16
# THRESH_TRUNC = 2

# for adaptiveThreshold()
# adaptiveMethod =  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.CALIB_CB_ADAPTIVE_THRESH
# type = cv2.THRESH_BINARY, "cv2.THRESH_BINARY_INV"
def threshold(img, method, type = cv2.THRESH_BINARY):
    # load input image in grayscale
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (9,9),0.8, 0.8)
    cv2.imshow("blurred", blurred)
    if method == "threshold":
        ret, thresh = cv2.threshold(blurred, threshold_value, max_value, type = type)
    elif method == "adaptiveThreshold":
        thresh = cv2.adaptiveThreshold(blurred, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         type, 51, -15)

    # display thresholded image
    cv2.imshow('Thresholded Image', thresh)
    cv2.waitKey(0)

    # release resources and close windows
    cv2.destroyAllWindows()
    return thresh


kernel = np.ones((3,3),np.uint8)
# cam1
Diff_H = 'Assignment2\step2\cam1\hsv\Diff_H.png'
Diff_S = 'Assignment2\step2\cam1\hsv\Diff_S.png'
cam1_Diff_V = 'Assignment2\step2\cam1\hsv\Diff_V.png'
# thresh_H = threshold(img=Diff_H, method='threshold', type = cv2.THRESH_BINARY)
# thresh_S = threshold(img=Diff_S, method='threshold', type = cv2.THRESH_BINARY)
# thresh_V = threshold(img=cam1_Diff_V, method='threshold', type = cv2.THRESH_BINARY)
# thresh_H = threshold(img=Diff_H, method='threshold', type = cv2.THRESH_OTSU)
# thresh_S = threshold(img=Diff_S, method='threshold', type = cv2.THRESH_OTSU)

# thresh_V = threshold(img=cam1_Diff_V, method='threshold', type=cv2.THRESH_TRIANGLE)
# thresh_V = threshold(img=cam1_Diff_V, method='threshold', type=cv2.THRESH_TRUNC)
# thresh_V = threshold(img=cam1_Diff_V, method='threshold', type=cv2.THRESH_TOZERO)
# thresh_V = threshold(img=cam1_Diff_V, method='adaptiveThreshold', type=cv2.THRESH_BINARY)
# thresh_V = threshold(img=cam1_Diff_V, method='threshold', type=cv2.THRESH_BINARY)
thresh_V = threshold(img=cam1_Diff_V, method='threshold', type=cv2.THRESH_OTSU)
# closing = cv2.morphologyEx(thresh_V,cv2.MORPH_CLOSE, kernel, iterations=2 )
# cv2.imshow("closing", closing)
opening = cv2.morphologyEx(thresh_V, cv2.MORPH_OPEN, kernel, iterations=2)
cv2.imshow("opening", opening)
cv2.imwrite("Assignment2\step2\cam1\hsv\Diff_V_threshold.png", opening)



# cam2
cam2_Diff_V = 'Assignment2\step2\cam2\hsv\Diff_V.png'
thresh_V = threshold(img=cam2_Diff_V, method='threshold', type = cv2.THRESH_BINARY)
thresh_V = threshold(img=cam2_Diff_V, method='threshold', type = cv2.THRESH_OTSU)
opening = cv2.morphologyEx(thresh_V, cv2.MORPH_OPEN, kernel, iterations=2)
cv2.imshow("opening", opening)
cv2.imwrite("Assignment2\step2\cam2\hsv\Diff_V_threshold.png", opening)
# closing = cv2.morphologyEx(thresh_V,cv2.MORPH_CLOSE, kernel, iterations=2 )
# cv2.imshow("closing", closing)

# cam3
cam3_Diff_V = 'Assignment2\step2\cam3\hsv\Diff_V.png'
thresh_V = threshold(img=cam3_Diff_V, method='threshold', type = cv2.THRESH_BINARY)
thresh_V = threshold(img=cam3_Diff_V, method='threshold', type = cv2.THRESH_OTSU)
opening = cv2.morphologyEx(thresh_V, cv2.MORPH_OPEN, kernel, iterations=2)
cv2.imshow("opening", opening)
cv2.imwrite("Assignment2\step2\cam3\hsv\Diff_V_threshold.png", opening)
# closing = cv2.morphologyEx(thresh_V,cv2.MORPH_CLOSE, kernel, iterations=2 )
# cv2.imshow("closing", closing)


# # cam4
cam4_Diff_V = 'Assignment2\step2\cam4\hsv\Diff_V.png'
thresh_V = threshold(img=cam4_Diff_V, method='threshold', type = cv2.THRESH_BINARY)
thresh_V = threshold(img=cam4_Diff_V, method='threshold', type = cv2.THRESH_OTSU)

# closing = cv2.morphologyEx(thresh_V,cv2.MORPH_CLOSE, kernel, iterations=2 )
# cv2.imshow("closing", closing)
opening = cv2.morphologyEx(thresh_V, cv2.MORPH_OPEN, kernel, iterations= 1)
cv2.imshow("opening", opening)
cv2.imwrite("Assignment2\step2\cam4\hsv\Diff_V_threshold.png", opening)
# eroding = cv2.erode(opening,  kernel)
# cv2.imshow("eroding", eroding)


cv2.waitKey(0)
# release resources and close windows
cv2.destroyAllWindows()


# generally Diff_V channel is the best channel to threshold, thresh value could be set between 40-55
# erosion or dilation is needed to remove small white areas.

# opening operation



# combine channel images together

# evaluation of thresholding

