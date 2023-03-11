import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# STEPS: 
# divide image into 3 hsv channels
# use H (hue) channel as it includes the color information
# draw histogram of H channel
# find the highest peak of the histogram
# simply query X for which Y is maxmized 
# More advanced methods work with windows - they average the Y-values of 10 consecutive data points, etc.

# get the highest peak hue color, apply it in color model 


if __name__ == '__main__':

    img = 'step1\cam4_sample.png'
    # show hsv image
    img = cv2.imread(img)
    cv2.imshow("image_RGB", img)
    cv2.waitKey(0)
    
    # convert from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow("image_HSV", hsv)
    cv2.waitKey(0)
    
    h = hsv[:,:,0]
    print(type(hsv)) # <class 'numpy.ndarray'>
    print(hsv.shape) # (486, 644, 3)
    s = hsv[:,:,1]
    v = hsv[:,:,2]

    # print("h channel: \n", h)
    print(h.shape) # (486, 644)
    # print(type(h)) # <class 'numpy.ndarray'>
    

    # histogram of h
    # use histogram() provided by numpy
    # in opencv, hsv range from 0 to 180
    hist,bins = np.histogram(h.ravel(),180,[0,180])  # return type: # <class 'numpy.ndarray'>, len(hist):180

    # return the index of highest peak in histogram
    dominant_Hue = np.argmax(hist)
    print("Dominant Color: ", dominant_Hue) 

    
    plt.plot(hist)
    plt.show()
    h_new = np.full((h.shape), dominant_Hue)
    print(h_new.shape)
    print(h_new)
    # # change h with new h
    hsv[:, :, 0] = h_new

    # s,v channel compute average
    s_avg = np.rint(np.average(s.ravel())).astype(int)
    s_new = np.full((s.shape), s_avg)
    v_avg = np.rint(np.average(v.ravel())).astype(int)
    v_new = np.full((s.shape), v_avg)
    hsv[:, :, 1] = s_new
    hsv[:, :, 2] = v_new

    dominant_Color = np.uint8([[[dominant_Hue, s_avg, v_avg]]])
    print("Dominant Color hsv value:" , dominant_Color)
    # convert to rgb values
    dominant_Color_rgb = cv2.cvtColor(dominant_Color, cv2.COLOR_HSV2RGB)
    print(dominant_Color_rgb)

    cv2.imshow("dominant color: new hsv", hsv)
    cv2.waitKey(0)

    cv2.destroyAllWindows()