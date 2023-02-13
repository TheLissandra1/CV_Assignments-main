# import numpy as np
# import cv2 as cv
# import glob

# path = "Assignment1\calib-checkerboard.png"
# img = cv.imread(path)

# cv.namedWindow('image', 0)
# cv.resizeWindow('image', 248, 351)
# cv.imshow('image', img)
# cv2.waitKey(0)

import cv2
import numpy as np
import glob
import os, sys
import re


def task1():
    print("hello world")
    # fname = "Assignment1\Checkerboards\C1.png"
    # termination criteria:
    # 30 = loops
    # 0.001 ?
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # size of checkerboard: a*b
    a = 9
    b = 6
    objp = np.zeros((a*b,3), np.float32)
    '''
    设定世界坐标下点的坐标值,因为用的是棋盘可以直接按网格取；
    假定棋盘正好在x-y平面上, 这样z值直接取0,简化初始化步骤。
    mgrid把列向量[0:cbraw]复制了cbcol列,把行向量[0:cbcol]复制了cbraw行。
    转置reshape后,每行都是4*6网格中的某个点的坐标。
    '''
    objp[:,:2] = np.mgrid[0:a,0:b].T.reshape(-1,2)

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # all image file names with .png suffix
    path = "Assignment1\Checkerboards\*.png"
    image_fnames = glob.iglob(path)
    # print(image_fnames)

    # iterate all training images
    for i_fname in image_fnames:
        print(i_fname)
        # read source image
        img = cv2.imread(i_fname)
    
        # view img
        cv2.namedWindow('image', 0)
        cv2.resizeWindow('image', 1000, 1000)
        cv2.imshow("image", img)
        cv2.waitKey(0)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
        # find corners and 
        # ret: flag of corners, boolean type
        ret,corners = cv2.findChessboardCorners(gray,(a,b),None)
        print(ret)

        if ret == True:
            objpoints.append(objp)

            # execute sub-pixel corner detection
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img,(a,b),corners2,ret)
            cv2.namedWindow('ChessboardCorners', 0)
            cv2.resizeWindow('ChessboardCorners', 1000, 1000)
            cv2.imshow("ChessboardCorners", img)
            cv2.waitKey(0)
            # save image with corners
            if i_fname.endswith('.png'):  
                new_fname = i_fname.replace('Checkerboards', "Result")
                print(new_fname)
                new_fname = new_fname.replace('.png','_Corners.png')
                print(new_fname)
            cv2.imwrite(new_fname, img)

    # Calibration
    # to estimate the intrinsic and extrinsic parameters of the camera.
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # return camera matrix, distortion coefficients, rotation and translation vectors etc.
    print("camera matrix:", mtx)
    print("distortion", dist)

    # ############## Save camera parameters
    np.savez("Assignment1\CameraData\camera_Data.npz", mtx=mtx, dist=dist,rvecs=rvecs,tvecs=tvecs )

    # # load the camera matrix and distortion coefficients from the previous calibration result.
    # npzfile = np.load("camera_Data.npz")
    # sorted(npzfile.files)









    


    # Validate the calibration
    # Undistortion
    img = cv2.imread('Assignment1\Checkerboards\C1.png')
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]
    # print(type(dst))
    # print(dst)
    cv2.namedWindow('undistorted result', 0)
    cv2.resizeWindow('undistorted result', 1000, 1000)
    cv2.imshow("undistorted result", dst)
    cv2.waitKey(0)
    cv2.imwrite(r'Assignment1\Undistortions\C1_Result.png', dst)
    # does not work for cb5


    # Re-projection Error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "total error: {}".format(mean_error/len(objpoints)) )    



if __name__== "__main__" :
    task1()
