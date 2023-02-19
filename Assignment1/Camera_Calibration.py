import cv2
import numpy as np
import glob
import os, sys
import re
np.set_printoptions(suppress=True)

def task1():
    # termination criteria:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # 30 = loops
    # size of checkerboard: a*b
    a = 9
    b = 6
    objp = np.zeros((a*b,3), np.float32)

    objp[:,:2] = np.mgrid[0:a,0:b].T.reshape(-1,2)

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # all image file names with .png suffix
    path = "Assignment1\Checkerboards\*.png"
    image_fnames = glob.iglob(path)

    # iterate all training images
    for i_fname in image_fnames:
        print(i_fname)
        # read source image
        img = cv2.imread(i_fname)
    
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
        # find corners 
        ret,corners = cv2.findChessboardCorners(gray,(a,b),None) # ret: flag of corners, boolean type
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
            # if i_fname.endswith('.png'):  
            #     new_fname = i_fname.replace('Checkerboards', "Result")
            #     print(new_fname)
            #     new_fname = new_fname.replace('.png','_Corners.png')
            #     print(new_fname)
            # cv2.imwrite(new_fname, img)

    # Calibration
    # to estimate the intrinsic and extrinsic parameters of the camera.
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # return camera matrix, distortion coefficients, rotation and translation vectors etc.
    print("camera matrix:", mtx)
    print("distortion", dist)

    # Save camera parameters
    np.savez("Assignment1\CameraData\camera_Data_5.npz", mtx=mtx, dist=dist,rvecs=rvecs,tvecs=tvecs )

if __name__== "__main__" :
    task1()
