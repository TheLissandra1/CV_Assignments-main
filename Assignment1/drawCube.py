import cv2
import numpy as np
import glob
import os, sys
import re


# def draw_line(img, corners, imgpts):
#     corner = tuple(corners[0].ravel())
#     img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
#     img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
#     img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
#     return img

# draw 3D coordinates axies
def draw_line(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    ## Attention! data type must be transformed into int32, or OpenCV will report bugs
    img = cv2.line(img, np.int32(corner), np.int32(tuple(imgpts[0].ravel())), (255, 0, 0), 5)
    img = cv2.line(img, np.int32(corner), np.int32(tuple(imgpts[1].ravel())), (0, 255, 0), 5)
    img = cv2.line(img, np.int32(corner), np.int32(tuple(imgpts[2].ravel())), (0, 0, 255), 5)
    return img

# draw 3D cube
def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # print(imgpts)
    # Bottom face: green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # Columns: blue
    for i,j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # Roof: red
    img = cv2.drawContours(img, [imgpts[4:]],-1, (0, 0, 255), 3)
    return img


    
if __name__ == '__main__':

    # Load previously saved data
    with np.load('Assignment1\CameraData\camera_Data.npz') as X:
        mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    a = 9
    b = 6
    objp = np.zeros((a*b,3), np.float32)

    objp[:,:2] = np.mgrid[0:a,0:b].T.reshape(-1,2)


    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

    axis_cube = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                            [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])



    path = "Assignment1\Checkerboards\*.png"
    for fname in glob.iglob(path):
        img = cv2.imread(fname)
        img_cube = cv2.imread(fname)
    
        print(fname)
    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (a, b), None)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
            # Find rotation and translation vectors
            ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)

            # Project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            img_line = draw_line(img, corners2, imgpts)
    
            # Cube point
            imgpts_cube, jac_cube = cv2.projectPoints(axis_cube, rvecs, tvecs, mtx, dist)  
            img_cube = draw_cube(img_cube, corners, imgpts_cube)
            cv2.namedWindow('image with line', 0)
            cv2.resizeWindow('image with line', 1000, 1000)
            cv2.namedWindow('image with cube', 0)
            cv2.resizeWindow('image with cube', 1000, 1000)
            cv2.imshow('image with line', img_line)  
            cv2.imshow('image with cube', img_cube)
            k = cv2.waitKey(0) & 0xFF

            if fname.endswith('.png'):  
                line_fname = fname.replace('Checkerboards', "Lines")
                line_fname = line_fname.replace('.png','_Line.png')

            if fname.endswith('.png'):  
                new_fname = fname.replace('Checkerboards', "Cubes")
                print(new_fname)
                new_fname = new_fname.replace('.png','_Cube.png')
                print(new_fname)
                
            cv2.imwrite(line_fname, img_line)
            cv2.imwrite(new_fname, img_cube)


    cv2.destroyAllWindows()
