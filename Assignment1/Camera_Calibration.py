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


def task1():
    print("hello world")
    fname = "Assignment1\cb2.png"
    # criteria:角点精准化迭代过程的终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((7*7,3), np.float32)
    '''
    设定世界坐标下点的坐标值,因为用的是棋盘可以直接按网格取；
    假定棋盘正好在x-y平面上, 这样z值直接取0,简化初始化步骤。
    mgrid把列向量[0:cbraw]复制了cbcol列,把行向量[0:cbcol]复制了cbraw行。
    转置reshape后,每行都是4*6网格中的某个点的坐标。
    '''
    objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # 识别出角点,记录世界物体坐标和图像坐标
    img = cv2.imread(fname) #source image
    
    cv2.namedWindow('image', 0)
    cv2.resizeWindow('image', 800, 600)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.imwrite('cb2_test1.png', img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #转灰度
    # 寻找角点,存入corners,ret是找到角点的flag
    ########## Begin ##########
    ret,corners = cv2.findChessboardCorners(gray,(7,7),None)

    ########## End ##########
    print(type(ret))
    print(ret)
    if ret == True:
        objpoints.append(objp)
        # 执行亚像素级角点检测
        ########## Begin ##########
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        
        ########## End ##########
        
        imgpoints.append(corners2)
        # 在棋盘上绘制角点
        img = cv2.drawChessboardCorners(img,(7,7),corners2,ret)
        cv2.imshow("ChessboardCorners", img)
        cv2.waitKey(0)
        # 保存图像
        cv2.imwrite("cb2_test2.png", img)
        # filepath = '/data/workspace/myshixun/task1/'
        # cv2.imwrite(filepath + 'out/img.png', img)

    # Calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Undistortion
    img = cv2.imread('Assignment1\cb2.png')
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    print(type(dst))
    print(dst)
    cv2.imshow("undistorted result", dst)
    cv2.waitKey(0)
    cv2.imwrite('cb2_result.jpg', dst)
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
