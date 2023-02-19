import cv2
import numpy as np
import drawCube

url = 0
cap = cv2.VideoCapture(url)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# size of checkerboard: a*b
a = 9
b = 6
objp = np.zeros((a * b, 3), np.float32)
objp[:, :2] = np.mgrid[0:a, 0:b].T.reshape(-1, 2)
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

axis_cube = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                        [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

while cap.isOpened():
    ret, frame = cap.read()
    #  flip
    frame = cv2.flip(frame, 1)
    if not ret:
        break
    cv2.imshow('frame', frame)

    img = frame
    # draw corners in each frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (a, b), None,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH
                                             + cv2.CALIB_CB_NORMALIZE_IMAGE
                                             + cv2.CALIB_CB_FAST_CHECK)
    if ret:
        objpoints.append(objp)

        # execute sub-pixel corner detection
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        imgpoints.append(corners2)

        # camera calibrations
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # draw cubes in each frame
        # Find rotation and translation vectors
        ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)

        # Cube point
        imgpts_cube, jac_cube = cv2.projectPoints(axis_cube, rvecs, tvecs, mtx, dist)
        img_cube = drawCube.draw_cube(img, corners, imgpts_cube)

        cv2.imshow("frame", img_cube)

    # set press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
