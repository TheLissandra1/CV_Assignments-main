import cv2
import numpy as np


# function to display the coordinates of
# the points clicked on the image
def click_event(event, x, y, flags, img):
    global coordinates, img_copy

    # check left mouse clicks
    if event == cv2.EVENT_FLAG_LBUTTON:
        # display the coordinates on the Shell
        print(x, ' ', y)
        coordinates.append([x, y])

        # display the coodinates on the image window
        font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        cv2.putText(img_copy, str(x) + ',' + str(y), (x, y), font, 1, (144, 0, 255), 2)
        cv2.imshow('manually click', img_copy)

    if event == cv2.EVENT_FLAG_RBUTTON:
        # clear the list
        # clear the text of the coordinates
        coordinates = []
        img_copy = img.copy()
        cv2.imshow('manually click', img_copy)


def manual_linear_interpolation(x0, y0, x1, y1, x):
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def task1():
    global coordinates, img_copy
    fname = "Checkerboards\C1.png"
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # size of checkerboard: a*b
    a = 9
    b = 6
    objp = np.zeros((a * b, 3), np.float32)
    '''
    设定世界坐标下点的坐标值,因为用的是棋盘可以直接按网格取；
    假定棋盘正好在x-y平面上, 这样z值直接取0,简化初始化步骤。
    mgrid把列向量[0:cbraw]复制了cbcol列,把行向量[0:cbcol]复制了cbraw行。
    转置reshape后,每行都是4*6网格中的某个点的坐标。
    '''
    objp[:, :2] = np.mgrid[0:a, 0:b].T.reshape(-1, 2)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # print(image_fnames)

    # read source image
    img = cv2.imread(fname)

    # view img
    cv2.namedWindow('image', 0)
    cv2.resizeWindow('image', 800, 800)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find corners and
    # ret: flag of corners, boolean type
    ret, corners = cv2.findChessboardCorners(gray, (a, b), None)
    print(ret)

    if ret:
        print(corners)
        objpoints.append(objp)

        # execute sub-pixel corner detection
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (a, b), corners2, ret)
        cv2.imshow("image", img)
        cv2.waitKey(0)
        # save image with corners
        cv2.imwrite("Checkerboards\C1_test.png", img)
        cv2.destroyAllWindows()

    else:
        cv2.namedWindow('manually click', 0)
        cv2.resizeWindow('manually click', 800, 800)
        cv2.imshow("manually click", img)
        img_copy = img.copy()
        cv2.setMouseCallback("manually click", click_event, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Failed")
        print(coordinates)
        print(len(coordinates))
        corners = []

        if len(coordinates) == 4:
            coordinates = np.asarray(coordinates).reshape(4, 1, -1).astype(np.float32)

            # Generate a grid of points on the plane
            square_size = 50  # in pixels
            grid_points = []
            for i in range(b):
                for j in range(a):
                    x = j * square_size
                    y = i * square_size
                    grid_points.append([x, y])
            grid_points = np.array(grid_points, dtype=np.float32).reshape(a * b, 1, -1)
            # print(grid_points)

            # Find the homography matrix
            plane_corners = np.array([[0, 0], [(a - 1) * square_size, 0], [0, (b - 1) * square_size],
                                     [(a - 1) * square_size, (b - 1) * square_size]],
                                     dtype=np.float32).reshape(4, 1, -1)
            M = cv2.findHomography(plane_corners, coordinates)[0]
            corners = cv2.perspectiveTransform(grid_points, M)

        objpoints.append(objp)

        # execute sub-pixel corner detection
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (a, b), corners2, True)
        cv2.imshow("image1", img)
        cv2.waitKey(0)
        # save image with corners
        #cv2.imwrite("Checkerboards\C1_test.png", img)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    coordinates = []
    img_copy = None
    task1()
'''
        if len(coordinates) == 4:
            start_a = coordinates[0]
            end_a = coordinates[1]

            start_b1 = coordinates[0]
            end_b1 = coordinates[2]

            start_b2 = coordinates[1]
            end_b2 = coordinates[3]

            d_b1 = (start_b1[0] - end_b1[0]) / (b-1)
            d_b2 = (start_b2[0] - end_b2[0]) / (b-1)
            for i in range(b):
                if i == b - 1:
                    start_a = coordinates[2]
                    end_a = coordinates[3]
                elif i > 0:
                    xs = start_b1[0] - d_b1 * i
                    ys = manual_linear_interpolation(start_b1[0], start_b1[1], end_b1[0], end_b1[1], xs)
                    start_a = [xs, ys]

                    xe = start_b2[0] - d_b2 * i
                    ye = manual_linear_interpolation(start_b2[0], start_b2[1], end_b2[0], end_b2[1], xe)
                    end_a = [xe, ye]

                print(start_a)
                print(end_a)
                d_a = (start_a[0] - end_a[0]) / (a-1)
                x = start_a[0]
                corners.append([start_a])
                for j in range(a - 2):
                    x = x - d_a
                    y = manual_linear_interpolation(start_a[0], start_a[1], end_a[0], end_a[1], x)
                    corners.append([[x, y]])
                corners.append([end_a])
        corners = np.asarray(corners).astype(np.float32)
        '''