import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
import numpy as np
import os
np.set_printoptions(suppress=True)


# User interface to load image, draw corners.
class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.image = None
        self.coordinates = []
        self.img_copy = None
        self.corner_check = False
        self.initUI()

    def initUI(self):
        # set window
        self.setGeometry(0, 0, 1920, 1080)

        # Enter the size of the chessboard corners
        self.width = QLineEdit(self)
        self.width.setMaximumWidth(50)
        self.width.setText("8")
        self.height = QLineEdit(self)
        self.height.setMaximumWidth(50)
        self.height.setText("6")

        # use goodFeaturesToTrack detection result to adjust manually selected corners position
        self.check = QtWidgets.QCheckBox("Use cv2.goodFeaturesToTrack\nwhen manually selecting corners")
        self.check.setChecked(False)
        self.check.stateChanged.connect(self.click_box)

        text_layout = QFormLayout()
        text_edit = QWidget()
        text_edit.setMaximumWidth(400)
        text_layout.addRow("Width: ", self.width)
        text_layout.addRow("Height: ", self.height)
        text_layout.addRow(self.check)
        text_edit.setLayout(text_layout)

        # Create buttons and labels
        self.load_button = QtWidgets.QPushButton("Load Image")
        self.load_button.setToolTip("Load an image")
        self.load_button.clicked.connect(self.load_image)

        # QPushButton to draw corners
        self.draw_button = QtWidgets.QPushButton('Draw Corners')
        self.draw_button.clicked.connect(self.draw_corners)
        self.draw_button.setEnabled(False)

        # QPushButton to load multiple images and store camera matrix
        self.all_button = QtWidgets.QPushButton('Run multiple')
        self.all_button.clicked.connect(self.run_All)
        self.all_button.setEnabled(True)

        # QLabel to present the loaded image
        self.image_label = QtWidgets.QLabel()

        # corner result
        self.result_label = QtWidgets.QLabel()

        # Create a group box to group buttons together
        button_group_box = QtWidgets.QGroupBox("Actions")

        # Create a layout for the group box
        button_group_box_layout = QtWidgets.QHBoxLayout()
        # Add buttons to the layout
        button_group_box_layout.addWidget(text_edit)
        button_group_box_layout.addWidget(self.load_button, alignment=QtCore.Qt.AlignTop)
        button_group_box_layout.addWidget(self.draw_button, alignment=QtCore.Qt.AlignTop)
        button_group_box_layout.addWidget(self.all_button, alignment=QtCore.Qt.AlignTop)

        # Set the layout for the group box
        button_group_box.setLayout(button_group_box_layout)

        # Create a layout for image show
        label_group_box = QtWidgets.QGroupBox("Images")

        # Create a layout for the label_group_box
        label_group_box_layout = QtWidgets.QHBoxLayout()
        # Add labels to the layout
        label_group_box_layout.addWidget(self.image_label)
        label_group_box_layout.addWidget(self.result_label)
        label_group_box.setLayout(label_group_box_layout)

        # Create a layout for the main window
        main_layout = QtWidgets.QVBoxLayout()
        # Add widgets to the main layout
        main_layout.addWidget(button_group_box)
        main_layout.addWidget(label_group_box)

        # Set the main layout for the main window
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def click_box(self, state):
        if state == QtCore.Qt.Checked:
            self.corner_check = True
        else:
            self.corner_check = False

    # Manually draw 4 corners of bad images
    def manually_draw(self, a, b, gray):
        self.result_label.setText("Corners not found!\n"
                                  "Please manually select 4 corners of the largest rectangle "
                                  "in the chessboard\n"
                                  "\n"
                                  "Please select corners in z-order (order: 1-2-3-4): \n"
                                  "\n"
                                  "            Width\n"
                                  "    1 ⌈¯¯¯¯¯¯¯¯¯¯¯¯¯¯⌉ 2\n"
                                  "      |           ⋰  |\n"
                                  "      |        ⋰     |  Height\n"
                                  "      |     ⋰        |\n"
                                  "      |  ⋰           |\n"
                                  "    3 ⌊______________⌋ 4\n"
                                  "\n"
                                  "After selection, press enter to confirm")
        cv2.namedWindow('manually click', 0)
        cv2.resizeWindow('manually click', 1000, 1000)
        cv2.moveWindow('manually click', 10, 10)
        cv2.imshow("manually click", self.image)
        self.img_copy = self.image.copy()
        cv2.setMouseCallback("manually click", self.click_event)
        while True:
            key = cv2.waitKey(1)
            if cv2.getWindowProperty('manually click', cv2.WND_PROP_VISIBLE) < 1:  # window closed
                break
            if key == 13:  # press enter
                break
        cv2.destroyAllWindows()

        if len(self.coordinates) == 4:
            flag_draw = True

            if self.corner_check:
                # define the size of the region around the selected point
                region = 50
                coord = []
                for point in self.coordinates:
                    # create a binary mask with the selected region set to 1
                    mask = np.zeros_like(self.image[:, :, 0])
                    mask[point[1] - region: point[1] + region, point[0] - region:point[0] + region] = 1

                    # detect the corner in the region around the selected point
                    accurate_cor = cv2.goodFeaturesToTrack(image=gray, maxCorners=5, qualityLevel=0.01,
                                                           minDistance=5, mask=mask)
                    # take the one closest to the manually selected corner
                    c = accurate_cor[0]
                    dist = np.linalg.norm(c - point)
                    for a_c in accurate_cor:
                        temp_dist = np.linalg.norm(a_c - point)
                        if temp_dist < dist:
                            c = a_c
                            dist = temp_dist

                    coord.append(c)

                coord = np.asarray(coord).reshape(4, 1, -1).astype(np.float32)
            else:
                coord = np.asarray(self.coordinates).reshape(4, 1, -1).astype(np.float32)

            # Generate a grid of points on the plane
            square_size = 50  # in pixels
            grid_points = []
            for i in range(b):
                for j in range(a):
                    x = j * square_size
                    y = i * square_size
                    grid_points.append([x, y])
            grid_points = np.array(grid_points, dtype=np.float32).reshape(a * b, 1, -1)

            # Find the homography matrix
            plane_corners = np.array([[0, 0], [(a - 1) * square_size, 0], [0, (b - 1) * square_size],
                                      [(a - 1) * square_size, (b - 1) * square_size]],
                                     dtype=np.float32).reshape(4, 1, -1)
            M = cv2.findHomography(plane_corners, coord)[0]
            corners = cv2.perspectiveTransform(grid_points, M)
            self.coordinates.clear()
        else:
            corners = 0
            flag_draw = False

        return corners, flag_draw

    # run multiple selected images at the same time
    # if ret returns false value, apply manual selection of 4 outer corners and generate inner corners automatically
    def run_All(self):
        if len(self.width.text()) == 0 or len(self.height.text()) == 0:
            self.result_label.setText("Please enter the size of the chessboard")

        elif self.width.text().isdigit() and self.height.text().isdigit():
            options = QtWidgets.QFileDialog.Options()
            file_names, _ = QFileDialog.getOpenFileNames(self, 'Open Image', '',
                                                         'Images (*.png *.xpm *.jpg *.bmp *.gif *.pbm *.pgm *.ppm *.xbm *.xpm);;All Files (*)',
                                                         options=options)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # size of checkerboard: a*b
            a = int(self.width.text())
            b = int(self.height.text())
            objp = np.zeros((a * b, 3), np.float32)
            objp[:, :2] = np.mgrid[0:a, 0:b].T.reshape(-1, 2)

            objpoints = []  # 3d point in real world space
            imgpoints = []  # 2d points in image plane.
            flag_draw = False

            # iterate all training images
            for i_fname in file_names:
                print(i_fname)
                # read source image
                self.image = cv2.imread(i_fname)
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                self.image = cv2.resize(self.image, [1000, 1000], None, None)
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

                # find corners and ret: flag of corners, boolean type
                ret, corners = cv2.findChessboardCorners(gray, (a, b), None)
                print(ret)

                if ret:
                    flag_draw = True

                else:
                    self.show_image("source")
                    corners, flag_draw = self.manually_draw(a, b, gray)

                if flag_draw:
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners2)
                    objpoints.append(objp)
                    # Draw and display the corners
                    img = cv2.drawChessboardCorners(self.image, (a, b), corners2, ret)
                    self.show_image("source")
                    # save image with corners
                    #if i_fname.endswith('.png'):
                    #    new_fname = i_fname.replace('Checkerboards', "Result")
                    #    print(new_fname)
                    #    new_fname = new_fname.replace('.png', '_Corners.png')
                    #    print(new_fname)
                    #cv2.imwrite(new_fname, self.image)

            # Calibration
            # to estimate the intrinsic and extrinsic parameters of the camera.
            # return camera matrix, distortion coefficients, rotation and translation vectors etc.
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

            print("camera matrix:", mtx)
            print("distortion", dist)
            self.result_label.setText("Camera matrix:\n" + str(mtx))

            # Save camera parameters
            camera_data_fname = "Assignment1\CameraData\cam1\camera_Data_cam_1.npz"
            np.savez(camera_data_fname, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)


            # to obtain the rotation and translation
            ret, rvec, tvec = cv2.solvePnP(objectPoints=objp, imagePoints=corners2, cameraMatrix=mtx, distCoeffs=dist, 
             useExtrinsicGuess=False, flags=cv2.SOLVEPNP_ITERATIVE)
            print("Extrinsic parameters:")
            print("rvec_solvePnP: " , rvec)
            print(type(rvec))
            print("tvec_solvePnP: " , tvec)

            
            # get rotation matrix R
            rotationMtx, _ = cv2.Rodrigues(src=np.asarray(rvec))
            print("Rotation matrix_Rodrigues: ", rotationMtx)

            tvec = np.asarray(tvec)
            extrinsic_Matrix = np.concatenate((rotationMtx, tvec), axis=1)
            print(type(extrinsic_Matrix))
            print("Extrinsic Matrix: ", extrinsic_Matrix)

            # save extrinsic matrix into .txt file
            cam_path, cam_name = os.path.split(camera_data_fname)
            txt_name, _ = os.path.splitext(cam_name)
            save_path = cam_path + "\Extrinsic_" + txt_name
            
            np.savez(save_path, Intrinsic=mtx, Extrinsic=extrinsic_Matrix)
        

            




        else:
            self.result_label.setText("Please enter integer in the size of the chessboard")

    def show_image(self, position="result"):
        h, w, channel = self.image.shape
        bytes_per_line = 3 * w
        q_image = QtGui.QImage(self.image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_image)
        if position == "result":
            self.result_label.setPixmap(pixmap)
        elif position == "source":
            self.image_label.setPixmap(pixmap)
        self.draw_button.setEnabled(True)

    # draw coordinates when mouse clicks
    def click_event(self, event, x, y, flags, param):
        # check left mouse clicks
        if event == cv2.EVENT_FLAG_LBUTTON:
            self.coordinates.append([x, y])

            # display the coordinates on the image window
            font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
            cv2.putText(self.img_copy, str(x) + ',' + str(y), (x, y), font, 1, (144, 0, 255), 2)
            cv2.imshow('manually click', self.img_copy)

        if event == cv2.EVENT_FLAG_RBUTTON:
            # clear the list
            # clear the text of the coordinates
            self.coordinates = []
            self.img_copy = self.image.copy()
            cv2.imshow('manually click', self.img_copy)

    # click function to load an image from device
    def load_image(self):
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image', '',
                                                   'Images (*.png *.xpm *.jpg *.bmp *.gif *.pbm *.pgm *.ppm *.xbm *.xpm);;All Files (*)',
                                                   options=options)
        if file_name:
            self.image = cv2.imread(file_name)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            # image is too large, resize it
            self.image = cv2.resize(self.image, [1000, 1000], None, None)
            self.show_image("source")

    # click function to draw corners of loaded image
    def draw_corners(self):
        if len(self.width.text()) == 0 or len(self.height.text()) == 0:
            self.result_label.setText("Please enter the size of the chessboard")

        elif self.width.text().isdigit() and self.height.text().isdigit():
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            a = int(self.width.text())
            b = int(self.height.text())
            objp = np.zeros((a * b, 3), np.float32)
            objp[:, :2] = np.mgrid[0:a, 0:b].T.reshape(-1, 2)

            objpoints = []  # 3d point in real world space
            imgpoints = []  # 2d points in image plane.
            flag_draw = False

            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # gray-scale
            ret, corners = cv2.findChessboardCorners(gray, (a, b), None)

            if ret:
                flag_draw = True
            else:
                corners, flag_draw = self.manually_draw(a, b, gray)

            if flag_draw:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners2)
                # Detect the calibration pattern in image:
                img_corners = cv2.drawChessboardCorners(self.image, (a, b), corners2, True)

                self.show_image("result")
        else:
            self.result_label.setText("Please enter integer in the size of the chessboard")


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
