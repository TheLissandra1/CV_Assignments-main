import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
import numpy as np


class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.image = None
        self.coordinates = []
        self.img_copy = None

        self.initUI()

    def initUI(self):
        # set window
        self.setGeometry(0, 0, 1920, 1080)
        self.setWindowIcon(QIcon("Sprites\Creeper.png"))

        self.width = QLineEdit(self)
        self.width.setMaximumWidth(50)
        self.width.setText("9")
        self.height = QLineEdit(self)
        self.height.setMaximumWidth(50)
        self.height.setText("6")
        text_layout = QFormLayout()

        text_edit = QWidget()
        text_edit.setMaximumWidth(300)
        text_layout.addRow("Width: ", self.width)
        text_layout.addRow("Height: ", self.height)
        text_edit.setLayout(text_layout)

        # Create buttons and labels
        self.load_button = QtWidgets.QPushButton("Load Image")
        self.load_button.setToolTip("Load an image")
        self.load_button.clicked.connect(self.load_image)

        # QPushButton to draw corners
        self.draw_button = QtWidgets.QPushButton('Draw Corners')
        # self.load_button.setGeometry(QtCore.QRect(400, 1600, 100, 23))
        self.draw_button.clicked.connect(self.draw_corners)
        self.draw_button.setEnabled(False)

        # QPushButton to draw corners
        self.all_button = QtWidgets.QPushButton('Run all')
        self.all_button.clicked.connect(self.draw_corners)
        self.all_button.setEnabled(False)

        # QLabel to present the loaded image
        self.image_label = QtWidgets.QLabel()
        # self.label.setAlignment(QtCore.Qt.AlignCenter)
        # self.image_label.setGeometry(QtCore.QRect(20, 20, 400, 400))

        # corner result
        self.result_label = QtWidgets.QLabel()
        # self.result_label.setAlignment(QtCore.Qt.AlignCenter)
        # self.image_label.setGeometry(QtCore.QRect(520, 520, 400, 400))

        # Create a group box to group buttons together
        button_group_box = QtWidgets.QGroupBox("Actions")

        # Create a layout for the group box
        button_group_box_layout = QtWidgets.QHBoxLayout()
        # Add buttons to the layout
        button_group_box_layout.addWidget(text_edit)
        button_group_box_layout.addWidget(self.load_button, alignment=QtCore.Qt.AlignTop)
        button_group_box_layout.addWidget(self.draw_button, alignment=QtCore.Qt.AlignTop)

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

    def run_All(self):
        return 0

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

    # draw_corners click function, connected to button
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
                print("Failed")
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
                    flag_draw = False

            if flag_draw:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners2)
                # Detect the calibration pattern in image:
                img_corners = cv2.drawChessboardCorners(self.image, (a, b), corners2, ret)

                self.show_image("result")
        else:
            self.result_label.setText("Please enter integer")


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
