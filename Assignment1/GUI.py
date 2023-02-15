import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QIcon
import numpy as np

class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.image = None

        self.initUI()
    
    def initUI(self):
        #set window
        self.setGeometry(0, 0, 1920, 1080)
        self.setWindowIcon(QIcon("Assignment1\Sprites\Creeper.png"))


        # Create buttons and labels
        self.load_button = QtWidgets.QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)

        # QPushButton to draw corners
        self.draw_button = QtWidgets.QPushButton('Draw Corners')
        # self.load_button.setGeometry(QtCore.QRect(400, 1600, 100, 23))
        self.draw_button.clicked.connect(self.draw_corners)
        self.draw_button.setEnabled(False)


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
        button_group_box_layout.addWidget(self.load_button)
        button_group_box_layout.addWidget(self.draw_button)
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
            height, width, channel = self.image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(self.image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap)
            self.draw_button.setEnabled(True)

    # draw_corners click function, connected to button
    def draw_corners(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        a = 9
        b = 6
        objp = np.zeros((a*b,3), np.float32)
        objp[:,:2] = np.mgrid[0:a,0:b].T.reshape(-1,2)

        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) #转灰度
        
        # 寻找角点,存入corners,ret是找到角点的flag
        ret,corners = cv2.findChessboardCorners(gray,(a,b),None)

        print(ret)
        if ret == True:
            objpoints.append(objp)
            print(objpoints)
            # 执行亚像素级角点检测
            ########## Begin ##########
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            
            ########## End ##########
            
            imgpoints.append(corners2)
            print(imgpoints)
            # Detect the calibration pattern in image:
            img_corners = cv2.drawChessboardCorners(self.image, (a,b),corners2,ret)
            height, width, channel = self.image.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(self.image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            self.result_label.setPixmap(pixmap)            
        else:
            self.result_label.setText("Corners not found!")

    
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
