# CV_Assignments

## Assignment 1
#### Camera_Calibration.py:
Basic code of camera calibration. Load all images in the directory and do cv2.drawChessboardCorners. Store the camera matrix.   
#### GUI.py: 
The user interface.
The user can load one original image of the chessboard and the detected corners will be shown on the result image.

If the corners cannot be detected automatically, the user can select 4 corners on the largest rectangle and generate the rest corners. 
The image with generated corners will be displayed on the user interface.

The user can also load multiple images, including images which need manually annotation, to get the camera matrix.
#### drawCube.py:
Load all images in the directory and load camera data file. Draw cube and axes on the images.

#### Webcam.py:
Open the webcam and draw cube on the chessboard plane in real-time.

#### CameraData directory:
The camera data for three run.
Run 1: camera_Data.npz
Run 2: camera_Data_10.npz
Run 3: camera_Data_5.npz

