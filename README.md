# CV_Assignments

## Assignment 2
#### GUI.py: 
The user interface in Assignment 1.

Add the "get extrinsic" button.

After the user can load multiple images to get the camera intrinsic matrix, the user can click "get extrinsic" to load the checkerboard image.

The user can use the manually click to choose the original point and get the extrinsic matrix.
#### drawCube.py:
File created in Assignment 1. Used to draw world axis on the board.

#### CameraData directory:
The camera data of 4 camera are stored in 4 config.xml files.

`data\cam1\config.xml`

`data\cam2\config.xml`

`data\cam3\config.xml`

`data\cam4\config.xml`

#### averageFrame.py:
Average frame of background.avi

#### computeImgDiff.py:
Get difference images.

#### hsv.py:
Get images in H, S, and V channels.

#### thresholding.py:
Do background subtraction.

#### assignment.py:
The task 3: voxel reconstruction.

We also changed the size of voxel in the cube.json. 
The original half edge length of voxel is 0.5. We changed it to 0.12 (around one quarter).



