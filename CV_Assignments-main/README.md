# CV_Assignments

## Assignment 1
#### Instructions:
Install Python (instructions for Windows) and OpenCV (instructions for Windows). Learn how to use OpenCV's geometric-camera calibration. For instance, read the OpenCV book (Chapter 7), read the OpenCV documentation or go directly to this tutorial. The offline phase is carried out only once for each of three runs. The online phase for each new image/frame from your webcam (or test image). You can also use C++ but realize your work can be used in Assignments 2 and 3, which are in Python.  
#### Offline phase:
Implement the geometric-camera calibration using OpenCV functions (only for this assignment, you are allowed to use code available on the internet). Print  this image . Measure the size of each cell and figure out where you have to use this in your code.  

While there is an OpenCV function that can find chessboard corners for you, this function sometimes fails. Therefore, you need to implement an interface that asks the user to manually provide the four corner points and that linearly interpolates all chessboard points from those coordinates. Tutorial for getting mouse click coordinates The output of your interface should be similar to that of the OpenCV function that finds the corner points automatically.  

With a camera or webcame, take 25 training images of the chessboard in different positions and orientations (rotated, tilted) in the image. Include 5 images with the chessboard in view but  where OpenCV could not detect the corners. This images require manual annotation of the corner points. Take a final test image that has the chessboard tilted significantly close to the image border. For this image, the corner points have to be found automatically.  

Calibrate your camera (geometrically) using the training images from step 3. Make sure the camera center is not fixed but is estimated and you can have different focal lengths in horizontal and vertical directions. You will do three runs of calibration. Run 1: use all training images (including the images with manually provided corner points). Run 2: use only ten images for which corner points were found automatically. Run 3: use only five out of the ten images in Run 2. In each run, you will calibrate the camera. After calibration, you will need the camera intrinsics (or cameraMatrix) (for this assignment) and the camera extrinsics (for Assignment 2).  

#### Online phase: 
For each run, take the test image and draw the world 3D axes (XYZ) with the origin at the center of the world coordinates, using the estimated camera parameters. Also draw a cube which is located at the origin of the world coordinates. You can get bonus points for doing this in real time using your webcam. See the example images below.  
Code: You can develop a script for the online and offline phase independently or combined. In your code, write comments at the beginning of every function (about the purposes of the function). This is compulsory.  
Report: Your report should be around 1 page and contain:  
1. For each of the three runs, the intrinsic camera matrix and an example screenshot with axes and cube. Provide the explicit form of camera intrinsics matrix K. This can also be done by an OpenCV-function, figure out which one.  
2. A brief explanation of how the estimation of each element in the intrinsics matrix changes for the three runs. What can you say about the quality of each run?  
3. A brief mention of the choice tasks that you've done, and how you implemented them. For some tasks, provide the requested information.  

#### Grading: The maximum score for this assignment is 100 (grade 10). The assignment counts for 10% of the course grade. You can get 80 regular points and max 20 points for chosen tasks:  
Offline calibration stage: 20  
Offline: manual corner annotation interface when no corners are found automatically: 15  
Online stage with cube drawing: 20  
Reporting: 10  
Screen-snapshots correct and accurate: 5  
CHOICE: real-time performance with webcam in online phase: 10  
CHOICE: iterative detection and rejection of low quality input images in offline phase: 10. Check for a function output that could provide an indication of the quality of the calibration.  
CHOICE: improving the localization of the corner points in your manual interface: 10. Make the estimation of (a) the four corner points or (b) the interpolated points more accurate.    
CHOICE: any animation/shading/etc. that demonstrates that you can project 3D to 2D: 10. There needs to be explicit depth reasoning so lines/vertices further away should not overlap nearer ones.  
CHOICE: implement a function that can provide a confidence for how well each variable has been estimated, perhaps by considering subsets of the input: 10  
CHOICE: implement a way to enhance the input to reduce the number of input images that are not correctly processed by findChessboardCorners, for example by enhancing edges or getting rid of light reflections: 10  
CHOICE: use your creativity. Check with Ronald Poppe for eligibility and number of points (via Teams or email).  

#### Submission: Submit a zip through Blackboard with  
Code (+ project files) but no libraries or images (max. submission size is 20MB)  
Report (1 page)  
Deadline: Sunday, February 19, 2023, at 23.00. For questions about the assignment, use the INFOMCV 2023 Teams (Assignment channel). If you found that your assignment partner did not work properly, notify Ronald Poppe as soon as possible.  
