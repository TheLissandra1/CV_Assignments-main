import cv2
import numpy as np


# video file path
path = r'Assignment2\Computer-Vision-3D-Reconstruction-master\data\cam4\background.avi'


# Open the video file
cap = cv2.VideoCapture(path)
if not cap.isOpened():
    print('Unable to open: ' + path)
    exit(0)
# Initialize variables
frame_count = 0
avg_frame = None
number_frame = 50 # 100 frames in total in a video

# Loop over the first n frames of the video
while frame_count < number_frame:
    # Read a frame from the video
    ret, frame = cap.read()

    # If the frame was successfully read
    if ret:
        # Convert the frame to grayscale
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = frame

        # Add the grayscale frame to the running average
        if avg_frame is None:
            avg_frame = np.zeros_like(gray, dtype=np.float32)
        cv2.accumulate(gray, avg_frame)

        # Increment the frame count
        frame_count += 1

# Convert the average frame to uint8 format
avg_frame = cv2.convertScaleAbs(avg_frame / frame_count)

# Display the average frame
cv2.imshow('Average Frame', avg_frame)
file_path = r'Assignment2\Computer-Vision-3D-Reconstruction-master\data\cam4\cam4_background_Avg.png'
cv2.imwrite(file_path, avg_frame)
cv2.waitKey(0)

# Release the video file and close the window
cap.release()
cv2.destroyAllWindows()