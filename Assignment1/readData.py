import numpy as np
np.set_printoptions(suppress=True)

def readData(path):
    with np.load(path) as Y:
        intrinsic, extrinsic = [Y[i] for i in ('Intrinsic', 'Extrinsic')]
    print(path)
    print("Intrinsic Matrix: \n", intrinsic)
    print("Extrinsic Matrix: \n", extrinsic)
    print("\n")
    return 0

path1 = 'Assignment1\CameraData\cam1\Extrinsic_camera_Data_cam_1.npz'
path2 = 'Assignment1\CameraData\cam2\Extrinsic_camera_Data_cam_2.npz'
path3 = 'Assignment1\CameraData\cam3\Extrinsic_camera_Data_cam_3.npz'
path4 = 'Assignment1\CameraData\cam4\Extrinsic_camera_Data_cam_4.npz'

readData(path1)
readData(path2)
readData(path3)
readData(path4)