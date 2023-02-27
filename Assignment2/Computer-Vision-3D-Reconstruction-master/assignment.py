import glm
import random
import numpy as np
import cv2

block_size = 1.0
# data\camX\config.xml file paths
config_cam1 = 'data\cam1\config.xml'
config_cam2 = 'data\cam2\config.xml'
config_cam3 = 'data\cam3\config.xml'
config_cam4 = 'data\cam4\config.xml'


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data = []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
    return data


def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    
    #  create the lookup-table for the valid voxel back-projections onto each of the camera’s FoVs
    # So, given a position in R3, we need a projection of it in R2.
    #  Luckily this can be done by the OpenCV function projectPoints. 
    # The projectPoints-method requires (aside from input and output points) the calibration matrices, 
    # which have to be saved in data/camX/config.xml:
    # imgpts, jac = cv2.projectPoints(objectPoints, rVec, tVec, intrisicMat, distCoeffs, projectedPoints)

    # 

    rvec_cam1 = cv2.Rodrigues(src=np.asarray(rmtx_cam1))

    # 3d points
    objectPoints = []
    objp = np.float32([[width, 0, 0], [0, height, 0], [0, 0, depth]]).reshape(-1, 3)
    objectPoints.append(objp)
    # 2d points
    projectedPoints = []
    projectedPoints, jac = cv2.projectPoints(objectPoints, rvec_cam1, tvec_cam1, intrinsic_mtx_cam1, 
                                    distCoeffs_cam1)


    data = projectedPoints
    data = []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if random.randint(0, 1000) < 5:
                    data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])


    # 先projection再循环
    print(data)
    return data


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    # camera positions given in R^3 , need to project to R^2

    # compute camera postion
    C_cam1 = -np.matrix(rmtx_cam1).T * np.matrix(tvec_cam1)
    C_cam2 = -np.matrix(rmtx_cam2).T * np.matrix(tvec_cam2)
    C_cam3 = -np.matrix(rmtx_cam3).T * np.matrix(tvec_cam3)
    C_cam4 = -np.matrix(rmtx_cam4).T * np.matrix(tvec_cam4)
    
    # concatenate 4 cam positions into one array
    cams_3d = np.array(C_cam1.T[0])
    cams_3d = np.concatenate((cams_3d, np.array(C_cam2.T[0])), axis=0)
    cams_3d = np.concatenate((cams_3d, np.array(C_cam3.T[0])), axis=0)
    cams_3d = np.concatenate((cams_3d, np.array(C_cam4.T[0])), axis=0)
    cams_3d = cams_3d/100

    # exchange y and z coordinates, minus changed y coordinates
    # y1 = -z0, z1 = y0
    for i in range(len(cams_3d)):
        c = cams_3d[i][1]
        cams_3d[i][1] = -cams_3d[i][2]
        cams_3d[i][2] = c
    print(cams_3d)

    # cams_3d = np.array([[2720.192 ,  1090.0698,  3051.115], 
    #                     [ 368.86923,  1157.011,  3410.093 ], 
    #                     [3074.9678, 1106.0835,    412.38507],
    #                     [-2246.4001, 1180.3229,  3075.0542]])/100
    
    return cams_3d

    # return [[-64 * block_size, 64 * block_size, 63 * block_size],
    #         [63 * block_size, 64 * block_size, 63 * block_size],
    #         [63 * block_size, 64 * block_size, -64 * block_size],
    #         [-64 * block_size, 64 * block_size, -64 * block_size]]

# read Config.xml matrix into variables
def readCamConfig(xml_path):
    cv_file = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)

    intrinsic_mtx = cv_file.getNode("Intrinsic").mat()
    distortCoeffs = cv_file.getNode('DistortionCoeffs').mat()
    rmtx = cv_file.getNode('RotationMatrix').mat()
    tvec = cv_file.getNode('TranslationMatrix').mat()

    cv_file.release()
    # matrix type: <class 'numpy.ndarray'>
    return intrinsic_mtx, distortCoeffs, rmtx, tvec

def expandTo4x4(mtx):

    # # get first 3 columns, [, ) of input extrinsic matrix R|T
    # rotation_mtx = mtx[:, 0:3] 
    # expand to 4*4 matrix by adding 1 in right-bottom corner.
    rotation_mtx = np.concatenate((mtx, [[0,0,0]]), axis=0)
    rotation_mtx = np.concatenate((rotation_mtx, [[0],[0],[0],[1]]), axis=1)
    
    return rotation_mtx


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    
    # need to transform rotation matrix
    # rotation = glm.rotate(rotation, np.pi / 2, [0, 0, 1])
    # camera rotation matrices in (4x4) form
    rotation_mtx_cam1 = glm.mat4x4(expandTo4x4(rmtx_cam1))
    rotation_mtx_cam2 = glm.mat4x4(expandTo4x4(rmtx_cam2))
    rotation_mtx_cam3 = glm.mat4x4(expandTo4x4(rmtx_cam3))
    rotation_mtx_cam4 = glm.mat4x4(expandTo4x4(rmtx_cam4))
    # print("rotation matrix")
    # print(rotation_mtx_cam1)
    # print(rotation_mtx_cam2)
    # print(rotation_mtx_cam3)
    # print(rotation_mtx_cam4)
    
    rotation_mtx_cam1 = glm.rotate(rotation_mtx_cam1, np.pi/2, [0,0,1])
    rotation_mtx_cam2 = glm.rotate(rotation_mtx_cam2, np.pi/2, [0,0,1])
    rotation_mtx_cam3 = glm.rotate(rotation_mtx_cam3, np.pi/2, [0,0,1])
    rotation_mtx_cam4 = glm.rotate(rotation_mtx_cam4, np.pi/2, [0,0,1])

    cam_rotations = [rotation_mtx_cam1, rotation_mtx_cam2, rotation_mtx_cam3, rotation_mtx_cam4]
    # cam_tvecs = [tvec_cam1, tvec_cam2, tvec_cam3, tvec_cam4]
    # print("translation vector: ", tvec_cam1)
    # print(tvec_cam2)
    # print(tvec_cam3)
    # print(tvec_cam4)
    # print(tvec_cam4)
    
    # # glm.rotate(angle: Number, axis: vector3), -> dmat4x4
    return cam_rotations

# load cam data from config.xml files: 
# intrinsic matrix, distortionCoefficients, rotation matrix R, translation vector T
intrinsic_mtx_cam1, distCoeffs_cam1, rmtx_cam1, tvec_cam1 = readCamConfig(config_cam1)
intrinsic_mtx, distCoeffs, rmtx_cam2, tvec_cam2 = readCamConfig(config_cam2)
intrinsic_mtx, distCoeffs, rmtx_cam3, tvec_cam3 = readCamConfig(config_cam3)
intrinsic_mtx, distCoeffs, rmtx_cam4, tvec_cam4 = readCamConfig(config_cam4)