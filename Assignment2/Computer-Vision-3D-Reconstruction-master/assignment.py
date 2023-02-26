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
    

    data = []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if random.randint(0, 1000) < 5:
                    data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
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
    print(C_cam1.T[0])
    print(C_cam2.T[0])
    print(C_cam3.T[0])
    print(C_cam4.T[0])


    return [np.asarray(C_cam1.T)[0]*0.01, 
            np.asarray(C_cam2.T)[0]*0.01, 
            np.asarray(C_cam3.T)[0]*0.01, 
            np.asarray(C_cam4.T)[0]*0.01]

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
    
    # Completed

    # camera rotation matrices in (4x4) form
    rotation_mtx_cam1 = expandTo4x4(rmtx_cam1)
    rotation_mtx_cam2 = expandTo4x4(rmtx_cam2)
    rotation_mtx_cam3 = expandTo4x4(rmtx_cam3)
    rotation_mtx_cam4 = expandTo4x4(rmtx_cam4)


    cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    # cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]

    '''
    Nisha
    '''
    cam_rotations = [rotation_mtx_cam1, rotation_mtx_cam2, rotation_mtx_cam3, rotation_mtx_cam4]
    cam_tvecs = [tvec_cam1, tvec_cam2, tvec_cam3, tvec_cam4]
    

    # glm.rotate(angle: Number, axis: vector3), -> dmat4x4
    for c in range(len(cam_rotations)):
        # translate the world/view position
        # See PyGLM wiki documentation
        # For matrix: transform numpy.ndarray into <class 'glm.mat4x4'> first
        cam_rotations[c] = glm.mat4(cam_rotations[c])
        # For vector: transform numpy.ndarray into <class 'glm.vec3'>
        cam_tvecs[c] = glm.vec3(cam_tvecs[c][0], cam_tvecs[c][1], cam_tvecs[c][2])
        cam_rotations[c] = glm.translate(cam_rotations[c], cam_tvecs[c])
        # cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        # cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        # cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations

# load cam data from config.xml files: 
# intrinsic matrix, distortionCoefficients, rotation matrix R, translation vector T
intrinsic_mtx, distCoeffs, rmtx_cam1, tvec_cam1 = readCamConfig(config_cam1)
intrinsic_mtx, distCoeffs, rmtx_cam2, tvec_cam2 = readCamConfig(config_cam2)
intrinsic_mtx, distCoeffs, rmtx_cam3, tvec_cam3 = readCamConfig(config_cam3)
intrinsic_mtx, distCoeffs, rmtx_cam4, tvec_cam4 = readCamConfig(config_cam4)