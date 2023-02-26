import glm
import random
import numpy as np

block_size = 1.0
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
    return [[-64 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, -64 * block_size],
            [-64 * block_size, 64 * block_size, -64 * block_size]]


def readCamData(path):
    with np.load(path) as Y:
        intrinsic, extrinsic = [Y[i] for i in ('Intrinsic', 'Extrinsic')]
    return intrinsic, extrinsic

def expandTo4x4(mtx):

    # get first 3 columns, [, ) of input extrinsic matrix R|T
    rotation_mtx = mtx[:, 0:3] 
    # expand to 4*4 matrix by adding 1 in right-bottom corner.
    rotation_mtx = np.concatenate((rotation_mtx, [[0,0,0]]), axis=0)
    rotation_mtx = np.concatenate((rotation_mtx, [[0],[0],[0],[1]]), axis=1)
    
    return rotation_mtx


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    # Completed
    path_cam1 = 'Assignment2\step1\Extrinsic_camera_Data_cam_1.npz'
    path_cam2 = 'Assignment2\step1\Extrinsic_camera_Data_cam_2.npz'
    path_cam3 = 'Assignment2\step1\Extrinsic_camera_Data_cam_3.npz'
    path_cam4 = 'Assignment2\step1\Extrinsic_camera_Data_cam_4.npz'
    _, extrinsic_cam1 = readCamData(path_cam1)
    _, extrinsic_cam2 = readCamData(path_cam2)
    _, extrinsic_cam3 = readCamData(path_cam3)
    _, extrinsic_cam4 = readCamData(path_cam4)
    rotation_mtx_cam1 = expandTo4x4(extrinsic_cam1)
    rotation_mtx_cam2 = expandTo4x4(extrinsic_cam2)
    rotation_mtx_cam3 = expandTo4x4(extrinsic_cam3)
    rotation_mtx_cam4 = expandTo4x4(extrinsic_cam4)

    cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    # cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    cam_rotations = [rotation_mtx_cam1, extrinsic_cam2, extrinsic_cam3, extrinsic_cam4]

    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations
