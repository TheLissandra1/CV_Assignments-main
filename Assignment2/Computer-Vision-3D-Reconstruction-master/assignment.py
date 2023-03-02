import glm
import random
import numpy as np
import cv2

block_size = 1.0
# data\camX\config.xml file paths
config_cam1 = '..\data\cam1\config.xml'
config_cam2 = '..\data\cam2\config.xml'
config_cam3 = '..\data\cam3\config.xml'
config_cam4 = '..\data\cam4\config.xml'


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data = []
    for x in range(width):
        for z in range(depth):
            data.append([x * block_size - width / 2, -block_size, z * block_size - depth / 2])
    return data


def get_index(data, img_path, rvec, tvec, intrinsic, dist):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, [1000, 1000], None, None)

    projectedPoints, jac = cv2.projectPoints(np.asarray(data), rvec, tvec, intrinsic, dist)
    points = []
    indx = []
    for i, p in enumerate(projectedPoints):
        if 0 <= p[0][0] < 1000 and 0 <= p[0][1] < 1000:
            if img[np.int32(p[0][1]), np.int32(p[0][0])] == 255:
                points.append(p)
                indx.append(i)

    return points, indx


def set_voxel_positions(width, height, depth):
    rvec_cam1 = cv2.Rodrigues(src=np.asarray(rmtx_cam1))[0]
    rvec_cam2 = cv2.Rodrigues(src=np.asarray(rmtx_cam2))[0]
    rvec_cam3 = cv2.Rodrigues(src=np.asarray(rmtx_cam3))[0]
    rvec_cam4 = cv2.Rodrigues(src=np.asarray(rmtx_cam4))[0]

    data_zy = []
    data = []
    width = 10
    height = 14
    depth = 12
    scale = 100
    # for x in range(width * 5):
    for x in range(width * 5):
        for y in range(height * 5):
            for z in range(depth * 5):
                data.append(
                    [scale * (0.2 * x * block_size), scale * (0.2 * y * block_size),
                     scale * (0.2 * z * block_size - depth / 2)])

                data_zy.append(
                    [scale * (0.2 * x * block_size), scale * (0.2 * z * block_size - depth / 2),
                     -scale * (0.2 * y * block_size)])

    points1, indx1 = get_index(data_zy, "..\step2\cam1\diff\Diff_threshold.png", rvec_cam1, tvec_cam1,
                               intrinsic_cam1, dist_cam1)
    points2, indx2 = get_index(data_zy, "..\step2\cam2\diff\Diff_threshold.png", rvec_cam2, tvec_cam2,
                               intrinsic_cam2, dist_cam2)
    points3, indx3 = get_index(data_zy, "..\step2\cam3\diff\Diff_threshold.png", rvec_cam3, tvec_cam3,
                               intrinsic_cam3, dist_cam3)
    points4, indx4 = get_index(data_zy, "..\step2\cam4\diff\Diff_threshold.png", rvec_cam4, tvec_cam4,
                               intrinsic_cam4, dist_cam4)

    indx = np.intersect1d(indx1, indx2)
    indx = np.intersect1d(indx, indx3)
    indx = np.intersect1d(indx, indx4)
    data_p = [data[ind] for ind in indx]
    # print(data_p)
    for k, dd in enumerate(data_p):
        temp = [d / scale for d in dd]
        data_p[k] = temp

    return data_p


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
    cams_3d = cams_3d / 100

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
    dist = cv_file.getNode('DistortionCoeffs').mat()
    rmtx = cv_file.getNode('RotationMatrix').mat()
    tvec = cv_file.getNode('TranslationMatrix').mat()

    cv_file.release()
    # matrix type: <class 'numpy.ndarray'>
    return intrinsic_mtx, dist, rmtx, tvec


def expandTo4x4(mtx):
    # # get first 3 columns, [, ) of input extrinsic matrix R|T
    # rotation_mtx = mtx[:, 0:3] 
    # expand to 4*4 matrix by adding 1 in right-bottom corner.
    rotation_mtx = np.concatenate((mtx, [[0, 0, 0]]), axis=0)
    rotation_mtx = np.concatenate((rotation_mtx, [[0], [0], [0], [1]]), axis=1)

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

    rotation_mtx_cam1 = glm.rotate(rotation_mtx_cam1, np.pi / 2, [0, 0, 1])
    rotation_mtx_cam2 = glm.rotate(rotation_mtx_cam2, np.pi / 2, [0, 0, 1])
    rotation_mtx_cam3 = glm.rotate(rotation_mtx_cam3, np.pi / 2, [0, 0, 1])
    rotation_mtx_cam4 = glm.rotate(rotation_mtx_cam4, np.pi / 2, [0, 0, 1])

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
intrinsic_cam1, dist_cam1, rmtx_cam1, tvec_cam1 = readCamConfig(config_cam1)
intrinsic_cam2, dist_cam2, rmtx_cam2, tvec_cam2 = readCamConfig(config_cam2)
intrinsic_cam3, dist_cam3, rmtx_cam3, tvec_cam3 = readCamConfig(config_cam3)
intrinsic_cam4, dist_cam4, rmtx_cam4, tvec_cam4 = readCamConfig(config_cam4)
