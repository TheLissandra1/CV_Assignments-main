import glm
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure

block_size = 1.0
# data\camX\config.xml file paths
config_cam1 = '..\step1\CameraData\extrinsic_cam1.xml'
config_cam2 = '..\step1\CameraData\extrinsic_cam2.xml'
config_cam3 = '..\step1\CameraData\extrinsic_cam3.xml'
config_cam4 = '..\step1\CameraData\extrinsic_cam4.xml'


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x * block_size - width / 2, -block_size, z * block_size - depth / 2])
            colors.append([1.0, 1.0, 1.0] if (x + z) % 2 == 0 else [0, 0, 0])
    return data, colors


def get_index(data, img_path, color_path, rvec, tvec, intrinsic, dist):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, [1000, 1000])

    color = cv2.imread(color_path, cv2.IMREAD_COLOR)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color = cv2.resize(color, [1000, 1000])

    projectedPoints, jac = cv2.projectPoints(np.asarray(data), rvec, tvec, intrinsic, dist)
    points = []
    indx = []
    colors = []
    for i, p in enumerate(projectedPoints):
        if 0 <= p[0][0] < 1000 and 0 <= p[0][1] < 1000:
            if img[np.int32(p[0][1]), np.int32(p[0][0])] == 255:
                points.append(p)
                indx.append(i)
            colors.append(color[np.int32(p[0][1]), np.int32(p[0][0])])
        else:
            colors.append([0, 0, 0])

    return points, indx, colors


def get_color(color_img, data, rvec, tvec, intrinsic, dist):
    # convert from BGR to rgb
    rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

    data = [i for i in data if -1300 < i[2] < -1100]
    colors = []
    projectedPoints, jac = cv2.projectPoints(np.asarray(data), rvec, tvec, intrinsic, dist)
    for i, p in enumerate(projectedPoints):
        if 0 <= p[0][0] < 1000 and 0 <= p[0][1] < 1000:
            colors.append(rgb[np.int32(p[0][1]), np.int32(p[0][0])])

    colors = np.asarray(colors).reshape([1, len(colors), 3])
    # convert from BGR to HSV
    hsv = cv2.cvtColor(colors, cv2.COLOR_RGB2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # histogram of h
    hist, bins = np.histogram(h.ravel(), 360, [0, 360])

    dominant_Hue = np.argmax(hist)
    # plt.plot(hist)
    # plt.show()
    print("Dominant Color: ", dominant_Hue)

    # s,v channel compute average
    s_avg = np.rint(np.average(s.ravel())).astype(int)
    s_new = np.full((s.shape), s_avg)
    v_avg = np.rint(np.average(v.ravel())).astype(int)
    v_new = np.full((s.shape), v_avg)
    hsv[:, :, 1] = s_new
    hsv[:, :, 2] = v_new

    dominant_color = np.uint8([[[dominant_Hue, s_avg, v_avg]]])
    # convert to rgb values
    dominant_color_rgb = cv2.cvtColor(dominant_color, cv2.COLOR_HSV2RGB)
    print(dominant_color_rgb)

    return dominant_color[0][0], dominant_color_rgb[0][0]


def get_person_color(color_img, persons, color_list, rvec, tvec, intrinsic, dist):
    color = cv2.imread(color_img, cv2.IMREAD_COLOR)
    color = cv2.resize(color, [1000, 1000])
    dominant_hsvs = []
    for person in persons:
        dominant_hsv, _ = get_color(color, list(person.values()), rvec, tvec, intrinsic, dist)
        dominant_hsvs.append(dominant_hsv)

    dominant_hsvs = np.asarray(dominant_hsvs)
    max_h = np.max(dominant_hsvs, axis=0)[0]
    min_h = np.min(dominant_hsvs, axis=0)[0]
    temp_h = min_h
    temp_i = -1
    new_hsv = []
    for i, d_h in enumerate(dominant_hsvs):
        if d_h[0] == max_h:
            new_hsv.append([300, d_h[1], 125])
            continue
        if d_h[0] == min_h:
            new_hsv.append([0, d_h[1], 125])
            continue

        if temp_i == -1:
            temp_h = d_h[0]
            temp_i = i
            new_hsv.append([d_h[0], d_h[1], 125])
        else:
            if d_h[0] > temp_h:
                new_hsv.append([200, d_h[1], 125])
                new_hsv[temp_i][0] = 100
            else:
                new_hsv.append([100, d_h[1], 125])
                new_hsv[temp_i][0] = 200

    print(new_hsv)

    c = []
    for h in new_hsv:
        c.append(cv2.cvtColor(np.float32([[h]]), cv2.COLOR_HSV2RGB))

    for i, person in enumerate(persons):
        for k in person.keys():
            color_list[k] = c[i]


def set_voxel_positions(width, height, depth):
    rvec_cam1 = cv2.Rodrigues(src=np.asarray(rmtx_cam1))[0]
    rvec_cam2 = cv2.Rodrigues(src=np.asarray(rmtx_cam2))[0]
    rvec_cam3 = cv2.Rodrigues(src=np.asarray(rmtx_cam3))[0]
    rvec_cam4 = cv2.Rodrigues(src=np.asarray(rmtx_cam4))[0]

    data, data_zy = [], []
    width = 50
    height = 25
    depth = 50
    scale = 100
    for x in range(width * 2):
        for y in range(height * 2):
            for z in range(depth * 2):
                data.append(
                    [scale * (0.5 * x * block_size - width / 2), scale * (0.5 * y * block_size),
                     scale * (0.5 * z * block_size - depth / 2)])

                data_zy.append(
                    [scale * (0.5 * x * block_size - width / 2), scale * (0.5 * z * block_size - depth / 2),
                     -scale * (0.5 * y * block_size)])

    points1, indx1, color1 = get_index(data_zy, "..\step1\Diff\cam1\diff\Diff_threshold.png",
                                       "..\step1\Diff\cam1\cam1 - frame at 0m0s.png",
                                       rvec_cam1, tvec_cam1, intrinsic_cam1, dist_cam1)
    points2, indx2, color2 = get_index(data_zy, "..\step1\Diff\cam2\diff\Diff_threshold.png",
                                       "..\step1\Diff\cam2\cam2 - frame at 0m0s.png",
                                       rvec_cam2, tvec_cam2, intrinsic_cam2, dist_cam2)
    points3, indx3, color3 = get_index(data_zy, "..\step1\Diff\cam3\diff\Diff_threshold.png",
                                       "..\step1\Diff\cam3\cam3 - frame at 0m0s.png",
                                       rvec_cam3, tvec_cam3, intrinsic_cam3, dist_cam3)
    points4, indx4, color4 = get_index(data_zy, "..\step1\Diff\cam4\diff\Diff_threshold.png",
                                       "..\step1\Diff\cam4\cam4 - frame at 0m0s.png",
                                       rvec_cam4, tvec_cam4, intrinsic_cam4, dist_cam4)

    colors = np.mean([color1, color2, color3, color4], axis=0)

    indx = np.intersect1d(indx1, indx2)
    indx = np.intersect1d(indx, indx3)
    indx = np.intersect1d(indx, indx4)

    data_p = [data[ind] for ind in indx]
    data_zy_p = [data_zy[ind] for ind in indx]
    color_p = [colors[ind] for ind in indx]

    for i, dd in enumerate(data_p):
        temp = [d / scale for d in dd]
        data_p[i] = temp

    c = [[d[0], d[2]] for d in data_p]
    c = np.asarray(c)
    c = np.float32(c)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # K-Means cluster
    ret, label, center = cv2.kmeans(c, 4, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    p0, p1, p2, p3 = {}, {}, {}, {}
    for i in range(label.shape[0]):
        if label[i] == 0:
            p0[i] = data_zy_p[i]
        elif label[i] == 1:
            p1[i] = data_zy_p[i]
        elif label[i] == 2:
            p2[i] = data_zy_p[i]
        elif label[i] == 3:
            p3[i] = data_zy_p[i]

    get_person_color("..\step1\Diff\cam2\cam2 - frame at 0m0s.png", [p0, p1, p2, p3], color_p,
                     rvec_cam2, tvec_cam2, intrinsic_cam2, dist_cam2)

    for i, cc in enumerate(color_p):
        temp = [c / 255 for c in cc]
        color_p[i] = temp
    return data_p, color_p


def get_cam_positions():
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

    # cams_3d = np.array([[ 27.66822    16.419746   22.834217 ]
    #                       [  5.399558   16.793898   30.340225 ]
    #                       [-22.179577   16.294825   -2.1881757]
    #                       [-21.764788   17.307413   23.527065 ]]

    return cams_3d, [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]

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

'''
voxels = np.asarray(data_p)
    ux = np.unique(voxels[:, 0])
    uy = np.unique(voxels[:, 1])
    uz = np.unique(voxels[:, 2])

    # create a meshgrid
    X, Y, Z = np.meshgrid(uy, ux, uz)
    v = np.zeros(X.shape)
    n = voxels.shape[0]
    for i in range(n):
        ix = ux == voxels[i, 0]
        iy = uy == voxels[i, 1]
        iz = uz == voxels[i, 2]
        v[ix, iy, iz] = 1

    # Use marching cubes to obtain the surface mesh
    verts, faces, normals, values = measure.marching_cubes(v)

    # Display resulting triangular mesh using Matplotlib
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=22, azim=20, roll=97.5)  # right
    # ax.view_init(elev=30, azim=25, roll=102.5)  # right
    # ax.view_init(elev=28, azim=165, roll=-97)  # left
    # ax.view_init(elev=65, azim=120, roll=-148.5)  # front

    # `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")

    ax.set_xlim(-10, 40)
    ax.set_ylim(0, 50)
    ax.set_zlim(-15, 35)
    plt.tight_layout()
    plt.show()
    fig.savefig("mesh.pdf")
'''
