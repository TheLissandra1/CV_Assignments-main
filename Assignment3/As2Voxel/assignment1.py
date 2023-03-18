import glm
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats import wasserstein_distance
from skimage import measure

block_size = 1.0
# data\camX\config.xml file paths
config_cam1 = '..\CameraData\extrinsic_cam1.xml'
config_cam2 = '..\CameraData\extrinsic_cam2.xml'
config_cam3 = '..\CameraData\extrinsic_cam3.xml'
config_cam4 = '..\CameraData\extrinsic_cam4.xml'


def generate_grid(width, depth):
    # Generates the floor grid locations
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x * block_size - width / 2, -block_size, z * block_size - depth / 2])
            colors.append([1.0, 1.0, 1.0])  # if (x + z) % 2 == 0 else [0, 0, 0])
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
        # else:
        #     colors.append([0, 0, 0])

    return img, points, indx, colors


def project_person(data, rvec, tvec, intrinsic, dist):
    data = [i for i in data if -1300 < i[2] < -1100]
    projectedPoints, jac = cv2.projectPoints(np.asarray(data), rvec, tvec, intrinsic, dist)
    return projectedPoints


def get_color(color_img, projected_points):
    # convert from BGR to rgb
    rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

    colors = []
    for i, p in enumerate(projected_points):
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
    h_peak = np.argmax(hist)

    # s and v also use the peak values
    s_hist, _ = np.histogram(s.ravel(), 255, [0, 255])
    s_peak = np.argmax(s_hist)

    v_hist, _ = np.histogram(v.ravel(), 255, [0, 255])
    v_peak = np.argmax(v_hist)

    dominant_color = [h_peak, s_peak, v_peak]

    return dominant_color


# compute manhattan distance between two 3d vectors
def ManhattanDistance(color1, color2):
    ManhattanDistance = sum(abs(val1 - val2) for val1, val2 in zip(color1, color2))
    return ManhattanDistance


def EMD(color1, color2):
    EMDist = wasserstein_distance(color1, color2)
    return EMDist


def euclidean(point1, point2):
    dist = np.linalg.norm(np.asarray(point1) - np.asarray(point2))
    return dist


def get_person_color(color_img, persons, color_list, pixels, best_cam):
    images = []
    for img in color_img:
        color = cv2.imread(img, cv2.IMREAD_COLOR)
        color = cv2.resize(color, [1000, 1000])
        images.append(color)

    dominant_hsvs = []
    for i in range(4):
        person_hsv = []
        for j in range(2):
            hsv = get_color(images[j], pixels[j][i])
            person_hsv.append(hsv)
        mean_hsv = np.mean(person_hsv, axis=0)
        dominant_hsvs.append(mean_hsv)

    dominant_hsvs = np.asarray(dominant_hsvs)
    max_h = np.max(dominant_hsvs, axis=0)[2]
    min_h = np.min(dominant_hsvs, axis=0)[2]
    temp_h = min_h
    temp_i = -1
    new_hsv = []
    for i, d_h in enumerate(dominant_hsvs):
        if d_h[2] == max_h:
            new_hsv.append([30, 225, 220])
            continue
        if d_h[2] == min_h:
            new_hsv.append([0, 225, 220])
            continue

        if temp_i == -1:
            temp_h = d_h[0]
            temp_i = i
            new_hsv.append([d_h[0], 225, 220])
        else:
            if d_h[0] > temp_h:
                new_hsv.append([120, 225, 220])
                new_hsv[temp_i][0] = 70
            else:
                new_hsv.append([70, 225, 220])
                new_hsv[temp_i][0] = 120

    print(dominant_hsvs)

    c = []
    for hsv in new_hsv:
        c.append(cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB))
    c = np.asarray(c).reshape(4, 3)

    for i, person in enumerate(persons):
        for k in person.keys():
            color_list[k] = c[i]

    return c, color_list


def init_voxel():
    data_zy = []
    width = 46
    height = 20
    depth = 56
    scale = 100
    for x in range(width * 2):
        for y in range(height * 2):
            for z in range(depth * 2):
                data_zy.append(
                    [scale * (0.5 * x * block_size - width / 2 + 7), scale * (0.5 * z * block_size - depth / 2 - 5),
                     -scale * (0.5 * y * block_size)])

    return data_zy


def build_voxel_model(foregrounds, cam_views, data, rvecs, tvecs, intrinsics, dists):
    data_ii = data
    color = None
    img_fg = []
    for i in range(4):
        fg, _, indx, color = get_index(data_ii, foregrounds[i], cam_views[i],
                                       rvecs[i], tvecs[i], intrinsics[i], dists[i])
        ii = indx
        data_ii = [data_ii[ind] for ind in ii]
        img_fg.append(fg)
    return data_ii, color, img_fg


def set_voxel(fg_path, cam_views, init=None):
    rvec_cam1 = cv2.Rodrigues(src=np.asarray(rmtx_cam1))[0]
    rvec_cam2 = cv2.Rodrigues(src=np.asarray(rmtx_cam2))[0]
    rvec_cam3 = cv2.Rodrigues(src=np.asarray(rmtx_cam3))[0]
    rvec_cam4 = cv2.Rodrigues(src=np.asarray(rmtx_cam4))[0]
    rvecs = [rvec_cam1, rvec_cam2, rvec_cam3, rvec_cam4]
    tvecs = [tvec_cam1, tvec_cam2, tvec_cam3, tvec_cam4]
    intrinsics = [intrinsic_cam1, intrinsic_cam2, intrinsic_cam3, intrinsic_cam4]
    dists = [dist_cam1, dist_cam2, dist_cam3, dist_cam4]

    if init is None:
        data_zy = init_voxel()
    else:
        data_zy = init

    result_zy, colors, img_fg = build_voxel_model(fg_path, cam_views, data_zy, rvecs, tvecs, intrinsics, dists)

    result = [[data[0], -data[2], data[1]] for data in result_zy]
    color_p = colors

    for i, dd in enumerate(result):
        temp = [d / 100 for d in dd]
        result[i] = temp

    c = [[d[0], d[2]] for d in result]
    c = np.asarray(c)
    c = np.float32(c)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    outlier_thresh = 4
    center = None
    for d in range(3):
        # K-Means cluster
        ret, label, center = cv2.kmeans(c, 4, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

        p0, p1, p2, p3 = {}, {}, {}, {}
        outliers = []
        counts = [0, 0, 0, 0]
        if d != 1:
            for i in range(label.shape[0]):
                if label[i] == 0:
                    p0[i] = result_zy[i]
                    counts[0] += 1

                elif label[i] == 1:
                    p1[i] = result_zy[i]
                    counts[1] += 1

                elif label[i] == 2:
                    p2[i] = result_zy[i]
                    counts[2] += 1

                elif label[i] == 3:
                    p3[i] = result_zy[i]
                    counts[3] += 1

            people = [p0, p1, p2, p3]

        elif d == 1:
            for i in range(label.shape[0]):
                for j in range(4):
                    if label[i] == j:
                        if euclidean(c[i], center[j]) > outlier_thresh:
                            outliers.append(i)

            if len(outliers) > 0:
                outliers.reverse()
                for o in outliers:
                    result.pop(o)
                    result_zy.pop(o)
                    color_p.pop(o)
                    c = [[d[0], d[2]] for d in result]
                    c = np.asarray(c)
                    c = np.float32(c)

        if d == 0:
            wrong = []
            # check wrong cluster because of ghost voxels
            for i, count in enumerate(counts):
                if count < 800:
                    wrong.append(i)
                else:
                    person = list(people[i].values())
                    p = [i for i in person if -1300 < i[2] < -1200]
                    if len(p) == 0:
                        wrong.append(i)
                        continue
                    p = [i for i in person if i[2] < -1400]
                    if len(p) == 0:
                        wrong.append(i)
                        continue

            print("ww", wrong)
            if len(wrong) > 0:
                for w in wrong:
                    keys = list(people[w].keys())
                    keys.reverse()
                    for key in keys:
                        result.pop(key)
                        result_zy.pop(key)
                        color_p.pop(key)
                        c = [[d[0], d[2]] for d in result]
                        c = np.asarray(c)
                        c = np.float32(c)

    # find the best three cam views
    counts_gap = []
    all_cam_pixel = []  # projected points of each person in each view
    for i in range(4):
        pixel_counts = 0
        cam_pixel = []
        for person in people:
            # projected points of one person in one view
            pixel = project_person(list(person.values()), rvecs[i], tvecs[i], intrinsics[i], dists[i])
            pixel_counts += pixel.shape[0]
            cam_pixel.append(pixel)
        total_pixels = project_person(result_zy, rvecs[i], tvecs[i], intrinsics[i], dists[i])
        gap = pixel_counts - total_pixels.shape[0]
        counts_gap.append(gap)
        all_cam_pixel.append(cam_pixel)

    best_index = np.argmax(counts_gap)
    best_cam = cam_views[best_index]
    worst_index = np.argmin(counts_gap)  # discard the worst cam view
    all_cam_pixel.pop(worst_index)
    cam_views.pop(worst_index)
    test_pixels = all_cam_pixel
    test_views = cam_views
    # test_index1 = np.argmax(counts_gap)
    # counts_gap[test_index1] = -1
    # test_index2 = np.argmax(counts_gap)
    # print("has cam", test_index1, test_index2)

    # test_pixels = [all_cam_pixel[test_index1], all_cam_pixel[test_index2]]
    # test_views = [cam_views[test_index1], cam_views[test_index2]]

    track_color, color_p = get_person_color(test_views, people, color_p, test_pixels, best_cam)

    for i, cc in enumerate(color_p):
        temp = [c / 255 for c in cc]
        color_p[i] = temp

    track_color = np.float32(track_color)
    for i, cc in enumerate(track_color):
        temp = [round(c / 255., 2) for c in cc]
        track_color[i] = temp

    return result, color_p, center, track_color


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

    rotation_mtx_cam1 = glm.rotate(rotation_mtx_cam1, np.pi / 2, [0, 0, 1])
    rotation_mtx_cam2 = glm.rotate(rotation_mtx_cam2, np.pi / 2, [0, 0, 1])
    rotation_mtx_cam3 = glm.rotate(rotation_mtx_cam3, np.pi / 2, [0, 0, 1])
    rotation_mtx_cam4 = glm.rotate(rotation_mtx_cam4, np.pi / 2, [0, 0, 1])

    cam_rotations = [rotation_mtx_cam1, rotation_mtx_cam2, rotation_mtx_cam3, rotation_mtx_cam4]

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
