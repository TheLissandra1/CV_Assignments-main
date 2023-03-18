import glm
import glfw
import numpy as np
from matplotlib import pyplot as plt

from engine.base.program import get_linked_program
from engine.renderable.model import Model
from engine.buffer.texture import *
from engine.buffer.hdrbuffer import HDRBuffer
from engine.buffer.blurbuffer import BlurBuffer
from engine.effect.bloom import Bloom
from assignment import set_voxel_positions, generate_grid, get_cam_positions, get_cam_rotation_matrices
from assignment1 import set_voxel, init_voxel
from engine.camera import Camera
from engine.config import config

data_zy, centers, center_colors = [], [], []
initial = 8
auto_draw = False
cube, track, hdrbuffer, blurbuffer, lastPosX, lastPosY = None, None, None, None, None, None
firstTime = True
window_width, window_height = config['window_width'], config['window_height']
camera = Camera(glm.vec3(0, 100, 0), pitch=-90, yaw=0, speed=40)


def draw_objs(obj, program, perspective, light_pos, texture, normal, specular, depth):
    program.use()
    program.setMat4('viewProject', perspective * camera.get_view_matrix())
    program.setVec3('viewPos', camera.position)
    program.setVec3('light_pos', light_pos)

    glActiveTexture(GL_TEXTURE1)
    program.setInt('mat.diffuseMap', 1)
    texture.bind()

    glActiveTexture(GL_TEXTURE2)
    program.setInt('mat.normalMap', 2)
    normal.bind()

    glActiveTexture(GL_TEXTURE3)
    program.setInt('mat.specularMap', 3)
    specular.bind()

    glActiveTexture(GL_TEXTURE4)
    program.setInt('mat.depthMap', 4)
    depth.bind()
    program.setFloat('mat.shininess', 128)
    program.setFloat('mat.heightScale', 0.12)

    obj.draw_multiple(program)


def main():
    global hdrbuffer, blurbuffer, cube, track, window_width, window_height, initial, auto_draw, centers, center_colors

    if not glfw.init():
        print('Failed to initialize GLFW.')
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
    glfw.window_hint(glfw.SAMPLES, config['sampling_level'])

    if config['fullscreen']:
        mode = glfw.get_video_mode(glfw.get_primary_monitor())
        window_width, window_height = mode.size.window_width, mode.size.window_height
        window = glfw.create_window(mode.size.window_width,
                                    mode.size.window_height,
                                    config['app_name'],
                                    glfw.get_primary_monitor(),
                                    None)
    else:
        window = glfw.create_window(window_width, window_height, config['app_name'], None, None)
    if not window:
        print('Failed to create GLFW Window.')
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    glfw.set_framebuffer_size_callback(window, resize_callback)
    glfw.set_cursor_pos_callback(window, mouse_move)
    glfw.set_key_callback(window, key_callback)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)

    program = get_linked_program('resources/shaders/vert.vs', 'resources/shaders/frag.fs')
    depth_program = get_linked_program('resources/shaders/shadow_depth.vs', 'resources/shaders/shadow_depth.fs')
    blur_program = get_linked_program('resources/shaders/blur.vs', 'resources/shaders/blur.fs')
    hdr_program = get_linked_program('resources/shaders/hdr.vs', 'resources/shaders/hdr.fs')

    blur_program.use()
    blur_program.setInt('image', 0)

    hdr_program.use()
    hdr_program.setInt('sceneMap', 0)
    hdr_program.setInt('bloomMap', 1)

    window_width_px, window_height_px = glfw.get_framebuffer_size(window)

    hdrbuffer = HDRBuffer()
    hdrbuffer.create(window_width_px, window_height_px)
    blurbuffer = BlurBuffer()
    blurbuffer.create(window_width_px, window_height_px)

    bloom = Bloom(hdrbuffer, hdr_program, blurbuffer, blur_program)

    light_pos = glm.vec3(0.5, 0.5, 0.5)
    perspective = glm.perspective(45, window_width / window_height, config['near_plane'], config['far_plane'])

    cam_rot_matrices = get_cam_rotation_matrices()
    cam_shapes = [Model('resources/models/camera.json', cam_rot_matrices[c]) for c in range(4)]
    square = Model('resources/models/square.json')
    cube = Model('resources/models/cube.json')
    track = Model('resources/models/track.json')
    texture = load_texture_2d('resources/textures/diffuse.jpg')
    texture_grid = load_texture_2d('resources/textures/diffuse_grid.jpg')
    normal = load_texture_2d('resources/textures/normal.jpg')
    normal_grid = load_texture_2d('resources/textures/normal_grid.jpg')
    specular = load_texture_2d('resources/textures/specular.jpg')
    specular_grid = load_texture_2d('resources/textures/specular_grid.jpg')
    depth = load_texture_2d('resources/textures/depth.jpg')
    depth_grid = load_texture_2d('resources/textures/depth_grid.jpg')

    grid_positions, grid_colors = generate_grid(config['world_width'], config['world_width'])
    square.set_multiple_positions(grid_positions, grid_colors)

    cam_positions, cam_colors = get_cam_positions()
    for c, cam_pos in enumerate(cam_positions):
        cam_shapes[c].set_multiple_positions([cam_pos], [cam_colors[c]])

    last_time = glfw.get_time()
    while not glfw.window_should_close(window):
        if config['debug_mode']:
            print(glGetError())

        current_time = glfw.get_time()
        delta_time = current_time - last_time
        last_time = current_time

        move_input(window, delta_time)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.1, 0.2, 0.8, 1)

        if auto_draw:
            fg_root = "../foreground/"
            view_root = "../videoframe/"

            print(initial, "start")

            fg_path, cam_views = [], []
            for j in range(4):
                fg_path.append(
                    fg_root + "cam" + str(j + 1) + "/" + str(initial * 10) + ".png")  # ../foreground/cam1/0.png
                cam_views.append(view_root + "cam" + str(j + 1) + "/" + str(initial * 10) + ".png")
            positions, colors, center, center_color = set_voxel(fg_path, cam_views, init=data_zy)
            for c in center:
                c = [c[0], 0, c[1]]
                centers.append(c)

            center_colors.extend(center_color)
            track.set_multiple_positions(centers, center_colors)
            cube.set_multiple_positions(positions, colors)
            print(initial, "done")
            initial = initial + 1

        square.draw_multiple(depth_program)
        cube.draw_multiple(depth_program)
        track.draw_multiple(depth_program)
        for cam in cam_shapes:
            cam.draw_multiple(depth_program)

        hdrbuffer.bind()

        window_width_px, window_height_px = glfw.get_framebuffer_size(window)
        glViewport(0, 0, window_width_px, window_height_px)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        draw_objs(square, program, perspective, light_pos, texture_grid, normal_grid, specular_grid, depth_grid)
        draw_objs(cube, program, perspective, light_pos, texture, normal, specular, depth)
        draw_objs(track, program, perspective, light_pos, texture, normal, specular, depth)
        for cam in cam_shapes:
            draw_objs(cam, program, perspective, light_pos, texture_grid, normal_grid, specular_grid, depth_grid)

        hdrbuffer.unbind()
        hdrbuffer.finalize()

        bloom.draw_processed_scene()

        glfw.poll_events()
        glfw.swap_buffers(window)

    if auto_draw:
        base_color = [[0.1, 0.1, 0.86], [0.86, 0.1, 0.1], [0.86, 0.86, 0.1], [0.1, 0.86, 0.36]]
        base_color = np.asarray(base_color).astype(np.float32)
        x, y, c = [], [], []
        for i, track_point in enumerate(centers):
            x.append(track_point[0])
            y.append(track_point[2])
            if (center_colors[i] == base_color[0]).all():
                c.append('b')
            elif (center_colors[i] == base_color[1]).all():
                c.append('r')
            elif (center_colors[i] == base_color[2]).all():
                c.append('y')
            elif (center_colors[i] == base_color[3]).all():
                c.append('g')
        plt.scatter(y, x, c=c, s=80)
        plt.show()
    glfw.terminate()


def resize_callback(window, w, h):
    if h > 0:
        global window_width, window_height, hdrbuffer, blurbuffer
        window_width, window_height = w, h
        glm.perspective(45, window_width / window_height, config['near_plane'], config['far_plane'])
        window_width_px, window_height_px = glfw.get_framebuffer_size(window)
        hdrbuffer.delete()
        hdrbuffer.create(window_width_px, window_height_px)
        blurbuffer.delete()
        blurbuffer.create(window_width_px, window_height_px)


def key_callback(window, key, scancode, action, mods):
    global cube, track, data_zy, initial, auto_draw, centers, center_colors
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, glfw.TRUE)
    if key == glfw.KEY_G and action == glfw.PRESS:
        positions, colors = set_voxel_positions(config['world_width'], config['world_height'], config['world_width'])
        cube.set_multiple_positions(positions, colors)
    if key == glfw.KEY_I and action == glfw.PRESS:
        data_zy = init_voxel()
        initial = 8
    if key == glfw.KEY_V and action == glfw.PRESS:
        fg_root = "../foreground/"
        view_root = "../videoframe/"

        print(initial, "start")

        fg_path, cam_views = [], []
        for j in range(4):
            fg_path.append(fg_root + "cam" + str(j + 1) + "/" + str(initial * 10) + ".png")  # ../foreground/cam1/0.png
            cam_views.append(view_root + "cam" + str(j + 1) + "/" + str(initial * 10) + ".png")
        positions, colors, center, center_color = set_voxel(fg_path, cam_views, init=data_zy)
        for c in center:
            c = [c[0], 0, c[1]]
            centers.append(c)

        center_colors.extend(center_color)
        print(center_colors)
        track.set_multiple_positions(centers, center_colors)
        cube.set_multiple_positions(positions, colors)
        print(initial, "done")
        initial = initial + 1

    if key == glfw.KEY_P and action == glfw.PRESS:
        auto_draw = True
    if key == glfw.KEY_O and action == glfw.PRESS:
        auto_draw = False


def mouse_move(win, pos_x, pos_y):
    global firstTime, camera, lastPosX, lastPosY
    if firstTime:
        lastPosX = pos_x
        lastPosY = pos_y
        firstTime = False

    camera.rotate(pos_x - lastPosX, lastPosY - pos_y)
    lastPosX = pos_x
    lastPosY = pos_y


def move_input(win, time):
    if glfw.get_key(win, glfw.KEY_W) == glfw.PRESS:
        camera.move_top(time)
    if glfw.get_key(win, glfw.KEY_S) == glfw.PRESS:
        camera.move_bottom(time)
    if glfw.get_key(win, glfw.KEY_A) == glfw.PRESS:
        camera.move_left(time)
    if glfw.get_key(win, glfw.KEY_D) == glfw.PRESS:
        camera.move_right(time)


if __name__ == '__main__':
    main()
