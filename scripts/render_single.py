
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("trimesh").setLevel(logging.ERROR)
import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import lib.renderer.opengl_util as opengl_util
from lib.renderer.mesh import load_fit_body, load_scan, compute_tangent, load_ori_fit_body
import lib.renderer.prt_util as prt_util
from lib.renderer.gl.init_gl import initialize_GL_context
from lib.renderer.gl.prt_render import PRTRender
from lib.renderer.gl.color_render import ColorRender
from lib.renderer.camera import Camera
import argparse
import os
import glob
import cv2
import numpy as np
import random
import math
import time
import trimesh
from matplotlib import cm

t0 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--subject', type=str, help='subject name')
parser.add_argument('-o', '--out_dir', type=str, help='output dir')
parser.add_argument('-r', '--rotation', type=str, help='rotation num')
parser.add_argument('-w', '--size', type=str, help='render size')
args = parser.parse_args()

subject = args.subject
save_folder = args.out_dir
rotation = int(args.rotation)
size = int(args.size)

# headless
egl = True

# render
initialize_GL_context(width=size, height=size, egl=egl)

dataset = save_folder.split("/")[-1].split("_")[0]
# ============test============
dataset='thuman2'

#=============================

format = 'obj'
scale = 100.0
up_axis = 1
pcd = False
smpl_type = "smplx"
with_light = True
depth = False
normal = True

mesh_file = os.path.join(f'./data/{dataset}/scans/{subject}',
                         f'{subject}.{format}')
smplx_file = f'./data/{dataset}/smplx/{subject}.obj'
tex_file = f'./data/{dataset}/scans/{subject}/material0.jpeg'
fit_file = f'./data/{dataset}/fits/{subject}/smplx_param.pkl'

# mesh
mesh = trimesh.load(mesh_file,
                    skip_materials=True,
                    process=False,
                    maintain_order=True,
                    force='mesh')

if not pcd:
    vertices, faces, normals, faces_normals, textures, face_textures = load_scan(
        mesh_file, with_normal=True, with_texture=True)
else:
    # remove floating outliers of scans
    mesh_lst = mesh.split(only_watertight=False)
    comp_num = [mesh.vertices.shape[0] for mesh in mesh_lst]
    mesh = mesh_lst[comp_num.index(max(comp_num))]

    vertices = mesh.vertices
    faces = mesh.faces
    normals = mesh.vertex_normals

# center

scan_scale = 1.8 / (vertices.max(0)[up_axis] - vertices.min(0)[up_axis])
rescale_fitted_body, joints = load_fit_body(fit_file,
                                            scale,
                                            smpl_type='smplx',
                                            smpl_gender='male')

os.makedirs(os.path.dirname(smplx_file), exist_ok=True)
ori_smplx = load_ori_fit_body(fit_file, smpl_type='smplx', smpl_gender='male')
ori_smplx.export(smplx_file)

vertices *= scale
vmin = vertices.min(0)
vmax = vertices.max(0)
vmed = joints[0]
vmed[up_axis] = 0.5 * (vmax[up_axis] + vmin[up_axis])

rndr_depth = ColorRender(width=size, height=size, egl=egl)
rndr_depth.set_mesh(rescale_fitted_body.vertices, rescale_fitted_body.faces,
                    rescale_fitted_body.vertices,
                    rescale_fitted_body.vertex_normals)
rndr_depth.set_norm_mat(scan_scale, vmed)

# camera
cam = Camera(width=size, height=size)
cam.ortho_ratio = 0.4 * (512 / size)

if pcd:

    colors = mesh.visual.vertex_colors[:, :3] / 255.0
    rndr = ColorRender(width=size, height=size, egl=egl)
    rndr.set_mesh(vertices, faces, colors, normals)
    rndr.set_norm_mat(scan_scale, vmed)

else:

    prt, face_prt = prt_util.computePRT(mesh_file, scale, 10, 2)
    shs = np.load('./scripts/env_sh.npy')
    rndr = PRTRender(width=size, height=size, ms_rate=16, egl=egl)
    rndr_uv=PRTRender(width=size, height=size, uv_mode=True, egl=egl)

    # texture
    texture_image = cv2.imread(tex_file)
    texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)

    # fake multiseg
    vertices_label_mode = np.random.randint(
        low=1, high=10,
        size=(vertices.shape[0], 10))  # [scan_verts_n, percomp]
    colormap = cm.get_cmap("rainbow")
    precomp_id = 4
    verts_label = colormap(vertices_label_mode[:, precomp_id] / np.max(
        vertices_label_mode[:, precomp_id]))[:, :3]  # [scan_verts_num, 3]

    tan, bitan = compute_tangent(vertices, faces, normals, textures,
                                 face_textures)

    rndr.set_norm_mat(scan_scale, vmed)
    rndr_uv.set_norm_mat(scan_scale, vmed)

    rndr.set_mesh(vertices, faces, normals, faces_normals, textures,
                  face_textures, prt, face_prt, tan, bitan, verts_label)
    rndr.set_albedo(texture_image)

    rndr_uv.set_mesh(vertices, faces, normals, faces_normals, textures,
                  face_textures, prt, face_prt, tan, bitan, verts_label)
    rndr_uv.set_albedo(texture_image)


os.makedirs(os.path.join(save_folder, subject, 'uv_color'), exist_ok=True)
os.makedirs(os.path.join(save_folder, subject, 'uv_mask'), exist_ok=True)
os.makedirs(os.path.join(save_folder, subject, 'uv_normal'), exist_ok=True)

for y in range(0, 360, 360 // rotation):

    cam.near = -100
    cam.far = 100
    cam.sanity_check()

    R = opengl_util.make_rotate(0, math.radians(y), 0)
    R_B = opengl_util.make_rotate(0, math.radians((y + 180) % 360), 0)

    if up_axis == 2:
        R = np.matmul(R, opengl_util.make_rotate(math.radians(90), 0, 0))

    rndr.rot_matrix = R
    rndr.set_camera(cam)

    rndr_uv.rot_matrix = R
    rndr_uv.set_camera(cam)


    if smpl_type != "none":
        rndr_depth.rot_matrix = R
        rndr_depth.set_camera(cam)

    dic = {
        'ortho_ratio': cam.ortho_ratio,
        'scale': scan_scale,
        'center': vmed,
        'R': R
    }

    if with_light:

        # random light
        sh_id = random.randint(0, shs.shape[0] - 1)
        sh = shs[sh_id]
        sh_angle = 0.2 * np.pi * (random.random() - 0.5)
        sh = opengl_util.rotateSH(sh,
                                  opengl_util.make_rotate(0, sh_angle, 0).T)
        dic.update({"sh": sh})

        rndr.set_sh(sh)
        rndr.analytic = False
        rndr.use_inverse_depth = False

        # uv_color
        rndr_uv.set_sh(sh)
        rndr_uv.analytic = False
        rndr_uv.use_inverse_depth = False
        rndr_uv.display()

        uv_color = rndr_uv.get_color(0)
        uv_color = cv2.cvtColor(uv_color, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(os.path.join(save_folder, subject, 'uv_color',
                                 f'{y:03d}.png'), 255.0*uv_color)
        # uv_normal, uv_mask
        if y==0:
            print(save_folder)
            print(1)
            uv_pos = rndr_uv.get_color(1)
            uv_mask = uv_pos[:,:,3]
            cv2.imwrite(os.path.join(save_folder, subject, 'uv_mask',
                                    '00.png'), 255.0*uv_mask)
            
            uv_normal = rndr_uv.get_color(2)
            uv_normal = cv2.cvtColor(uv_normal, cv2.COLOR_RGBA2BGR)
            cv2.imwrite(os.path.join(save_folder, subject, 'uv_normal',
                                    '00.png'), 255.0*uv_normal)

    # ==================================================================

    # calib
    calib = opengl_util.load_calib(dic, render_size=size)

    export_calib_file = os.path.join(save_folder, subject, 'calib',
                                     f'{y:03d}.txt')
    os.makedirs(os.path.dirname(export_calib_file), exist_ok=True)
    np.savetxt(export_calib_file, calib)

    # ==================================================================

    # front render
    rndr.display()

    opengl_util.render_result(
        rndr, 0, os.path.join(save_folder, subject, 'render', f'{y:03d}.png'))
    if normal:
        opengl_util.render_result(
            rndr, 1,
            os.path.join(save_folder, subject, 'normal_F', f'{y:03d}.png'))

    if depth:
        opengl_util.render_result(
            rndr, 2,
            os.path.join(save_folder, subject, 'depth_F', f'{y:03d}.png'))

    if smpl_type != "none":
        rndr_depth.display()
        opengl_util.render_result(
            rndr_depth, 1,
            os.path.join(save_folder, subject, 'T_normal_F', f'{y:03d}.png'))
        if depth:
            opengl_util.render_result(
                rndr_depth, 2,
                os.path.join(save_folder, subject, 'T_depth_F',
                             f'{y:03d}.png'))

    # ==================================================================

    # back render
    cam.near = 100
    cam.far = -100
    cam.sanity_check()

    rndr.set_camera(cam)
    rndr.display()
    opengl_util.render_result(
        rndr, 0, os.path.join(save_folder, subject, 'render_B', f'{y:03d}.png'))
    if normal:
        opengl_util.render_result(
            rndr, 1,
            os.path.join(save_folder, subject, 'normal_B', f'{y:03d}.png'))
    if depth:
        opengl_util.render_result(
            rndr, 2,
            os.path.join(save_folder, subject, 'depth_B', f'{y:03d}.png'))

    if smpl_type != "none":
        rndr_depth.set_camera(cam)
        rndr_depth.display()
        opengl_util.render_result(
            rndr_depth, 1,
            os.path.join(save_folder, subject, 'T_normal_B', f'{y:03d}.png'))
        if depth:
            opengl_util.render_result(
                rndr_depth, 2,
                os.path.join(save_folder, subject, 'T_depth_B',
                             f'{y:03d}.png'))


    # left render
    R_L = opengl_util.make_rotate(0, math.radians((y + 90) % 360), 0)
    cam.near = -100
    cam.far = 100
    cam.sanity_check()

    rndr.rot_matrix = R_L

    rndr.set_camera(cam)
    rndr.display()
    opengl_util.render_result(
        rndr, 0, os.path.join(save_folder, subject, 'render_L', f'{y:03d}.png'))
    
    if smpl_type != "none":
        rndr_depth.set_camera(cam)
        rndr_depth.rot_matrix = R_L
        rndr_depth.display()
        opengl_util.render_result(
            rndr_depth, 1,
            os.path.join(save_folder, subject, 'T_normal_L', f'{y:03d}.png'))
        if depth:
            opengl_util.render_result(
                rndr_depth, 2,
                os.path.join(save_folder, subject, 'T_depth_L',
                             f'{y:03d}.png'))



    # right render
    cam.near=100
    cam.far=-100
    cam.sanity_check()

    rndr.set_camera(cam)
    rndr.display()
    opengl_util.render_result(
        rndr, 0, os.path.join(save_folder, subject, 'render_R', f'{y:03d}.png'))
    if smpl_type != "none":
        rndr_depth.set_camera(cam)
        rndr_depth.display()
        opengl_util.render_result(
            rndr_depth, 1,
            os.path.join(save_folder, subject, 'T_normal_R', f'{y:03d}.png'))
        if depth:
            opengl_util.render_result(
                rndr_depth, 2,
                os.path.join(save_folder, subject, 'T_depth_R',
                             f'{y:03d}.png'))


done_jobs = len(glob.glob(f"{save_folder}/*/render"))
all_jobs = len(os.listdir(f"./data/{dataset}/scans"))
print(
    f"Finish rendering {subject}| {done_jobs}/{all_jobs} | Time: {(time.time()-t0):.0f} secs"
)
