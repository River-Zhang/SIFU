# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import numpy as np
import cv2
import pymeshlab
import torch
import torchvision
import trimesh
import json
from pytorch3d.io import load_obj
import os
from termcolor import colored
import os.path as osp
from scipy.spatial import cKDTree
import _pickle as cPickle
import open3d as o3d

from pytorch3d.structures import Meshes
import torch.nn.functional as F
from lib.pymaf.utils.imutils import uncrop
from lib.common.render_utils import Pytorch3dRasterizer, face_vertices

from pytorch3d.renderer.mesh import rasterize_meshes
from PIL import Image, ImageFont, ImageDraw
from kaolin.ops.mesh import check_sign
from kaolin.metrics.trianglemesh import point_to_mesh_distance

from pytorch3d.loss import (mesh_laplacian_smoothing, mesh_normal_consistency)

import tinyobjloader


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def obj_loader(path):
    # Create reader.
    reader = tinyobjloader.ObjReader()

    # Load .obj(and .mtl) using default configuration
    ret = reader.ParseFromFile(path)

    if ret == False:
        print("Failed to load : ", path)
        return None

    # note here for wavefront obj, #v might not equal to #vt, same as #vn.
    attrib = reader.GetAttrib()
    verts = np.array(attrib.vertices).reshape(-1, 3)

    shapes = reader.GetShapes()
    tri = shapes[0].mesh.numpy_indices().reshape(-1, 9)
    faces = tri[:, [0, 3, 6]]

    return verts, faces


class HoppeMesh:

    def __init__(self, verts, faces):
        '''
        The HoppeSDF calculates signed distance towards a predefined oriented point cloud
        http://hhoppe.com/recon.pdf
        For clean and high-resolution pcl data, this is the fastest and accurate approximation of sdf
        :param points: pts
        :param normals: normals
        '''
        self.trimesh = trimesh.Trimesh(verts, faces, process=True)
        self.verts = np.array(self.trimesh.vertices)
        self.faces = np.array(self.trimesh.faces)
        self.vert_normals, self.faces_normals = compute_normal(
            self.verts, self.faces)

    def contains(self, points):

        labels = check_sign(
            torch.as_tensor(self.verts).unsqueeze(0),
            torch.as_tensor(self.faces),
            torch.as_tensor(points).unsqueeze(0))
        return labels.squeeze(0).numpy()

    def triangles(self):
        return self.verts[self.faces]  # [n, 3, 3]


def tensor2variable(tensor, device):
    # [1,23,3,3]
    return torch.tensor(tensor, device=device, requires_grad=True)


class GMoF(torch.nn.Module):

    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        dist = torch.div(residual, residual + self.rho**2)
        return self.rho**2 * dist


def mesh_edge_loss(meshes, target_length: float = 0.0):
    """
    Computes mesh edge length regularization loss averaged across all meshes
    in a batch. Each mesh contributes equally to the final loss, regardless of
    the number of edges per mesh in the batch by weighting each mesh with the
    inverse number of edges. For example, if mesh 3 (out of N) has only E=4
    edges, then the loss for each edge in mesh 3 should be multiplied by 1/E to
    contribute to the final loss.

    Args:
        meshes: Meshes object with a batch of meshes.
        target_length: Resting value for the edge length.

    Returns:
        loss: Average loss across the batch. Returns 0 if meshes contains
        no meshes or all empty meshes.
    """
    if meshes.isempty():
        return torch.tensor([0.0],
                            dtype=torch.float32,
                            device=meshes.device,
                            requires_grad=True)

    N = len(meshes)
    edges_packed = meshes.edges_packed()  # (sum(E_n), 3)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    edge_to_mesh_idx = meshes.edges_packed_to_mesh_idx()  # (sum(E_n), )
    num_edges_per_mesh = meshes.num_edges_per_mesh()  # N

    # Determine the weight for each edge based on the number of edges in the
    # mesh it corresponds to.
    # TODO (nikhilar) Find a faster way of computing the weights for each edge
    # as this is currently a bottleneck for meshes with a large number of faces.
    weights = num_edges_per_mesh.gather(0, edge_to_mesh_idx)
    weights = 1.0 / weights.float()

    verts_edges = verts_packed[edges_packed]
    v0, v1 = verts_edges.unbind(1)
    loss = ((v0 - v1).norm(dim=1, p=2) - target_length)**2.0
    loss_vertex = loss * weights
    # loss_outlier = torch.topk(loss, 100)[0].mean()
    # loss_all = (loss_vertex.sum() + loss_outlier.mean()) / N
    loss_all = loss_vertex.sum() / N

    return loss_all


def remesh(obj_path, perc, device):

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_path)
    ms.laplacian_smooth()
    ms.remeshing_isotropic_explicit_remeshing(
        targetlen=pymeshlab.Percentage(perc), adaptive=True)
    ms.save_current_mesh(obj_path.replace("recon", "remesh"))
    polished_mesh = trimesh.load_mesh(obj_path.replace("recon", "remesh"))
    verts_pr = torch.tensor(
        polished_mesh.vertices).float().unsqueeze(0).to(device)
    faces_pr = torch.tensor(polished_mesh.faces).long().unsqueeze(0).to(device)

    return verts_pr, faces_pr


def possion(mesh, obj_path):

    mesh.export(obj_path)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_path)
    ms.surface_reconstruction_screened_poisson(depth=10)
    ms.set_current_mesh(1)
    ms.save_current_mesh(obj_path)

    return trimesh.load(obj_path)


def get_mask(tensor, dim):

    mask = torch.abs(tensor).sum(dim=dim, keepdims=True) > 0.0
    mask = mask.type_as(tensor)

    return mask


def blend_rgb_norm(rgb, norm, mask):

    # [0,0,0] or [127,127,127] should be marked as mask
    final = rgb * (1 - mask) + norm * (mask)

    return final.astype(np.uint8)


def unwrap(image, data):

    img_uncrop = uncrop(
        np.array(
            Image.fromarray(image).resize(
                data['uncrop_param']['box_shape'][:2])),
        data['uncrop_param']['center'], data['uncrop_param']['scale'],
        data['uncrop_param']['crop_shape'])

    img_orig = cv2.warpAffine(img_uncrop,
                              np.linalg.inv(data['uncrop_param']['M'])[:2, :],
                              data['uncrop_param']['ori_shape'][::-1][1:],
                              flags=cv2.INTER_CUBIC)

    return img_orig


# Losses to smooth / regularize the mesh shape
def update_mesh_shape_prior_losses(mesh, losses):

    # and (b) the edge length of the predicted mesh
    losses["edge"]['value'] = mesh_edge_loss(mesh)
    # mesh normal consistency
    losses["nc"]['value'] = mesh_normal_consistency(mesh)
    # mesh laplacian smoothing
    losses["laplacian"]['value'] = mesh_laplacian_smoothing(mesh,
                                                            method="uniform")


def rename(old_dict, old_name, new_name):
    new_dict = {}
    for key, value in zip(old_dict.keys(), old_dict.values()):
        new_key = key if key != old_name else new_name
        new_dict[new_key] = old_dict[key]
    return new_dict


def load_checkpoint(model, cfg):

    model_dict = model.state_dict()
    main_dict = {}
    normal_dict = {}

    device = torch.device(f"cuda:{cfg['test_gpus'][0]}")

    if os.path.exists(cfg.resume_path) and cfg.resume_path.endswith("ckpt"):
        main_dict = torch.load(cfg.resume_path,
                               map_location=device)['state_dict']

        main_dict = {
            k: v
            for k, v in main_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape and (
                'reconEngine' not in k) and ("normal_filter" not in k) and (
                    'voxelization' not in k)
        }
        print(colored(f"Resume MLP weights from {cfg.resume_path}", 'green'))

    if os.path.exists(cfg.normal_path) and cfg.normal_path.endswith("ckpt"):
        normal_dict = torch.load(cfg.normal_path,
                                 map_location=device)['state_dict']

        for key in normal_dict.keys():
            normal_dict = rename(normal_dict, key,
                                 key.replace("netG", "netG.normal_filter"))

        normal_dict = {
            k: v
            for k, v in normal_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        print(colored(f"Resume normal model from {cfg.normal_path}", 'green'))

    model_dict.update(main_dict)
    model_dict.update(normal_dict)
    model.load_state_dict(model_dict)

    model.netG = model.netG.to(device)
    model.reconEngine = model.reconEngine.to(device)

    model.netG.training = False
    model.netG.eval()

    del main_dict
    del normal_dict
    del model_dict

    return model


def read_smpl_constants(folder):
    """Load smpl vertex code"""
    smpl_vtx_std = np.loadtxt(os.path.join(folder, 'vertices.txt'))
    min_x = np.min(smpl_vtx_std[:, 0])
    max_x = np.max(smpl_vtx_std[:, 0])
    min_y = np.min(smpl_vtx_std[:, 1])
    max_y = np.max(smpl_vtx_std[:, 1])
    min_z = np.min(smpl_vtx_std[:, 2])
    max_z = np.max(smpl_vtx_std[:, 2])

    smpl_vtx_std[:, 0] = (smpl_vtx_std[:, 0] - min_x) / (max_x - min_x)
    smpl_vtx_std[:, 1] = (smpl_vtx_std[:, 1] - min_y) / (max_y - min_y)
    smpl_vtx_std[:, 2] = (smpl_vtx_std[:, 2] - min_z) / (max_z - min_z)
    smpl_vertex_code = np.float32(np.copy(smpl_vtx_std))
    """Load smpl faces & tetrahedrons"""
    smpl_faces = np.loadtxt(os.path.join(folder, 'faces.txt'),
                            dtype=np.int32) - 1
    smpl_face_code = (smpl_vertex_code[smpl_faces[:, 0]] +
                      smpl_vertex_code[smpl_faces[:, 1]] +
                      smpl_vertex_code[smpl_faces[:, 2]]) / 3.0
    smpl_tetras = np.loadtxt(os.path.join(folder, 'tetrahedrons.txt'),
                             dtype=np.int32) - 1

    return smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras


def surface_field_deformation(xyz,  de_nn_verts, de_nn_normals, ori_nn_verts, ori_nn_normals):
    '''
    xyz: [B, N, 3]
    de_nn_verts: [B, N, 3]
    de_nn_normals: [B, N, 3]
    ori_nn_verts: [B, N, 3]
    ori_nn_normals: [B, N, 3]
    '''
    vector=xyz-de_nn_verts # [B, N, 3]
    delta=torch.sum(vector*de_nn_normals, dim=-1, keepdim=True)*ori_nn_normals
    ori_xyz=ori_nn_verts+delta
    
    return ori_xyz # the deformed xyz
    

def feat_select(feat, select):

    # feat [B, featx2, N]
    # select [B, 1, N]
    # return [B, feat, N]

    dim = feat.shape[1] // 2
    idx = torch.tile((1-select), (1, dim, 1))*dim + \
        torch.arange(0, dim).unsqueeze(0).unsqueeze(2).type_as(select)
    feat_select = torch.gather(feat, 1, idx.long())

    return feat_select

def get_visibility_color(xy, z, faces):
    """get the visibility of vertices

    Args:
        xy (torch.tensor): [N,2]
        z (torch.tensor): [N,1]
        faces (torch.tensor): [N,3]
        size (int): resolution of rendered image
    """

    xyz = torch.cat((xy, -z), dim=1)
    xyz = (xyz + 1.0) / 2.0
    faces = faces.long()

    rasterizer = Pytorch3dRasterizer(image_size=2**12)
    meshes_screen = Meshes(verts=xyz[None, ...], faces=faces[None, ...])
    raster_settings = rasterizer.raster_settings

    pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
        meshes_screen,
        image_size=raster_settings.image_size,
        blur_radius=raster_settings.blur_radius,
        faces_per_pixel=raster_settings.faces_per_pixel,
        bin_size=raster_settings.bin_size,
        max_faces_per_bin=raster_settings.max_faces_per_bin,
        perspective_correct=raster_settings.perspective_correct,
        cull_backfaces=raster_settings.cull_backfaces,
    )

    vis_vertices_id = torch.unique(faces[torch.unique(pix_to_face), :])
    vis_mask = torch.zeros(size=(z.shape[0], 1))
    vis_mask[vis_vertices_id] = 1.0

    # 新增的部分: 检测边缘像素
    edge_mask = torch.zeros_like(pix_to_face)
    offset=1
    for i in range(-1-offset, 2+offset):
        for j in range(-1-offset, 2+offset):
            if i == 0 and j == 0:
                continue
            shifted = torch.roll(pix_to_face, shifts=(i,j), dims=(0,1))
            edge_mask = torch.logical_or(edge_mask, shifted == -1)

    # 更新可见性掩码
    edge_faces = torch.unique(pix_to_face[edge_mask])
    edge_vertices = torch.unique(faces[edge_faces])
    vis_mask[edge_vertices] = 0.0

    return vis_mask


def get_visibility(xy, z, faces):
    """get the visibility of vertices

    Args:
        xy (torch.tensor): [N,2]
        z (torch.tensor): [N,1]
        faces (torch.tensor): [N,3]
        size (int): resolution of rendered image
    """

    xyz = torch.cat((xy, -z), dim=1)
    xyz = (xyz + 1.0) / 2.0
    faces = faces.long()

    rasterizer = Pytorch3dRasterizer(image_size=2**12)
    meshes_screen = Meshes(verts=xyz[None, ...], faces=faces[None, ...])
    raster_settings = rasterizer.raster_settings

    pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
        meshes_screen,
        image_size=raster_settings.image_size,
        blur_radius=raster_settings.blur_radius,
        faces_per_pixel=raster_settings.faces_per_pixel,
        bin_size=raster_settings.bin_size,
        max_faces_per_bin=raster_settings.max_faces_per_bin,
        perspective_correct=raster_settings.perspective_correct,
        cull_backfaces=raster_settings.cull_backfaces,
    )

    vis_vertices_id = torch.unique(faces[torch.unique(pix_to_face), :])
    vis_mask = torch.zeros(size=(z.shape[0], 1))
    vis_mask[vis_vertices_id] = 1.0

    # print("------------------------\n")
    # print(f"keep points : {vis_mask.sum()/len(vis_mask)}")

    return vis_mask


def barycentric_coordinates_of_projection(points, vertices):
    ''' https://github.com/MPI-IS/mesh/blob/master/mesh/geometry/barycentric_coordinates_of_projection.py
    '''
    """Given a point, gives projected coords of that point to a triangle
    in barycentric coordinates.
    See
        **Heidrich**, Computing the Barycentric Coordinates of a Projected Point, JGT 05
        at http://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf
    
    :param p: point to project. [B, 3]
    :param v0: first vertex of triangles. [B, 3]
    :returns: barycentric coordinates of ``p``'s projection in triangle defined by ``q``, ``u``, ``v``
            vectorized so ``p``, ``q``, ``u``, ``v`` can all be ``3xN``
    """
    #(p, q, u, v)
    v0, v1, v2 = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    p = points

    q = v0
    u = v1 - v0
    v = v2 - v0
    n = torch.cross(u, v)
    s = torch.sum(n * n, dim=1)
    # If the triangle edges are collinear, cross-product is zero,
    # which makes "s" 0, which gives us divide by zero. So we
    # make the arbitrary choice to set s to epsv (=numpy.spacing(1)),
    # the closest thing to zero
    s[s == 0] = 1e-6
    oneOver4ASquared = 1.0 / s
    w = p - q
    b2 = torch.sum(torch.cross(u, w) * n, dim=1) * oneOver4ASquared
    b1 = torch.sum(torch.cross(w, v) * n, dim=1) * oneOver4ASquared
    weights = torch.stack((1 - b1 - b2, b1, b2), dim=-1)
    # check barycenric weights
    # p_n = v0*weights[:,0:1] + v1*weights[:,1:2] + v2*weights[:,2:3]
    return weights


def cal_sdf_batch(verts, faces, cmaps, vis, points):

    # verts [B, N_vert, 3]
    # faces [B, N_face, 3]
    # triangles [B, N_face, 3, 3]
    # points [B, N_point, 3]
    # cmaps [B, N_vert, 3]

    Bsize = points.shape[0]

    normals = Meshes(verts, faces).verts_normals_padded()

    # SMPL has watertight mesh, but SMPL-X has two eyeballs and open mouth
    # 1. remove eye_ball faces from SMPL-X: 9928-9383, 10474-9929
    # 2. fill mouth holes with 30 more faces

    if verts.shape[1] == 10475:
        faces = faces[:, ~SMPLX().smplx_eyeball_fid_mask]
        mouth_faces = torch.as_tensor(
            SMPLX().smplx_mouth_fid).unsqueeze(0).repeat(Bsize, 1,
                                                         1).to(faces.device)
        faces = torch.cat([faces, mouth_faces], dim=1)

    triangles = face_vertices(verts, faces)
    normals = face_vertices(normals, faces)
    cmaps = face_vertices(cmaps, faces)
    vis = face_vertices(vis, faces)

    residues, pts_ind, _ = point_to_mesh_distance(points, triangles)
    closest_triangles = torch.gather(
        triangles, 1, pts_ind[:, :, None, None].expand(-1, -1, 3,
                                                       3)).view(-1, 3, 3)
    closest_normals = torch.gather(
        normals, 1, pts_ind[:, :, None, None].expand(-1, -1, 3,
                                                     3)).view(-1, 3, 3)
    closest_cmaps = torch.gather(
        cmaps, 1, pts_ind[:, :, None, None].expand(-1, -1, 3,
                                                   3)).view(-1, 3, 3)
    closest_vis = torch.gather(vis, 1, pts_ind[:, :, None,
                                               None].expand(-1, -1, 3,
                                                            1)).view(-1, 3, 1)
    bary_weights = barycentric_coordinates_of_projection(
        points.view(-1, 3), closest_triangles)

    pts_cmap = (closest_cmaps * bary_weights[:, :, None]).sum(1).unsqueeze(0)
    pts_vis = (closest_vis *
               bary_weights[:, :, None]).sum(1).unsqueeze(0).ge(1e-1)
    pts_norm = (closest_normals *
                bary_weights[:, :, None]).sum(1).unsqueeze(0) * torch.tensor(
                    [-1.0, 1.0, -1.0]).type_as(normals)
    pts_norm = F.normalize(pts_norm, dim=2)
    pts_dist = torch.sqrt(residues) / torch.sqrt(torch.tensor(3))

    pts_signs = 2.0 * (check_sign(verts, faces[0], points).float() - 0.5)
    pts_sdf = (pts_dist * pts_signs).unsqueeze(-1)

    return pts_sdf.view(Bsize, -1,
                        1), pts_norm.view(Bsize, -1, 3), pts_cmap.view(
                            Bsize, -1, 3), pts_vis.view(Bsize, -1, 1)


def orthogonal(points, calibrations, transforms=None):
    '''
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 3, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts


def projection(points, calib):
    if torch.is_tensor(points):
        calib = torch.as_tensor(calib) if not torch.is_tensor(calib) else calib
        return torch.mm(calib[:3, :3], points.T).T + calib[:3, 3]
    else:
        return np.matmul(calib[:3, :3], points.T).T + calib[:3, 3]


def load_calib(calib_path):
    calib_data = np.loadtxt(calib_path, dtype=float)
    extrinsic = calib_data[:4, :4]
    intrinsic = calib_data[4:8, :4]
    calib_mat = np.matmul(intrinsic, extrinsic)
    calib_mat = torch.from_numpy(calib_mat).float()
    return calib_mat


def load_obj_mesh_for_Hoppe(mesh_file):
    vertex_data = []
    face_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(
                    map(lambda x: int(x.split('/')[0]),
                        [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)

    vertices = np.array(vertex_data)
    faces = np.array(face_data)
    faces[faces > 0] -= 1

    normals, _ = compute_normal(vertices, faces)

    return vertices, normals, faces


def load_obj_mesh_with_color(mesh_file):
    vertex_data = []
    color_data = []
    face_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
            c = list(map(float, values[4:7]))
            color_data.append(c)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(
                    map(lambda x: int(x.split('/')[0]),
                        [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)

    vertices = np.array(vertex_data)
    colors = np.array(color_data)
    faces = np.array(face_data)
    faces[faces > 0] -= 1

    return vertices, colors, faces


def load_obj_mesh(mesh_file, with_normal=False, with_texture=False):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(
                    map(lambda x: int(x.split('/')[0]),
                        [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)

            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(
                        map(lambda x: int(x.split('/')[1]),
                            [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(
                        map(lambda x: int(x.split('/')[2]),
                            [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)

    vertices = np.array(vertex_data)
    faces = np.array(face_data)
    faces[faces > 0] -= 1

    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data)
        face_uvs[face_uvs > 0] -= 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms, _ = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data)
            face_normals[face_normals > 0] -= 1
        return vertices, faces, norms, face_normals, uvs, face_uvs

    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        return vertices, faces, uvs, face_uvs

    if with_normal:
        norms = np.array(norm_data)
        norms = normalize_v3(norms)
        face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals

    return vertices, faces


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    vert_norms = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    face_norms = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(face_norms)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    vert_norms[faces[:, 0]] += face_norms
    vert_norms[faces[:, 1]] += face_norms
    vert_norms[faces[:, 2]] += face_norms
    normalize_v3(vert_norms)

    return vert_norms, face_norms


def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')
    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' %
                   (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()


def calculate_mIoU(outputs, labels):

    SMOOTH = 1e-6

    outputs = outputs.int()
    labels = labels.int()

    intersection = (
        outputs
        & labels).float().sum()  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum()  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH
                                     )  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(
        20 * (iou - 0.5), 0,
        10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded.mean().detach().cpu().numpy(
    )  # Or thresholded.mean() if you are interested in average across the batch


def mask_filter(mask, number=1000):
    """only keep {number} True items within a mask

    Args:
        mask (bool array): [N, ]
        number (int, optional): total True item. Defaults to 1000.
    """
    true_ids = np.where(mask)[0]
    keep_ids = np.random.choice(true_ids, size=number)
    filter_mask = np.isin(np.arange(len(mask)), keep_ids)

    return filter_mask


def query_mesh(path):

    verts, faces_idx, _ = load_obj(path)

    return verts, faces_idx.verts_idx


def add_alpha(colors, alpha=0.7):

    colors_pad = np.pad(colors, ((0, 0), (0, 1)),
                        mode='constant',
                        constant_values=alpha)

    return colors_pad


def get_optim_grid_image(per_loop_lst, loss=None, nrow=4, type='smpl'):

    font_path = os.path.join(os.path.dirname(__file__), "tbfo.ttf")
    font = ImageFont.truetype(font_path, 30)
    grid_img = torchvision.utils.make_grid(torch.cat(per_loop_lst, dim=0),
                                           nrow=nrow)
    grid_img = Image.fromarray(
        ((grid_img.permute(1, 2, 0).detach().cpu().numpy() + 1.0) * 0.5 *
         255.0).astype(np.uint8))

    # add text
    draw = ImageDraw.Draw(grid_img)
    grid_size = 512
    if loss is not None:
        draw.text((10, 5), f"error: {loss:.3f}", (255, 0, 0), font=font)

    if type == 'smpl':
        for col_id, col_txt in enumerate([
                'image', 'smpl-norm(render)', 'cloth-norm(pred)', 'diff-norm',
                'diff-mask'
        ]):
            draw.text((10 + (col_id * grid_size), 5),
                      col_txt, (255, 0, 0),
                      font=font)
    elif type == 'cloth':
        for col_id, col_txt in enumerate(
            ['cloth-norm(recon)']):
            draw.text((10 + (col_id * grid_size), 5),
                      col_txt, (255, 0, 0),
                      font=font)
        for col_id, col_txt in enumerate(['0', '90', '180', '270']):
            draw.text((10 + (col_id * grid_size), grid_size * 2 + 5),
                      col_txt, (255, 0, 0),
                      font=font)
    else:
        print(f"{type} should be 'smpl' or 'cloth'")

    grid_img = grid_img.resize((grid_img.size[0], grid_img.size[1]),
                               Image.LANCZOS)

    return grid_img


def clean_mesh(verts, faces):

    device = verts.device

    mesh_lst = trimesh.Trimesh(verts.detach().cpu().numpy(),
                               faces.detach().cpu().numpy())
    mesh_lst = mesh_lst.split(only_watertight=False)
    comp_num = [mesh.vertices.shape[0] for mesh in mesh_lst]
    mesh_clean = mesh_lst[comp_num.index(max(comp_num))]

    final_verts = torch.as_tensor(mesh_clean.vertices).float().to(device)
    final_faces = torch.as_tensor(mesh_clean.faces).int().to(device)

    return final_verts, final_faces


def merge_mesh(verts_A, faces_A, verts_B, faces_B, color=False):

    sep_mesh = trimesh.Trimesh(np.concatenate([verts_A, verts_B], axis=0),
                               np.concatenate(
                                   [faces_A, faces_B + faces_A.max() + 1],
                                   axis=0),
                               maintain_order=True,
                               process=False)
    if color:
        colors = np.ones_like(sep_mesh.vertices)
        colors[:verts_A.shape[0]] *= np.array([255.0, 0.0, 0.0])
        colors[verts_A.shape[0]:] *= np.array([0.0, 255.0, 0.0])
        sep_mesh.visual.vertex_colors = colors

    # union_mesh = trimesh.boolean.union([trimesh.Trimesh(verts_A, faces_A),
    #                                     trimesh.Trimesh(verts_B, faces_B)], engine='blender')

    return sep_mesh


def mesh_move(mesh_lst, step, scale=1.0):

    trans = np.array([1.0, 0.0, 0.0]) * step

    resize_matrix = trimesh.transformations.scale_and_translate(
        scale=(scale), translate=trans)

    results = []

    for mesh in mesh_lst:
        mesh.apply_transform(resize_matrix)
        results.append(mesh)

    return results


def rescale_smpl(fitted_path, scale=100, translate=(0, 0, 0)):

    fitted_body = trimesh.load(fitted_path,
                               process=False,
                               maintain_order=True,
                               skip_materials=True)
    resize_matrix = trimesh.transformations.scale_and_translate(
        scale=(scale), translate=translate)

    fitted_body.apply_transform(resize_matrix)

    return np.array(fitted_body.vertices)


class SMPLX():

    def __init__(self):

        self.current_dir = "./data/smpl_related"  # new smplx file in ECON folder

        self.smpl_verts_path = osp.join(self.current_dir,
                                        "smpl_data/smpl_verts.npy")
        self.smpl_faces_path = osp.join(self.current_dir,
                                        "smpl_data/smpl_faces.npy")
        self.smplx_verts_path = osp.join(self.current_dir,
                                         "smpl_data/smplx_verts.npy")
        self.smplx_faces_path = osp.join(self.current_dir,
                                         "smpl_data/smplx_faces.npy")
        self.cmap_vert_path = osp.join(self.current_dir,
                                       "smpl_data/smplx_cmap.npy")

        self.smplx_to_smplx_path = osp.join(self.current_dir,
                                            "smpl_data/smplx_to_smpl.pkl")

        self.smplx_eyeball_fid = osp.join(self.current_dir,
                                          "smpl_data/eyeball_fid.npy")
        self.smplx_fill_mouth_fid = osp.join(self.current_dir,
                                             "smpl_data/fill_mouth_fid.npy")

        self.smplx_faces = np.load(self.smplx_faces_path)
        self.smplx_verts = np.load(self.smplx_verts_path)
        self.smpl_verts = np.load(self.smpl_verts_path)
        self.smpl_faces = np.load(self.smpl_faces_path)

        self.smplx_eyeball_fid_mask = np.load(self.smplx_eyeball_fid)
        self.smplx_mouth_fid = np.load(self.smplx_fill_mouth_fid)

        self.smplx_to_smpl = cPickle.load(open(self.smplx_to_smplx_path, 'rb'))

        self.model_dir = osp.join(self.current_dir, "models")
        self.tedra_dir = osp.join(self.current_dir, "../tedra_data")



        # copy from econ
        self.smplx_flame_vid_path = osp.join(
            self.current_dir, "smpl_data/FLAME_SMPLX_vertex_ids.npy"
        )
        self.smplx_mano_vid_path = osp.join(self.current_dir, "smpl_data/MANO_SMPLX_vertex_ids.pkl")
        self.smpl_vert_seg_path = osp.join(
            osp.dirname(__file__), "../../lib/common/smpl_vert_segmentation.json"
        )
        self.front_flame_path = osp.join(self.current_dir, "smpl_data/FLAME_face_mask_ids.npy")
        self.smplx_vertex_lmkid_path = osp.join(
            self.current_dir, "smpl_data/smplx_vertex_lmkid.npy"
        )

        self.smplx_vertex_lmkid = np.load(self.smplx_vertex_lmkid_path)
        self.smpl_vert_seg = json.load(open(self.smpl_vert_seg_path))
        self.smpl_mano_vid = np.concatenate(
            [
                self.smpl_vert_seg["rightHand"], self.smpl_vert_seg["rightHandIndex1"],
                self.smpl_vert_seg["leftHand"], self.smpl_vert_seg["leftHandIndex1"]
            ]
        )

        self.smplx_mano_vid_dict = np.load(self.smplx_mano_vid_path, allow_pickle=True)
        self.smplx_mano_vid = np.concatenate(
            [self.smplx_mano_vid_dict["left_hand"], self.smplx_mano_vid_dict["right_hand"]]
        )
        self.smplx_flame_vid = np.load(self.smplx_flame_vid_path, allow_pickle=True)
        self.smplx_front_flame_vid = self.smplx_flame_vid[np.load(self.front_flame_path)]


        # hands
        self.smplx_mano_vertex_mask = torch.zeros(self.smplx_verts.shape[0], ).index_fill_(
            0, torch.tensor(self.smplx_mano_vid), 1.0
        )
        self.smpl_mano_vertex_mask = torch.zeros(self.smpl_verts.shape[0], ).index_fill_(
            0, torch.tensor(self.smpl_mano_vid), 1.0
        )

         # face
        self.front_flame_vertex_mask = torch.zeros(self.smplx_verts.shape[0], ).index_fill_(
            0, torch.tensor(self.smplx_front_flame_vid), 1.0
        )
        self.eyeball_vertex_mask = torch.zeros(self.smplx_verts.shape[0], ).index_fill_(
            0, torch.tensor(self.smplx_faces[self.smplx_eyeball_fid_mask].flatten()), 1.0
        )


        self.ghum_smpl_pairs = torch.tensor(
            [
                (0, 24), (2, 26), (5, 25), (7, 28), (8, 27), (11, 16), (12, 17), (13, 18), (14, 19),
                (15, 20), (16, 21), (17, 39), (18, 44), (19, 36), (20, 41), (21, 35), (22, 40),
                (23, 1), (24, 2), (25, 4), (26, 5), (27, 7), (28, 8), (29, 31), (30, 34), (31, 29),
                (32, 32)
            ]
        ).long()

        # smpl-smplx correspondence
        self.smpl_joint_ids_24 = np.arange(22).tolist() + [68, 73]
        self.smpl_joint_ids_24_pixie = np.arange(22).tolist() + [61 + 68, 72 + 68]
        self.smpl_joint_ids_45 = np.arange(22).tolist() + [68, 73] + np.arange(55, 76).tolist()

        self.extra_joint_ids = np.array(
            [
                61, 72, 66, 69, 58, 68, 57, 56, 64, 59, 67, 75, 70, 65, 60, 61, 63, 62, 76, 71, 72,
                74, 73
            ]
        )

        self.extra_joint_ids += 68

        self.smpl_joint_ids_45_pixie = (np.arange(22).tolist() + self.extra_joint_ids.tolist())


    def cmap_smpl_vids(self, type):

        # keys:
        # closest_faces -   [6890, 3] with smplx vert_idx
        # bc            -   [6890, 3] with barycentric weights

        cmap_smplx = torch.as_tensor(np.load(self.cmap_vert_path)).float()
        if type == 'smplx':
            return cmap_smplx
        elif type == 'smpl':
            bc = torch.as_tensor(self.smplx_to_smpl['bc'].astype(np.float32))
            closest_faces = self.smplx_to_smpl['closest_faces'].astype(
                np.int32)

            cmap_smpl = torch.einsum('bij, bi->bj', cmap_smplx[closest_faces],
                                     bc)

            return cmap_smpl



# copy from ECON

def apply_face_mask(mesh, face_mask):

    mesh.update_faces(face_mask)
    mesh.remove_unreferenced_vertices()

    return mesh


def apply_vertex_mask(mesh, vertex_mask):

    faces_mask = vertex_mask[mesh.faces].any(dim=1)
    mesh = apply_face_mask(mesh, faces_mask)

    return mesh


def apply_vertex_face_mask(mesh, vertex_mask, face_mask):

    faces_mask = vertex_mask[mesh.faces].any(dim=1) * torch.tensor(face_mask)
    mesh.update_faces(faces_mask)
    mesh.remove_unreferenced_vertices()

    return mesh


def clean_floats(mesh):
    thres = mesh.vertices.shape[0] * 1e-2
    mesh_lst = mesh.split(only_watertight=False)
    clean_mesh_lst = [mesh for mesh in mesh_lst if mesh.vertices.shape[0] > thres]
    return sum(clean_mesh_lst)

def isin(input, test_elements):
    # 扩展输入和测试元素的维度以进行广播
    input = input.unsqueeze(-1)
    test_elements = test_elements.unsqueeze(0)
    
    # 比较两个张量的元素
    comparison_result = torch.eq(input, test_elements)
    
    # 沿着新添加的维度进行求和，以检查每个输入元素是否在测试元素中
    isin_result = comparison_result.sum(-1).bool()
    
    return isin_result



def part_removal(full_mesh, part_mesh, thres, device, smpl_obj, region, clean=True):

    smpl_tree = cKDTree(smpl_obj.vertices)
    SMPL_container = SMPLX()

    from lib.dataset.PointFeat import ECON_PointFeat

    part_extractor = ECON_PointFeat(
        torch.tensor(part_mesh.vertices).unsqueeze(0).to(device),
        torch.tensor(part_mesh.faces).unsqueeze(0).to(device)
    )

    (part_dist, _) = part_extractor.query(torch.tensor(full_mesh.vertices).unsqueeze(0).to(device))

    remove_mask = part_dist < thres

    if region == "hand":
        _, idx = smpl_tree.query(full_mesh.vertices, k=1)
        full_lmkid = SMPL_container.smplx_vertex_lmkid[idx]
        remove_mask = torch.logical_and(
            remove_mask,
            torch.tensor(full_lmkid >= 20).type_as(remove_mask).unsqueeze(0)
        )

    elif region == "face":
        _, idx = smpl_tree.query(full_mesh.vertices, k=5)
        face_space_mask = isin(
            torch.tensor(idx), torch.tensor(SMPL_container.smplx_front_flame_vid)
        )
        remove_mask = torch.logical_and(
            remove_mask,
            face_space_mask.any(dim=1).type_as(remove_mask).unsqueeze(0)
        )

    BNI_part_mask = ~(remove_mask).flatten()[full_mesh.faces].any(dim=1)
    full_mesh.update_faces(BNI_part_mask.detach().cpu())
    full_mesh.remove_unreferenced_vertices()

    if clean:
        full_mesh = clean_floats(full_mesh)

    return full_mesh

def keep_largest(mesh):
    mesh_lst = mesh.split(only_watertight=False)
    keep_mesh = mesh_lst[0]
    for mesh in mesh_lst:
        if mesh.vertices.shape[0] > keep_mesh.vertices.shape[0]:
            keep_mesh = mesh
    return keep_mesh


def poisson(mesh, obj_path, depth=10, decimation=True):

    pcd_path = obj_path[:-4] + "_soups.ply"
    assert (mesh.vertex_normals.shape[1] == 3)
    mesh.export(pcd_path)
    pcl = o3d.io.read_point_cloud(pcd_path)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcl, depth=depth, n_threads=6
        )

    # only keep the largest component
    largest_mesh = keep_largest(trimesh.Trimesh(np.array(mesh.vertices), np.array(mesh.triangles)))
    largest_mesh.export(obj_path)

    if decimation:
        # mesh decimation for faster rendering
        low_res_mesh = largest_mesh.simplify_quadratic_decimation(50000)
        return low_res_mesh
    else:
        return largest_mesh