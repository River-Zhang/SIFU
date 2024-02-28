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

from pytorch3d.renderer import (
    BlendParams,
    blending,
    look_at_view_transform,
    FoVOrthographicCameras,
    PointLights,
    RasterizationSettings,
    PointsRasterizationSettings,
    PointsRenderer,
    AlphaCompositor,
    PointsRasterizer,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesVertex,
)
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.structures import Meshes
from lib.dataset.mesh_util import get_visibility, get_visibility_color

import lib.common.render_utils as util
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import cv2
import math
from termcolor import colored


def image2vid(images, vid_path):

    w, h = images[0].size
    videodims = (w, h)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(vid_path, fourcc, len(images) / 5.0, videodims)
    for image in images:
        video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    video.release()


def query_color(verts, faces, image, device, predicted_color):
    """query colors from points and image

    Args:
        verts ([B, 3]): [query verts]
        faces ([M, 3]): [query faces]
        image ([B, 3, H, W]): [full image]

    Returns:
        [np.float]: [return colors]
    """

    verts = verts.float().to(device)
    faces = faces.long().to(device)
    predicted_color=predicted_color.to(device)
    (xy, z) = verts.split([2, 1], dim=1)
    visibility = get_visibility_color(xy, z, faces[:, [0, 2, 1]]).flatten()
    uv = xy.unsqueeze(0).unsqueeze(2)  # [B, N, 2]
    uv = uv * torch.tensor([1.0, -1.0]).type_as(uv)
    colors = (torch.nn.functional.grid_sample(
        image, uv, align_corners=True)[0, :, :, 0].permute(1, 0) +
              1.0) * 0.5 * 255.0
    colors[visibility == 0.0]=(predicted_color* 255.0)[visibility == 0.0]

    return colors.detach().cpu()


class cleanShader(torch.nn.Module):

    def __init__(self, device="cpu", cameras=None, blend_params=None):
        super().__init__()
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams(
        )

    def forward(self, fragments, meshes, **kwargs):
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of TexturedSoftPhongShader"

            raise ValueError(msg)

        # get renderer output
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = blending.softmax_rgb_blend(texels,
                                            fragments,
                                            blend_params,
                                            znear=-256,
                                            zfar=256)

        return images


class Render:

    def __init__(self, size=512, device=torch.device("cuda:0")):
        self.device = device
        self.size = size

        # camera setting
        self.dis = 100.0
        self.scale = 100.0
        self.mesh_y_center = 0.0

        self.reload_cam()

        self.type = "color"

        self.mesh = None
        self.deform_mesh = None
        self.pcd = None
        self.renderer = None
        self.meshRas = None

        self.uv_rasterizer = util.Pytorch3dRasterizer(self.size)

    def reload_cam(self):

        self.cam_pos = [
            (0, self.mesh_y_center, self.dis),
            (self.dis, self.mesh_y_center, 0),
            (0, self.mesh_y_center, -self.dis),
            (-self.dis, self.mesh_y_center, 0),
            (0,self.mesh_y_center+self.dis,0),
            (0,self.mesh_y_center-self.dis,0),
        ]

    def get_camera(self, cam_id):
        
        if cam_id == 4:
            R, T = look_at_view_transform(
                eye=[self.cam_pos[cam_id]],
                at=((0, self.mesh_y_center, 0), ),
                up=((0, 0, 1), ),
            )
        elif cam_id == 5:
            R, T = look_at_view_transform(
                eye=[self.cam_pos[cam_id]],
                at=((0, self.mesh_y_center, 0), ),
                up=((0, 0, 1), ),
            )

        else:
            R, T = look_at_view_transform(
                eye=[self.cam_pos[cam_id]],
                at=((0, self.mesh_y_center, 0), ),
                up=((0, 1, 0), ),
            )

        camera = FoVOrthographicCameras(
            device=self.device,
            R=R,
            T=T,
            znear=100.0,
            zfar=-100.0,
            max_y=100.0,
            min_y=-100.0,
            max_x=100.0,
            min_x=-100.0,
            scale_xyz=(self.scale * np.ones(3), ),
        )

        return camera

    def init_renderer(self, camera, type="clean_mesh", bg="gray"):

        if "mesh" in type:

            # rasterizer
            self.raster_settings_mesh = RasterizationSettings(
                image_size=self.size,
                blur_radius=np.log(1.0 / 1e-4) * 1e-7,
                faces_per_pixel=30,
            )
            self.meshRas = MeshRasterizer(
                cameras=camera, raster_settings=self.raster_settings_mesh)

        if bg == "black":
            blendparam = BlendParams(1e-4, 1e-4, (0.0, 0.0, 0.0))
        elif bg == "white":
            blendparam = BlendParams(1e-4, 1e-8, (1.0, 1.0, 1.0))
        elif bg == "gray":
            blendparam = BlendParams(1e-4, 1e-8, (0.5, 0.5, 0.5))

        if type == "ori_mesh":

            lights = PointLights(
                device=self.device,
                ambient_color=((0.8, 0.8, 0.8), ),
                diffuse_color=((0.2, 0.2, 0.2), ),
                specular_color=((0.0, 0.0, 0.0), ),
                location=[[0.0, 200.0, 0.0]],
            )

            self.renderer = MeshRenderer(
                rasterizer=self.meshRas,
                shader=SoftPhongShader(
                    device=self.device,
                    cameras=camera,
                    lights=None,
                    blend_params=blendparam,
                ),
            )

        if type == "silhouette":
            self.raster_settings_silhouette = RasterizationSettings(
                image_size=self.size,
                blur_radius=np.log(1.0 / 1e-4 - 1.0) * 5e-5,
                faces_per_pixel=50,
                cull_backfaces=True,
            )

            self.silhouetteRas = MeshRasterizer(
                cameras=camera,
                raster_settings=self.raster_settings_silhouette)
            self.renderer = MeshRenderer(rasterizer=self.silhouetteRas,
                                         shader=SoftSilhouetteShader())

        if type == "pointcloud":
            self.raster_settings_pcd = PointsRasterizationSettings(
                image_size=self.size, radius=0.006, points_per_pixel=10)

            self.pcdRas = PointsRasterizer(
                cameras=camera, raster_settings=self.raster_settings_pcd)
            self.renderer = PointsRenderer(
                rasterizer=self.pcdRas,
                compositor=AlphaCompositor(background_color=(0, 0, 0)),
            )

        if type == "clean_mesh":

            self.renderer = MeshRenderer(
                rasterizer=self.meshRas,
                shader=cleanShader(device=self.device,
                                   cameras=camera,
                                   blend_params=blendparam),
            )

    def VF2Mesh(self, verts, faces, vertex_texture = None):

        if not torch.is_tensor(verts):
            verts = torch.tensor(verts)
        if not torch.is_tensor(faces):
            faces = torch.tensor(faces)

        if verts.ndimension() == 2:
            verts = verts.unsqueeze(0).float()
        if faces.ndimension() == 2:
            faces = faces.unsqueeze(0).long()

        verts = verts.to(self.device)
        faces = faces.to(self.device)
        if vertex_texture is not None:
            vertex_texture = vertex_texture.to(self.device)

        mesh = Meshes(verts, faces).to(self.device)

        if vertex_texture is None:
            mesh.textures = TexturesVertex(
                verts_features=(mesh.verts_normals_padded() + 1.0) * 0.5)#modify
        else:    
            mesh.textures = TexturesVertex(
                verts_features = vertex_texture.unsqueeze(0))#modify
        return mesh

    def load_meshes(self, verts, faces,offset=None, vertex_texture = None):
        """load mesh into the pytorch3d renderer

        Args:
            verts ([N,3]): verts
            faces ([N,3]): faces
            offset ([N,3]): offset
        """
        if offset is not None:
            verts = verts + offset

        if isinstance(verts, list):
            self.meshes = []
            for V, F in zip(verts, faces):
                if vertex_texture is None:
                    self.meshes.append(self.VF2Mesh(V, F))
                else:
                    self.meshes.append(self.VF2Mesh(V, F, vertex_texture))
        else:
            if vertex_texture is None:
                self.meshes = [self.VF2Mesh(verts, faces)]
            else:
                self.meshes = [self.VF2Mesh(verts, faces, vertex_texture)]

    def get_depth_map(self, cam_ids=[0, 2]):

        depth_maps = []
        for cam_id in cam_ids:
            self.init_renderer(self.get_camera(cam_id), "clean_mesh", "gray")
            fragments = self.meshRas(self.meshes[0])
            depth_map = fragments.zbuf[..., 0].squeeze(0)
            if cam_id == 2:
                depth_map = torch.fliplr(depth_map)
            depth_maps.append(depth_map)

        return depth_maps

    def get_rgb_image(self, cam_ids=[0, 2], bg='gray'):

        images = []
        for cam_id in range(len(self.cam_pos)):
            if cam_id in cam_ids:
                self.init_renderer(self.get_camera(cam_id), "clean_mesh", bg)
                if len(cam_ids) == 4:
                    rendered_img = (self.renderer(
                        self.meshes[0])[0:1, :, :, :3].permute(0, 3, 1, 2) -
                                    0.5) * 2.0
                else:
                    rendered_img = (self.renderer(
                        self.meshes[0])[0:1, :, :, :3].permute(0, 3, 1, 2) -
                                    0.5) * 2.0
                if cam_id == 2 and len(cam_ids) == 2:
                    rendered_img = torch.flip(rendered_img, dims=[3])
                images.append(rendered_img)

        return images

    def get_rendered_video(self, images, save_path):

        self.cam_pos = []
        for angle in range(360):
            self.cam_pos.append((
                100.0 * math.cos(np.pi / 180 * angle),
                self.mesh_y_center,
                100.0 * math.sin(np.pi / 180 * angle),
            ))

        old_shape = np.array(images[0].shape[:2])
        new_shape = np.around(
            (self.size / old_shape[0]) * old_shape).astype(np.int)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(save_path, fourcc, 10,
                                (self.size * len(self.meshes) +
                                 new_shape[1] * len(images), self.size))

        pbar = tqdm(range(len(self.cam_pos)))
        pbar.set_description(
            colored(f"exporting video {os.path.basename(save_path)}...",
                    "blue"))
        for cam_id in pbar:
            self.init_renderer(self.get_camera(cam_id), "clean_mesh", "gray")

            img_lst = [
                np.array(Image.fromarray(img).resize(new_shape[::-1])).astype(
                    np.uint8)[:, :, [2, 1, 0]] for img in images
            ]

            for mesh in self.meshes:
                rendered_img = ((self.renderer(mesh)[0, :, :, :3] *
                                 255.0).detach().cpu().numpy().astype(
                                     np.uint8))

                img_lst.append(rendered_img)
            final_img = np.concatenate(img_lst, axis=1)
            video.write(final_img)

        video.release()
        self.reload_cam()

    def get_silhouette_image(self, cam_ids=[0, 2]):

        images = []
        for cam_id in range(len(self.cam_pos)):
            if cam_id in cam_ids:
                self.init_renderer(self.get_camera(cam_id), "silhouette")
                rendered_img = self.renderer(self.meshes[0])[0:1, :, :, 3]
                if cam_id == 2 and len(cam_ids) == 2:
                    rendered_img = torch.flip(rendered_img, dims=[2])
                images.append(rendered_img)

        return images
