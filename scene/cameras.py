#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
np.random.seed(42)
import os
import torch
from torch import nn
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
import random
random.seed(42)
# import lietorch
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, rbt2qtvec
import json
import pypose as pp
from utils.graphics_utils import fov2focal
import lietorch
# from gaussian_renderer import render


def qtvec2viewmatrix(viewqtvec):
    viewmatrix = np.zeros(16)
    viewmatrix[0] = 1 - 2 * (viewqtvec[4] * viewqtvec[4] + viewqtvec[5] * viewqtvec[5])
    viewmatrix[1] = 2 * (viewqtvec[3] * viewqtvec[4] + viewqtvec[6] * viewqtvec[5])
    viewmatrix[2] = 2 * (viewqtvec[3] * viewqtvec[5] - viewqtvec[6] * viewqtvec[4])
    viewmatrix[3] = 0
    viewmatrix[4] = 2 * (viewqtvec[3] * viewqtvec[4] - viewqtvec[6] * viewqtvec[5])
    viewmatrix[5] = 1 - 2 * (viewqtvec[3] * viewqtvec[3] + viewqtvec[5] * viewqtvec[5])
    viewmatrix[6] = 2 * (viewqtvec[4] * viewqtvec[5] + viewqtvec[6] * viewqtvec[3])
    viewmatrix[7] = 0
    viewmatrix[8] = 2 * (viewqtvec[3] * viewqtvec[5] + viewqtvec[6] * viewqtvec[4]) 
    viewmatrix[9] = 2 * (viewqtvec[4] * viewqtvec[5] - viewqtvec[6] * viewqtvec[3])
    viewmatrix[10] = 1 - 2 * (viewqtvec[3] * viewqtvec[3] + viewqtvec[4] * viewqtvec[4])
    viewmatrix[11] = 0
    viewmatrix[12] = viewqtvec[0] 
    viewmatrix[13] = viewqtvec[1] 
    viewmatrix[14] = viewqtvec[2]
    viewmatrix[15] = 1
    return viewmatrix.reshape(4,4)


class CameraQuat(nn.Module):
    def __init__(self, colmap_id, R, T, qvec, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, cx, cy,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(CameraQuat, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.fx = fov2focal(self.FoVx, self.image_width)
        self.fy = fov2focal(self.FoVy, self.image_height)
        self.cx = cx
        self.cy = cy

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        self.rotation_activation = torch.nn.functional.normalize
        world_view_transform = rbt2qtvec(torch.tensor(getWorld2View2(R, T, trans, scale))).cuda() 
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = self.projection_matrix
        self.world_view_transform_ = nn.Parameter(torch.cat((world_view_transform[:3], 
                                                            self.rotation_activation(world_view_transform[3:], dim = 0))).requires_grad_(True))

    @property
    def world_view_transform(self):
        return (torch.cat((self.world_view_transform_[:3], self.rotation_activation(self.world_view_transform_[3:], dim = 0))))


class CameraLie(nn.Module):
    def __init__(self, colmap_id, R, T, qvec, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, cx, cy,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(CameraLie, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.qvec = qvec


        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.fx = fov2focal(self.FoVx, self.image_width)
        self.fy = fov2focal(self.FoVy, self.image_height)
        self.cx = cx
        self.cy = cy

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        self.rotation_activation = torch.nn.functional.normalize
        world_view_transform = rbt2qtvec(torch.tensor(getWorld2View2(R, T, trans, scale))).cuda()
        world_view_transform = torch.cat((world_view_transform[:3], self.rotation_activation(world_view_transform[3:], dim = 0)))
        self.world_view_transform_lie = pp.SE3(torch.tensor(world_view_transform))
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = self.projection_matrix
        self.world_view_transform_ = pp.Parameter(self.world_view_transform_lie)
        
    @property
    def world_view_transform(self):
        return self.world_view_transform_ 


class CameraLieTorch(nn.Module):
    def __init__(self, colmap_id, R, T, qvec, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, cx, cy,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(CameraLieTorch, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name


        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
       
        self.fx = fov2focal(self.FoVx, self.image_width)
        self.fy = fov2focal(self.FoVy, self.image_height)
        self.cx = cx
        self.cy = cy


        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        self.rotation_activation = torch.nn.functional.normalize

        world_view_transform = rbt2qtvec(torch.tensor(getWorld2View2(R, T, trans, scale))).cuda()

        self.world_view_transform_ = lietorch.LieGroupParameter(lietorch.SE3(world_view_transform.unsqueeze(0)))

        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = self.projection_matrix
        self.world_view_transform = nn.Parameter(self.world_view_transform_[0].data.detach().requires_grad_())

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform_qvec, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform_qvec
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(world_view_transform)
        self.camera_center = view_inv[3][:3]

