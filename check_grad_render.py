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

import torch
from scene import Scene
import os
import subprocess
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, pose_estimation_params
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr
import numpy as np
from utils.loss_utils import l1_loss, ssim
from copy import deepcopy
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

from diff_gaussian_rasterization import GaussianRasterizationSettings, rasterize_gaussians, GaussianRasterizer
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import math

def render_trainset(model_path, name, iteration, views, gaussians, pipeline, background):
    print(model_path)
    render_path = os.path.join(model_path, "train", "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)
    lpips = LPIPS(normalize=True).cuda()
    psnr_test = []
    ssim_test = []
    lpips_test = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        # rendering = rendering/rendering.max()
        rendering = torch.clamp(rendering, 0, 1)
        gt = view.original_image[0:3, :, :]
        psnr_value = psnr(rendering, gt).mean().item()
        ssim_value = ssim(rendering, gt).mean().item()
        # print(rendering.min(), rendering.max(), gt.min(), gt.max())
        lpips_value = lpips(rendering.unsqueeze(0), gt.unsqueeze(0)).mean().item()
        psnr_test.append(psnr_value)
        ssim_test.append(ssim_value)
        lpips_test.append(lpips_value)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx)                                                                         
                                                                        + '_' + format(psnr_value, '0.4f')
                                                                        + '_' + format(ssim_value, '0.4f')
                                                                        + '_' + format(lpips_value, '0.4f') + '_pred.png'))
        torchvision.utils.save_image(gt, os.path.join(render_path, '{0:05d}'.format(idx) + "_gt.png"))
    return np.mean(np.array(psnr_test)), np.mean(np.array(ssim_test)), np.mean(np.array(lpips_test)) 


def render_adjust_testset(scene, gaussians, pipe, opt, dataset, background, pose_adjustment, save_tag = 'test', img_tag = ''):
    img_tag = '_' + img_tag if img_tag else img_tag
    exit_psnr_parameter = 1e-5
    lambda_dssim = 0.2
    iterations  = opt.pose_lr_max_steps
    lpips = LPIPS(normalize=True).cuda()
    scene.cam_adjustment_setup(opt, 'test')

    render_path = os.path.join(dataset.model_path, save_tag, "ours_{}".format(scene.loaded_iter), "renders")

    makedirs(render_path, exist_ok=True)

    viewpoint_stack = scene.getTestCameras()

    optimized_cameras = {}
    psnrs_orig_test = []
    ssim_orig_test = []
    lpips_orig_test = []

    psnrs_adj_test = []
    ssim_adj_test = []
    lpips_adj_test = []

    for camera_index in tqdm(range(len(viewpoint_stack))):
        init_viewpoint_cam = deepcopy(viewpoint_stack[camera_index])
        viewpoint_cam = viewpoint_stack[camera_index]
        if pose_adjustment:
            prev_psnr = 1000
            max_psnr = -1
            init_psnr = 0
            # progress_bar = tqdm(range(0, iterations), desc="Optimizing camera " + str(camera_index))
            best_viewpoint_cam = deepcopy(viewpoint_cam)
            scene.reset_pose_scheduler(opt)

            for iteration in range(0, iterations):        
                
                # image = render(viewpoint_cam, gaussians, pipe, background)["render"]
                
                screenspace_points = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, requires_grad=True, device="cuda") + 0
                try:
                    screenspace_points.retain_grad()
                except:
                    pass

                # Set up rasterization configuration
                tanfovx = math.tan(viewpoint_cam.FoVx * 0.5)
                tanfovy = math.tan(viewpoint_cam.FoVy * 0.5)

                raster_settings = GaussianRasterizationSettings(
                    image_height=int(viewpoint_cam.image_height),
                    image_width=int(viewpoint_cam.image_width),
                    tanfovx=tanfovx,
                    tanfovy=tanfovy,
                    bg=background,
                    scale_modifier=1,
                    projmatrix=viewpoint_cam.full_proj_transform,
                    sh_degree=gaussians.active_sh_degree,
                    prefiltered=False,
                    debug=pipe.debug
                )


                viewmatrix=viewpoint_cam.world_view_transform
                means3D = gaussians.get_xyz
                means2D = screenspace_points
                opacity = gaussians.get_opacity

                # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
                # scaling / rotation by the rasterizer.

                cov3D_precomp = torch.Tensor([])
                scales = gaussians.get_scaling
                rotations = gaussians.get_rotation

                # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
                # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.

                shs = gaussians.get_features
                colors_precomp = torch.Tensor([])

                sparsify_ind = 1

                print(means3D.shape)

                viewmatrix_ = viewmatrix.detach().clone()
                means3D_ = means3D[::sparsify_ind].detach().clone()
                means2D_ = means2D[::sparsify_ind].detach().clone()
                shs_ = shs[::sparsify_ind].detach().clone()
                colors_precomp_ = colors_precomp.detach().clone()
                opacity_ = opacity[::sparsify_ind].detach().clone()
                scales_ = scales[::sparsify_ind].detach().clone()
                rotations_ = rotations[::sparsify_ind].detach().clone()
                cov3D_precomp_ = cov3D_precomp.detach().clone()


                viewmatrix_ = torch.tensor(viewmatrix_, dtype=torch.float32, requires_grad=True)
                means3D_ = torch.tensor(means3D_, dtype=torch.float32, requires_grad=False)
                means2D_ = torch.tensor(means2D_, dtype=torch.float32, requires_grad=False)
                shs_ = torch.tensor(shs_, dtype=torch.float32, requires_grad=False)
                colors_precomp_ = torch.tensor(colors_precomp_, dtype=torch.float32, requires_grad=False)
                opacity_ = torch.tensor(opacity_, dtype=torch.float32, requires_grad=False)
                scales_ = torch.tensor(scales_, dtype=torch.float32, requires_grad=False)
                rotations_ = torch.tensor(rotations_, dtype=torch.float32, requires_grad=False)
                cov3D_precomp_ = torch.tensor(cov3D_precomp_, dtype=torch.float32, requires_grad=False)

                torchvision.utils.save_image(rasterize_gaussians(
                    viewmatrix_,
                    means3D_,
                    means2D_,
                    shs_,
                    colors_precomp_,
                    opacity_,
                    scales_, 
                    rotations_,
                    cov3D_precomp_,
                    raster_settings, 
                    )[0], 'img_debug.png')
                

                torch.autograd.gradcheck(rasterize_gaussians, (
                    viewmatrix_,
                    means3D_,
                    means2D_,
                    shs_,
                    colors_precomp_,
                    opacity_,
                    scales_, 
                    rotations_,
                    cov3D_precomp_,
                    raster_settings, 
                    ), eps=1e-5, atol=5e-3)


def render_sets(dataset, iteration, opt, pipe, skip_train : bool, skip_test : bool, pose_adjustment = False, img_tag = ''):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle = False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # if not skip_train:
    #     psnr_train, ssim_train, lpips_train = render_trainset(dataset.model_path, "train", scene.loaded_iter,
    #                                                     scene.getTrainCameras(), gaussians, pipe, background)
    #     with open(os.path.join(dataset.model_path, 'final_rendering_metrics.txt'), 'w') as fw:
    #         print('Train_PSNR_' + format(psnr_train, '.5f') + '_Train_SSIM_' + format(ssim_train, '.5f') + '_Train_LPIPS_' + format(lpips_train, '.5f'))
    #         fw.write('Train_PSNR_' + format(psnr_train, '.5f') + '_Train_SSIM_' + format(ssim_train, '.5f') + '_Train_LPIPS_' + format(lpips_train, '.5f') + '\n')

    if not skip_test:
        scene.save_camera_poses('test', -1, '_init')
        render_adjust_testset(scene, gaussians, pipe, opt, dataset, background, pose_adjustment, 'test', img_tag)
        scene.save_camera_poses('test', -1, '_final')
        scene.combine_adj_camera_poses()

    if dataset.custom_trajectory:
        scene.init_custom_trajectory(dataset, gaussians, load_iteration=iteration)
        custom_name = dataset.custom_trajectory.split('/')[-1].split('.json')[0]
        scene.save_camera_poses('test', -1, custom_name + '_init')
        render_adjust_testset(scene, gaussians, pipe, opt, dataset, background, pose_adjustment, save_tag=custom_name, img_tag = img_tag)
        scene.save_camera_poses('test', -1, custom_name + '_final')

        
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--pose_adjustment", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    render_sets(model.extract(args), args.iteration, pose_estimation_params(0.001, 0.000001, 1, 2000, False), 
               pipeline.extract(args), args.skip_train, args.skip_test, args.pose_adjustment, args.img_tag)

