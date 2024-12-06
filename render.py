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
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
import random
random.seed(42)

from scene import Scene
import os
import subprocess
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_lie as render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, pose_estimation_params
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import EdgeDetection
import numpy as np
from utils.loss_utils import l1_loss, ssim, l2_loss
from copy import deepcopy
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
import pypose as pp

def render_trainset(model_path, name, iteration, views, 
                    gaussians, pipeline, background, cam_extr_file):
    print(model_path)
    folder_name = 'full_pa_trajectory' if cam_extr_file else "train"  
    render_path = os.path.join(model_path, folder_name, "ours_{}".format(iteration), "renders")
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
                                                            + '_' + format(lpips_value, '0.4f') 
                                                            + '_pred.png'))
        torchvision.utils.save_image(gt, os.path.join(render_path, '{0:05d}'.format(idx) + "_gt.png"))
    return np.mean(np.array(psnr_test)), np.mean(np.array(ssim_test)), np.mean(np.array(lpips_test)) 


def render_adjust_testset(scene, gaussians, pipe, opt, dataset, background, 
                          pose_adjustment, save_tag = 'test', img_tag = ''):
    img_tag = '_' + img_tag if img_tag else img_tag
    exit_psnr_parameter = 1e-9
    lambda_dssim = 0
    lambda_l2grad = 0
    iterations  = opt.pose_lr_max_steps
    lpips = LPIPS(normalize=True).cuda()
    scene.cam_adjustment_setup(opt, 'test')

    render_path = os.path.join(dataset.model_path, save_tag, 
                               "ours_{}".format(scene.loaded_iter), "renders")

    makedirs(render_path, exist_ok=True)

    viewpoint_stack = scene.getTestCameras()

    optimized_cameras = {}
    psnrs_orig_test = []
    ssim_orig_test = []
    lpips_orig_test = []

    psnrs_adj_test = []
    ssim_adj_test = []
    lpips_adj_test = []
    edges = EdgeDetection()
    edges = edges.cuda()

    for camera_index in tqdm(range(len(viewpoint_stack))):
        viewpoint_cam = viewpoint_stack[camera_index]
        # print(viewpoint_cam.FoVx, viewpoint_cam.FoVy, viewpoint_cam.image_width, viewpoint_cam.image_height, viewpoint_cam.projection_matrix)
        rendering_orig = render(viewpoint_stack[camera_index], gaussians, pipe, background)["render"]
        rendering_orig = torch.clamp(rendering_orig, 0, 1)
        if pose_adjustment:
            prev_psnr = 1000
            max_psnr = -1
            # max_psnr = 1000
            init_psnr = 0
            progress_bar = tqdm(range(0, iterations), desc="Optimizing camera " + str(camera_index))
            best_viewpoint_cam = deepcopy(viewpoint_cam.world_view_transform)
            scene.reset_pose_scheduler(opt)
            counter = 0
            for iteration in range(0, iterations):        
                
                image = render(viewpoint_cam, gaussians, pipe, background)["render"]
                # image = image/image.max()
                image = torch.clamp(image, 0, 1)
                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l2_loss(image, gt_image)
                loss = (1.0 - lambda_dssim - lambda_l2grad) * Ll1 + lambda_dssim * (1.0 - ssim(image, gt_image))
                psnr_value = psnr(image, gt_image).mean().item()
                # _, image_grad_mag, _, _, image_grad_mag_thresh, _ = edges(image.unsqueeze(0))
                # _, gt_image_grad_mag, _, _, gt_image_grad_mag_thresh, _ = edges(gt_image.unsqueeze(0))

                # loss += l2_loss(image_grad_mag, gt_image_grad_mag)*lambda_l2grad

                # loss += l2_loss(image_grad_mag_thresh, gt_image_grad_mag_thresh)*lambda_l2grad

                # print(image.shape, gt_image.shape)
                # psnr_value = deepcopy(loss.item())
                loss.backward()

                print(viewpoint_cam.world_view_transform, 'grad')
                print(viewpoint_cam.world_view_transform_lie, 'se3lie')

                # print(pp.SE3(pp.se3(-viewpoint_cam.world_view_transform.grad.squeeze()[:-1]*1).Exp()).mul(viewpoint_cam.world_view_transform))

                # viewpoint_cam.world_view_transform_lie.grad[0][3:] *= 0.01

                if psnr_value >= max_psnr:
                # if psnr_value <= max_psnr:
                    max_psnr = psnr_value 
                    best_viewpoint_cam = deepcopy(viewpoint_cam.world_view_transform)
                # print(psnr_value, prev_psnr)
                if abs(psnr_value - prev_psnr) <= exit_psnr_parameter:
                    counter += 1
                    if counter == 3:
                        print('Converged. PSNR optimized as: ' \
                        + format(init_psnr, '.7f') + ' -> ' + format(max_psnr, '.7f'))
                        counter = 0
                        progress_bar.close()
                        break


                progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}", 'PSNR' : f"{psnr_value:.{7}f}"})
                progress_bar.update()
                prev_psnr = psnr_value
                
                init_psnr = psnr_value if iteration == 0 else init_psnr

                with torch.no_grad():
                    scene.update_learning_rate_pose(iteration)
                    scene.pose_optimizer.step()
                    scene.pose_optimizer.zero_grad(set_to_none = True)
                    # viewpoint_cam.world_view_transform = torch.nn.Parameter(viewpoint_cam.world_view_transform_lie[0].data.detach().requires_grad_())

            progress_bar.close()
            viewpoint_stack[camera_index].world_view_transform = best_viewpoint_cam
            rendering_adj = render(viewpoint_stack[camera_index], gaussians, pipe, background)["render"]
            rendering_adj = torch.clamp(rendering_adj, 0, 1)
            gt = viewpoint_stack[camera_index].original_image[0:3, :, :]
            best_psnr = psnr(rendering_adj, gt).mean().item()
            best_ssim = ssim(rendering_adj, gt).mean().item()
            best_lpips = lpips(rendering_adj.unsqueeze(0), gt.unsqueeze(0)).mean().item()
            psnrs_adj_test.append(best_psnr)
            ssim_adj_test.append(best_ssim)
            lpips_adj_test.append(best_lpips)
            torchvision.utils.save_image(rendering_adj, os.path.join(render_path, 
                                                        '{0:05d}'.format(camera_index) 
                                                        + '_' + format(best_psnr, '0.4f')
                                                        + '_' + format(best_ssim, '0.4f')
                                                        + '_' + format(best_lpips, '0.4f') 
                                                        + '_adj' + img_tag + '.png'))
            
        gt = viewpoint_stack[camera_index].original_image[0:3, :, :]
        orig_psnr = psnr(rendering_orig, gt).mean().item()
        orig_ssim = ssim(rendering_orig, gt).mean().item()
        orig_lpips = lpips(rendering_orig.unsqueeze(0), gt.unsqueeze(0)).mean().item()
        psnrs_orig_test.append(orig_psnr)
        ssim_orig_test.append(orig_ssim)
        lpips_orig_test.append(orig_lpips)
        torchvision.utils.save_image(gt, os.path.join(render_path, 
                                    '{0:05d}'.format(camera_index) + "_gt.png"))
        torchvision.utils.save_image(rendering_orig, os.path.join(render_path, 
                                                    '{0:05d}'.format(camera_index)                                                                        
                                                    + '_' + format(orig_psnr, '0.4f')
                                                    + '_' + format(orig_ssim, '0.4f')
                                                    + '_' + format(orig_lpips, '0.4f') 
                                                    + '_orig' + img_tag + '.png'))
        
        viewpoint_stack[camera_index].world_view_transform = best_viewpoint_cam
        
        del rendering_orig
        del rendering_adj
        del image
        del gt

    with open(os.path.join(dataset.model_path, 'final_rendering_metrics.txt'), 'a+') as fw:
        print('OrigTest_PSNR_' + format(np.mean(psnrs_orig_test), '.5f') + 
              '_OrigTest_SSIM_' + format(np.mean(ssim_orig_test), '.5f') + 
              '_OrigTest_LPIPS_' + format(np.mean(lpips_orig_test), '.5f'))
        fw.write('OrigTest_PSNR_' + format(np.mean(psnrs_orig_test), '.5f') + 
                 '_OrigTest_SSIM_' + format(np.mean(ssim_orig_test), '.5f') + 
                 '_OrigTest_LPIPS_' + format(np.mean(lpips_orig_test), '.5f') + '\n')
        if pose_adjustment:
            print('PATest_PSNR ' + format(np.mean(psnrs_adj_test), '.5f') + 
                  ' PATest_SSIM ' + format(np.mean(ssim_adj_test), '.5f') + 
                  ' PATest_LPIPS ' + format(np.mean(lpips_adj_test), '.5f'))
            fw.write('PATest_PSNR_' + format(np.mean(psnrs_adj_test), '.5f') + 
                     '_PATest_SSIM_' + format(np.mean(ssim_adj_test), '.5f') + 
                     '_PATest_LPIPS_' + format(np.mean(lpips_adj_test), '.5f') + '\n')


def render_sets(dataset, iteration, opt, pipe, skip_train : bool, 
                skip_test : bool, pose_adjustment = False, img_tag = ''):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle = False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if not skip_train:
        psnr_train, ssim_train, lpips_train = render_trainset(dataset.model_path, "train", scene.loaded_iter,
                        scene.getTrainCameras(), gaussians, pipe, background, dataset.cameras_extrinsic_file)
        with open(os.path.join(dataset.model_path, 'final_rendering_metrics.txt'), 'w') as fw:
            print('Train_PSNR_' + format(psnr_train, '.5f') + '_Train_SSIM_' +
                   format(ssim_train, '.5f') + '_Train_LPIPS_' + format(lpips_train, '.5f'))
            fw.write('Train_PSNR_' + format(psnr_train, '.5f') + '_Train_SSIM_' +
                      format(ssim_train, '.5f') + '_Train_LPIPS_' + format(lpips_train, '.5f') + '\n')

    if not skip_test:
        scene.save_camera_poses('test', -1, '_init')
        render_adjust_testset(scene, gaussians, pipe, opt, dataset, background, 
                              pose_adjustment, 'test', img_tag)
        scene.save_camera_poses('test', -1, '_final')
        scene.combine_adj_camera_poses()

    if dataset.custom_trajectory:
        scene.init_custom_trajectory(dataset, gaussians, load_iteration=iteration)
        custom_name = dataset.custom_trajectory.split('/')[-1].split('.json')[0]
        # scene.save_camera_poses('test', -1, custom_name + '_init')
        render_adjust_testset(scene, gaussians, pipe, opt, dataset, background, 
                              pose_adjustment, save_tag=custom_name, img_tag = img_tag)
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
    render_sets(model.extract(args), args.iteration, 
                pose_estimation_params(0.01, 1e-5, 1, 2000, False), 
                pipeline.extract(args), args.skip_train, args.skip_test, 
                args.pose_adjustment, args.img_tag)

