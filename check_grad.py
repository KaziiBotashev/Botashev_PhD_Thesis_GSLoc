import numpy as np
np.random.seed(42)
import os
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
import random
random.seed(42)
from random import randint
from utils.loss_utils import l1_loss, ssim
# from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, pose_estimation_params
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
# from render import render_sets
from diff_gaussian_rasterization import GaussianRasterizationSettings, rasterize_gaussians, GaussianRasterizer
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import math


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        prefiltered=False,
        debug=pipe.debug
    )


    viewmatrix=viewpoint_camera.world_view_transform
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.

    cov3D_precomp = torch.Tensor([])
    scales = pc.get_scaling
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.

    shs = pc.get_features
    colors_precomp = torch.Tensor([])

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rasterize_gaussians(
                viewmatrix,
                means3D,
                means2D,
                shs,
                colors_precomp,
                opacity,
                scales, 
                rotations,
                cov3D_precomp,
                raster_settings, 
            )


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, agjust_poses, debug_from):
    first_iter = 0
    #lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').cuda()
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    scene.cam_adjustment_setup(opt)
    # return
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # viewpoint_stack = None
    cam_indices = None
    viewpoint_stack = scene.getTrainCameras()
    # print(viewpoint_stack)
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    if opt.adaptive_cam_choice:
        cam_indices = {cam_id:1 for cam_id in np.arange(len(viewpoint_stack)).tolist()}
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussian_lr = gaussians.update_learning_rate(iteration)
        pose_lr = scene.update_learning_rate_pose(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not opt.adaptive_cam_choice:
            if not cam_indices:
                cam_indices = np.arange(len(viewpoint_stack)).tolist()

            current_cam_ind = cam_indices.pop(randint(0, len(cam_indices)-1))
        else:
            current_cam_ind = random.choices(list(cam_indices.keys()), weights=cam_indices.values())[0]
            
        viewpoint_cam = viewpoint_stack[current_cam_ind]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # render_pkg = render(viewpoint_cam, gaussians, pipe, background)


            # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
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

        # print(viewmatrix)
        # # viewmatrix.requires_grad = True
        # means3D.requires_grad = False
        # means2D.requires_grad = False
        # shs.requires_grad = False
        # colors_precomp.requires_grad = False
        # opacity.requires_grad = False
        # scales.requires_grad = False
        # rotations.requires_grad = False
        # cov3D_precomp.requires_grad = False

        # print(viewmatrix, 'viewmatrix')
        # print(means3D, 'means3D')
        # print(means2D, 'means2D')
        # print(shs, 'shs')
        # print(colors_precomp, 'colors_precomp')
        # print(opacity, 'opacity')
        # print(scales, 'scales')
        # print(rotations, 'rotations')
        # print(cov3D_precomp, 'cov3D_precomp')
        # print(raster_settings, 'raster_settings')

        # print(means3D.shape, means2D.shape, opacity.shape, scales.shape, rotations.shape, shs.shape)
        sparsify_ind = 1

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
        
        # print(torch.autograd.gradcheck(rasterize_gaussians, (viewmatrix, means3D, means2D, shs, colors_precomp,
        #                                                 opacity, scales, rotations, cov3D_precomp, raster_settings), eps=1e-5, atol=5e-3))

        


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6008)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000, 40_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000, 40_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000, 40_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--pose_adjustment", action="store_true")
    parser.add_argument('--exp_tag', type=str, default="")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    out_pa_tag = 'pa-lr-' + str(args.pose_lr_init) if args.pose_adjustment else 'van'
    out_pa_tag = out_pa_tag + args.exp_tag
    path_sep_start = 1 if args.source_path.split('/')[-1] == '' else 0
    pose_method_tag = args.source_path.split('/')[-1 - path_sep_start]
    scene_tag = args.source_path.split('/')[-2 - path_sep_start]
    dataset_tag = args.source_path.split('/')[-3 - path_sep_start]
    args.model_path = os.path.join('./output', dataset_tag + '_' + scene_tag + '_' + pose_method_tag +  '_' + out_pa_tag)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, 
             args.checkpoint_iterations, args.start_checkpoint, args.pose_adjustment, args.debug_from)
    # render_sets(lp.extract(args), -1, pose_estimation_params(0.001, 0.000001, 1, 2000), 
    #            pp.extract(args), skip_train=False, skip_test=False, pose_adjustment=True)

