import numpy as np
np.random.seed(42)
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
import random
random.seed(42)
import os
from gaussian_renderer import render_lie, render_qtvec
from arguments import  pose_estimation_params
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim, l2_loss
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from utils.camera_utils import loadCam
from collections import namedtuple
from scene.gaussian_model import GaussianModel
import pickle
from scene.dataset_readers import CameraInfo
from PIL import Image
from arguments import  pose_estimation_params
from utils.general_utils import get_expon_lr_func
import cv2
from glob import glob
import copy
import json
import fire
import kernels
import torch.nn.functional as torch_F
import tqdm
import shutil

def interp_schedule(x, schedule, left=0, right=1):
    # linear interprete between a list of schedule values
    assert left <= x and right >= x
    if isinstance(schedule, torch.Tensor):
        schedule = schedule.cpu().detach().numpy()
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    xs = np.linspace(left, right, len(schedule))
    return np.interp(x, xs, schedule)

def get_number_of_max_iou_iters(num, max_iters=64):
    return num if num < max_iters else max_iters

def get_render_func(camera_type):
    if camera_type == 'qtvec':
        render = render_qtvec
    elif camera_type == 'lietorch':
        render = render_lie
    return render

def get_loss_func(loss_type):
    if loss_type == 'l1':
        loss_func = l1_loss
    elif loss_type == 'l2':
        loss_func = l2_loss
    return loss_func

def get_optimizer_func(optimizer_type):
    if optimizer_type == 'adam':
        optimizer_func = torch.optim.Adam
    return optimizer_func

def image_torch_to_np(image_init):
    norm_image = cv2.normalize(image_init.squeeze().detach().cpu().numpy().transpose(1,2,0), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    return norm_image

class Solver:
    def __init__(self, camera, optimizer_func, camera_pose_estimation_params):
        
        self.pose_optimizer = optimizer_func([{'params': camera.world_view_transform_, 
                                                 'lr': camera_pose_estimation_params.pose_lr_init, "name": "pose"}], 
                                               lr = camera_pose_estimation_params.pose_lr_init)
        self.pose_scheduler_args = get_expon_lr_func(lr_init=camera_pose_estimation_params.pose_lr_init,
                            lr_final=camera_pose_estimation_params.pose_lr_final,
                            lr_delay_steps=camera_pose_estimation_params.pose_lr_delay_steps, 
                            lr_delay_mult=camera_pose_estimation_params.pose_lr_delay_mult,
                            max_steps=camera_pose_estimation_params.pose_lr_max_steps)
        
    def update_learning_rate_pose(self, iteration, factor = 1):
        ''' Learning rate scheduling per step '''
        for param_group in self.pose_optimizer.param_groups:
            if param_group["name"] == "pose":
                if type(self.pose_scheduler_args) == list:
                    lr = self.pose_scheduler_args[iteration] * factor
                else:
                    lr = self.pose_scheduler_args(iteration) * factor
                param_group['lr'] = lr
                return lr


def solve_vanilla(cam_info_init, task_info, gaussians, solving_args = None):
    args = namedtuple('args', ['resolution', 'data_device'])
    args = args(task_info['init_render_resolution'], 'cuda')
    pipe = namedtuple('pipe', ['convert_SHs_python', 'compute_cov3D_python', 'debug'])
    pipe = pipe(False, False, False)
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    camera_type = task_info['camera_type']
    loss_type = task_info['loss_type']
    optimizer_type = task_info['optimizer_type']

    iterations = task_info['iterations']
    exit_psnr_parameter = task_info['exit_psnr_parameter']

    pose_lr_init = task_info['pose_lr_init']
    pose_lr_final = task_info['pose_lr_final']
    pose_lr_delay_steps = task_info['pose_lr_delay_steps']
    pose_lr_delay_mult = task_info['pose_lr_delay_mult']

    render_func = get_render_func(camera_type)
    loss_func = get_loss_func(loss_type)
    optimizer_func = get_optimizer_func(optimizer_type)

    # camera_gt = loadCam(args=args, id = 0, cam_info=cam_info_gt, resolution_scale=1, camera_type=camera_type)
    camera_init = loadCam(args=args, id = 0, cam_info=cam_info_init, resolution_scale=1, camera_type=camera_type)

    pose_solver = Solver(camera_init, optimizer_func, pose_estimation_params(pose_lr_init, pose_lr_final, pose_lr_delay_steps,
                                                            pose_lr_delay_mult, iterations, False))

    prev_psnr = 1000
    max_psnr = -1
    init_psnr = 0
    current_try = 0
    lpips = LPIPS(normalize=True).cuda()
    init_qtvec = camera_init.world_view_transform_
    rendering_init = torch.clamp(render_func(camera_init, gaussians, pipe, background)["render"], 0, 1)
    gt_image = camera_init.original_image.cuda()

    # progress_bar = tqdm(range(0, iterations), desc="Optimizing camera " + str(camera_gt.image_name))
    if camera_type == 'lietorch':
        best_viewpoint_cam = torch.nn.Parameter(camera_init.world_view_transform_[0].data.detach().requires_grad_())
    else:
        best_viewpoint_cam = camera_init.world_view_transform_
    converged = False
    counter = 0
    for iteration in range(0, iterations):        
        image = render_func(camera_init, gaussians, pipe, background)["render"]
        image = torch.clamp(image, 0, 1)
        L1 = loss_func(image, gt_image)
        loss_value = L1
        psnr_value = psnr(image, gt_image).mean().item()
        loss_value.backward()

        if iteration == 0:
            loss_init = loss_value.item()

        if psnr_value >= max_psnr:
            max_psnr = psnr_value 
            if camera_type == 'lietorch':
                best_viewpoint_cam = torch.nn.Parameter(camera_init.world_view_transform_[0].data.detach().requires_grad_())
            else:
                best_viewpoint_cam = camera_init.world_view_transform_

        if abs(psnr_value - prev_psnr) <= exit_psnr_parameter:
            counter += 1
            if counter == 3:
                converged = True
                # print('Converged. PSNR optimized as: ' \
                # + format(init_psnr, '.7f') + ' -> ' + format(max_psnr, '.7f'))
                counter = 0
                # progress_bar.close()
                break

        # progress_bar.set_postfix({"Loss": f"{loss_value.item():.{7}f}", 'PSNR' : f"{psnr_value:.{7}f}"})
        # progress_bar.update()
        prev_psnr = psnr_value
        
        init_psnr = psnr_value if iteration == 0 else init_psnr

        with torch.no_grad():
            if camera_type == 'lietorch':
                camera_init.world_view_transform_.grad = camera_init.world_view_transform.grad[:-1].unsqueeze(0)
            pose_solver.update_learning_rate_pose(iteration)
            pose_solver.pose_optimizer.step()
            pose_solver.pose_optimizer.zero_grad(set_to_none = True)
            if camera_type == 'lietorch':
                camera_init.world_view_transform = torch.nn.Parameter(camera_init.world_view_transform_[0].data.detach().requires_grad_())
    # progress_bar.close()

    if camera_type == 'lietorch':
        camera_init.world_view_transform  = best_viewpoint_cam
    else:
        camera_init.world_view_transform_ = best_viewpoint_cam 

    rendering_result = torch.clamp(render_func(camera_init, gaussians, pipe, background)['render'], 0, 1)
    result_psnr = psnr(rendering_result, gt_image).mean().item()
    result_ssim = ssim(rendering_result, gt_image).mean().item()
    result_lpips = lpips(rendering_result.unsqueeze(0), gt_image.unsqueeze(0)).mean().item()

    init_psnr = psnr(rendering_init, gt_image).mean().item()
    init_ssim = ssim(rendering_init, gt_image).mean().item()
    init_lpips = lpips(rendering_init.unsqueeze(0), gt_image.unsqueeze(0)).mean().item()

    success = 1 if result_psnr > 25 else 0

    return {'qtvec_init':init_qtvec.detach().cpu().numpy(),
            'psnr_init':init_psnr,
            'ssim_init':init_ssim,
            'lpips_init':init_lpips,
            'loss_init':loss_init,
            'image_init':image_torch_to_np(rendering_init),
            'image_gt':image_torch_to_np(gt_image),

            'qtvec_result':best_viewpoint_cam.detach().cpu().numpy(),
            'psnr_result':result_psnr,
            'ssim_result':result_ssim,
            'lpips_result':result_lpips,
            'loss_result':loss_value.item(),
            'image_result':image_torch_to_np(rendering_result),
            'iterations_to_result':iteration,
            'converged':converged,
            'num_tries':current_try,
            'success':success,
            }

def blur_image(gt_image, iteration, iterations, 
               max_scale, blur_2d_c2f_kernel_size, 
               blur_2d_c2f_schedule,
               init_render_resolution):
    H = gt_image.shape[1]
    W = gt_image.shape[2]
    c2f_alternate_2D_mode = False
    device = torch.device('cuda')

    # blur_2d_c2f_schedule = [0.05, 0.025, 0.0125, 0.00625, 0.00625, 0.0, 0.0, 0.0, 0.0, 0.0]
    # blur_2d_c2f_schedule = [0.025, 0.0125, 0.00625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # blur_2d_c2f_kernel_size = 201
    # max_scale = 1
    
    blur_2d_c2f_kernel_size_used = int(blur_2d_c2f_kernel_size/init_render_resolution) + 1

    if c2f_alternate_2D_mode == "sample":
        scales = [0.0, 0.25, 0.5, 0.75, 1.0]
    else:
        scales = [0.0, max_scale]
    # get kernels
    kernels_dict = dict() # dictionary from scale to kernel
    for sc in scales:
        blur_param = interp_schedule(float(iteration/iterations), blur_2d_c2f_schedule)
        blur_param = torch.tensor(blur_param, device=device)
        blur_param *= sc
        # get kernel

        kernel_width = blur_param * (W + H)/2
        kernel_1d = kernels.get_gaussian_kernel(kernel_width, blur_2d_c2f_kernel_size_used)

        kernel_1d = kernel_1d.to(device=device,dtype=torch.float32)

        kernel_1d = kernel_1d.expand(1,1,-1)

        kernels_dict[sc] = (kernel_1d, kernel_width)

    # generte blurred GT images
    blurred_gt_cached_images = dict()

    for sc, k in kernels_dict.items():
        kernel_1d, kernel_width = k
        # skip kernel if kernel_width too small
        if kernel_width < 0.01:
            images = gt_image
        else:
            # perform 2D seperated convolution
            images = gt_image
            kernel_size = kernel_1d.shape[-1]
            pad_size= (kernel_size //2, kernel_size //2)
            images = torch_F.pad(images, pad_size, mode="replicate")
            images = torch_F.conv1d(images, kernel_1d.expand(H,1,-1), bias=None, stride=1, padding=0, dilation=1, groups=H)
            images = images.permute(0,2,1)
            images = torch_F.pad(images, pad_size, mode="replicate")
            images = torch_F.conv1d(images, kernel_1d.expand(W,1,-1), bias=None, stride=1, padding=0, dilation=1, groups=W)
            images = images.permute(0,2,1).reshape(1, 3, H, W).contiguous()
        blurred_gt_cached_images[sc] = images
    return blurred_gt_cached_images


def solve_coarse(cam_info_init, task_info, gaussians, solving_args = None):
    args = namedtuple('args', ['resolution', 'data_device'])
    args = args(task_info['init_render_resolution'], 'cuda')
    pipe = namedtuple('pipe', ['convert_SHs_python', 'compute_cov3D_python', 'debug'])
    pipe = pipe(False, False, False)
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    camera_type = task_info['camera_type']
    loss_type = task_info['loss_type']
    optimizer_type = task_info['optimizer_type']

    iterations = task_info['iterations']
    exit_psnr_parameter = task_info['exit_psnr_parameter']

    pose_lr_init = task_info['pose_lr_init']
    pose_lr_final = task_info['pose_lr_final']
    pose_lr_delay_steps = task_info['pose_lr_delay_steps']
    pose_lr_delay_mult = task_info['pose_lr_delay_mult']

    render_func = get_render_func(camera_type)
    loss_func = get_loss_func(loss_type)
    optimizer_func = get_optimizer_func(optimizer_type)

    # camera_gt = loadCam(args=args, id = 0, cam_info=cam_info_gt, resolution_scale=1, camera_type=camera_type)
    camera_init = loadCam(args=args, id = 0, cam_info=cam_info_init, resolution_scale=1, camera_type=camera_type)

    pose_solver = Solver(camera_init, optimizer_func, pose_estimation_params(pose_lr_init, pose_lr_final, pose_lr_delay_steps,
                                                        pose_lr_delay_mult, iterations, False))
    
    num_tries = solving_args['num_tries']
    max_scale = solving_args['max_scale']
    blur_2d_c2f_kernel_size = solving_args['blur_2d_c2f_kernel_size']
    blur_2d_c2f_schedule = solving_args['blur_2d_c2f_schedule']
    
    current_try = 0
    success = 0
    best_psnr = -1

    while current_try < num_tries and success == 0:
        # if current_try > 1:
        #     max_scale += 1
        prev_psnr = 1000
        max_psnr = -1
        init_psnr = 0

        lpips = LPIPS(normalize=True).cuda()
        init_qtvec = camera_init.world_view_transform_
        if current_try == 0:
            rendering_init = torch.clamp(render_func(camera_init, gaussians, pipe, background)["render"], 0, 1)
        gt_image = camera_init.original_image.cuda()

        # progress_bar = tqdm(range(0, iterations), desc="Optimizing camera ")
        if camera_type == 'lietorch':
            best_viewpoint_cam = torch.nn.Parameter(camera_init.world_view_transform_[0].data.detach().requires_grad_())
        else:
            best_viewpoint_cam = camera_init.world_view_transform_
        converged = False
        counter = 0
        for iteration in range(0, iterations):        
            image = render_func(camera_init, gaussians, pipe, background)["render"]
            image = torch.clamp(image, 0, 1)

            gt_image_blur_dict = blur_image(gt_image, iteration, 
                                            iterations, max_scale=max_scale, 
                                            blur_2d_c2f_kernel_size = blur_2d_c2f_kernel_size,
                                            blur_2d_c2f_schedule = blur_2d_c2f_schedule,
                                            init_render_resolution = task_info['init_render_resolution'])
            image_blur_dict = blur_image(image, iteration, 
                                         iterations, max_scale=max_scale, 
                                        blur_2d_c2f_kernel_size = blur_2d_c2f_kernel_size,
                                        blur_2d_c2f_schedule = blur_2d_c2f_schedule, 
                                        init_render_resolution = task_info['init_render_resolution'])

            L1 = loss_func(image_blur_dict[max_scale], gt_image_blur_dict[max_scale])
            loss_value = L1
            psnr_value = psnr(image, gt_image).mean().item()
            loss_value.backward()

            if iteration == 0 and current_try == 0:
                loss_init = loss_value.item()

            if psnr_value >= max_psnr:
                max_psnr = psnr_value 
                if camera_type == 'lietorch':
                    best_viewpoint_cam = torch.nn.Parameter(camera_init.world_view_transform_[0].data.detach().requires_grad_())
                else:
                    best_viewpoint_cam = camera_init.world_view_transform_

            if psnr_value >= best_psnr:
                best_psnr = psnr_value
                if camera_type == 'lietorch':
                    best_try_camera = torch.nn.Parameter(camera_init.world_view_transform_[0].data.detach().requires_grad_())
                else:
                    best_try_camera = camera_init.world_view_transform_

            if abs(psnr_value - prev_psnr) <= exit_psnr_parameter:
                counter += 1
                if counter == 3:
                    converged = True
                    print('Converged. PSNR optimized as: ' \
                    + format(init_psnr, '.7f') + ' -> ' + format(max_psnr, '.7f'))
                    counter = 0
                    # progress_bar.close()
                    break
            
            # progress_bar.set_postfix({"Loss": f"{loss_value.item():.{7}f}", 'PSNR' : f"{psnr_value:.{7}f}"})
            # progress_bar.update()
            prev_psnr = psnr_value
            
            init_psnr = psnr_value if iteration == 0 else init_psnr

            with torch.no_grad():
                if camera_type == 'lietorch':
                    camera_init.world_view_transform_.grad = camera_init.world_view_transform.grad[:-1].unsqueeze(0)
                pose_solver.update_learning_rate_pose(iteration)
                pose_solver.pose_optimizer.step()
                pose_solver.pose_optimizer.zero_grad(set_to_none = True)
                if camera_type == 'lietorch':
                    camera_init.world_view_transform = torch.nn.Parameter(camera_init.world_view_transform_[0].data.detach().requires_grad_())
        # progress_bar.close()

        if camera_type == 'lietorch':
            camera_init.world_view_transform  = best_viewpoint_cam
        else:
            camera_init.world_view_transform_ = best_viewpoint_cam 

        rendering_result = torch.clamp(render_func(camera_init, gaussians, pipe, background)['render'], 0, 1)
        result_psnr = psnr(rendering_result, gt_image).mean().item()
        result_ssim = ssim(rendering_result, gt_image).mean().item()
        result_lpips = lpips(rendering_result.unsqueeze(0), gt_image.unsqueeze(0)).mean().item()

        if current_try == 0:
            init_psnr = psnr(rendering_init, gt_image).mean().item()
            init_ssim = ssim(rendering_init, gt_image).mean().item()
            init_lpips = lpips(rendering_init.unsqueeze(0), gt_image.unsqueeze(0)).mean().item()

        success = 1 if result_psnr > 25 else 0
        current_try += 1
    
    if camera_type == 'lietorch':
        camera_init.world_view_transform  = best_try_camera
    else:
        camera_init.world_view_transform_ = best_try_camera 

    rendering_result = torch.clamp(render_func(camera_init, gaussians, pipe, background)['render'], 0, 1)
    result_psnr = psnr(rendering_result, gt_image).mean().item()
    result_ssim = ssim(rendering_result, gt_image).mean().item()
    result_lpips = lpips(rendering_result.unsqueeze(0), gt_image.unsqueeze(0)).mean().item()

    success = 1 if result_psnr > 25 else 0
    
    return {'qtvec_init':init_qtvec.detach().cpu().numpy(),
            'psnr_init':init_psnr,
            'ssim_init':init_ssim,
            'lpips_init':init_lpips,
            'loss_init':loss_init,
            'image_init':image_torch_to_np(rendering_init),
            'image_gt':image_torch_to_np(gt_image),

            'qtvec_result':best_viewpoint_cam.detach().cpu().numpy(),
            'psnr_result':result_psnr,
            'ssim_result':result_ssim,
            'lpips_result':result_lpips,
            'loss_result':loss_value.item(),
            'image_result':image_torch_to_np(rendering_result),
            'iterations_to_result':iteration,
            'converged':converged,
            'num_tries':current_try,
            'success':success,
            }


def get_solving_func(solving_method):
    if solving_method == 'vanilla':
        solving_func = solve_vanilla
    if solving_method == 'coarse':
        solving_func = solve_coarse
    return solving_func

def solve_pose_estimation(  room_name,
                            start_frame,
                            end_frame,
                            camera_type,
                            solving_method,
                            solving_method_args_={ 
                                                    'max_scale':2,
                                                    'num_tries':5,
                                                    'blur_2d_c2f_kernel_size':400,
                                                    'blur_2d_c2f_schedule':[0.05, 0.025, 0.0125, 0.00625, 0.00625, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                },
                            loss_type='l1',
                            optimizer_type='adam',
                            init_render_resolution=2,
                            iterations=2000,
                            exit_psnr_parameter=5e-5,
                            pose_lr_init=0.01,
                            pose_lr_final=1e-5,
                            pose_lr_delay_steps=0,
                            pose_lr_delay_mult=0,
                            results_root='/mnt/sdb1/home/kbotashev/iros_paper/replica_dataset_1/results_task_rebase_vladnet/solutions'
 ):
    solving_method_args = None if solving_method == 'vanilla' else solving_method_args_
    os.makedirs(results_root, exist_ok=True)
    json_task_paths_template = '/mnt/sdb1/home/kbotashev/iros_paper/replica_dataset_1/results_task_rebase_vladnet/tasks/@room_name@/@frame_num@/*/*.json'
    experiments_base_dir = json_task_paths_template.split('/results_task_rebase_vladnet/')[0]

    ply_path = os.path.join(experiments_base_dir, 
                            'output', room_name, 
                            'point_cloud/iteration_40000/point_cloud.ply')

    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(ply_path)

    frame_nums = np.arange(start_frame, end_frame)
    json_task_paths = []
    for frame_num in frame_nums:
        json_task_paths += glob(json_task_paths_template.replace('@room_name@', room_name).replace('@frame_num@', str(frame_num).zfill(5)))
    json_task_paths = np.array(json_task_paths)

    sorted_json_task_paths = json_task_paths[np.argsort([json_task_path.split('/')[-1].split('.json')[0] for json_task_path in json_task_paths])]

    for json_id, json_task_path in enumerate(sorted_json_task_paths):

        img_name = json_task_path.split('/')[-1].split('_')[-1].split('.json')[0]

        
        with open(json_task_path, 'r') as f:
            task_info = json.load(f)

        task_info['camera_type'] = camera_type
        task_info['loss_type'] = loss_type
        task_info['optimizer_type'] = optimizer_type

        task_info['iterations'] = iterations
        task_info['exit_psnr_parameter'] = exit_psnr_parameter

        task_info['pose_lr_init'] = pose_lr_init
        task_info['pose_lr_final'] = pose_lr_final
        task_info['pose_lr_delay_steps'] = pose_lr_delay_steps
        task_info['pose_lr_delay_mult'] = pose_lr_delay_mult
        task_info['init_render_resolution'] = init_render_resolution
        task_info['solving_method_args'] = solving_method_args
        task_info['solving_method'] = solving_method

        print("Solving: " + camera_type + ' ' + solving_method + ' ' + str(json_id) + '/' +
               str(len(sorted_json_task_paths)) + ' ' + json_task_path)

        iou_bin = task_info['iou_bin']
        index = task_info['init_id']
        solving_method = task_info['solving_method']
        solving_args = task_info['solving_method_args']
        solve_func = get_solving_func(solving_method=solving_method)

        solving_method_args_str = str(solving_args).replace(', ', '-').replace('[', '').replace(']', '').replace("'", '').replace(": ", "--").replace('}', '').replace('{', '')
        experiment_result_dir = os.path.join(results_root, 
                                        str(task_info['camera_type']) + '_' + 
                                        str(task_info['loss_type']) + '_' + 
                                        str(task_info['optimizer_type']) + '_' + 
                                        str(task_info['init_render_resolution']) + '_' + 
                                        str(task_info['iterations']) + '_' + 
                                        str(task_info['pose_lr_init']) + '_' + 
                                        str(task_info['pose_lr_final']) + '_' + 
                                        str(task_info['pose_lr_delay_steps']) + '_' + 
                                        str(task_info['pose_lr_delay_mult']) + '_' + 
                                        str(task_info['exit_psnr_parameter']) + '_' + 
                                        str(task_info['solving_method']) + '_' + 
                                        solving_method_args_str)
        
        experiment_result_dir = os.path.join(experiment_result_dir, task_info['room_name'], img_name)
        experiment_result_dir = os.path.join(experiment_result_dir, ('%.2f'%iou_bin).replace('.', ''))

        os.makedirs(experiment_result_dir, exist_ok=True)

        pickle_outs_exist = [os.path.exists(os.path.join(experiment_result_dir, 
                        json_task_path.split('/')[-1].replace('.json', '_' + str(i) + '.pickle'))) for i in np.arange(2)]
        
        if np.sum(pickle_outs_exist) > 0:
            print('ALREADY EXISTS')
            continue

        # pickle_outs_prev_all = np.array([os.path.join(experiment_result_dir, 
        #         json_task_path.split('/')[-1].replace('.json', '_' + str(i) + '.pickle')).replace('/results/', \
        #                     '/results_1/') for i in np.arange(2)])
        
        # pickle_outs_exist_prev_all = np.array([os.path.exists(pickle_outs_prev_path) for pickle_outs_prev_path in pickle_outs_prev_all])
        
        # if np.sum(pickle_outs_exist_prev_all) > 0:
        #     existing_path = pickle_outs_prev_all[pickle_outs_exist_prev_all].item()
        #     existing_path_result = int(existing_path.split('/')[-1].split('_')[-1].split('.pickle')[0])
        #     if solving_method != 'vanilla' and existing_path_result == 0:
        #         print('COARSE RESULT EXISTS BUT IT IS 0 SO WE RECOMPUTE IT')
        #     else:
        #         shutil.copy(existing_path, experiment_result_dir)
        #         print(existing_path)
        #         print('COPIED FROM RESULTS_1')
        #         continue


        # pickle_outs_prev_van = np.array([os.path.join(experiment_result_dir, 
        #         json_task_path.split('/')[-1].replace('.json', '_' + str(i) + '.pickle')).replace('/results/', \
        #                     '/results_0/').replace('_5e-05_vanilla', '_0.0001_vanilla') for i in np.arange(2)])
        
        # pickle_outs_exist_prev_van = np.array([os.path.exists(pickle_outs_prev_path) for pickle_outs_prev_path in pickle_outs_prev_van])
        
        # if np.sum(pickle_outs_exist_prev_van) > 0:
        #     existing_path_van = pickle_outs_prev_van[pickle_outs_exist_prev_van][0]
        #     shutil.copy(existing_path_van, experiment_result_dir)
        #     print(existing_path_van)
        #     print('COPIED FROM RESULTS_0')
        #     continue


        with open(task_info['pickle_path'], 'rb') as handle:
            frames_info = pickle.load(handle)

        FovY = frames_info['frames']['FovY']
        FovX = frames_info['frames']['FovX']
        width = frames_info['frames']['width']
        height = frames_info['frames']['height']
        cx = frames_info['frames']['cx']
        cy = frames_info['frames']['cy']

        image_gt = Image.fromarray(frames_info['frames']['image'])
        image_path = frames_info['frames']['img_path']
        image_name = frames_info['img_name']
        R_gt = frames_info['frames']['R']
        t_gt = frames_info['frames']['T']

        # cam_info_gt = CameraInfo(uid=0, R=R_gt, T=t_gt, FovY=FovY, FovX=FovX, image=image_gt,
        #                          image_path=image_path, image_name=image_name, width=width,
        #                          height=height, qvec = None, cx = cx, cy = cy)

        image_init = frames_info['frames']['init_frames_iou_bins'][iou_bin][index]['init_image']
        R_init = frames_info['frames']['init_frames_iou_bins'][iou_bin][index]['R_init']
        t_init = frames_info['frames']['init_frames_iou_bins'][iou_bin][index]['T_init']

        cam_info_init = CameraInfo(uid=0, R=R_init, T=t_init, FovY=FovY, FovX=FovX, image=image_gt,
                                image_path=image_path, image_name=image_name, width=width,
                                height=height, qvec = None, cx = cx, cy = cy)

        iou_value = frames_info['frames']['init_frames_iou_bins'][iou_bin][index]['iou']
        black_pixels_ratio = frames_info['frames']['init_frames_iou_bins'][iou_bin][index]['black_pixels_ratio']
        init_outside = True if black_pixels_ratio > 0.3 else False


        result = solve_func(cam_info_init=cam_info_init, task_info=task_info, gaussians=gaussians, solving_args=solving_args)

        output_result = copy.copy(task_info)
        output_result['image_original_path'] = frames_info['frames']['img_path']
        output_result['width_original'] = frames_info['frames']['width']
        output_result['height_original'] = frames_info['frames']['height']
        output_result['cx_original'] = frames_info['frames']['cx']
        output_result['cy_original'] = frames_info['frames']['cy']
        output_result['FovX'] = frames_info['frames']['FovX']
        output_result['FovY'] = frames_info['frames']['FovY']

        output_result['R_gt'] = frames_info['frames']['R']
        output_result['t_gt'] = frames_info['frames']['T']
        output_result['qvec_gt'] = frames_info['frames']['qvec']
        output_result['black_pixels_ratio'] = black_pixels_ratio
        output_result['init_outside'] = init_outside
        output_result['iou_value'] = iou_value
        output_result = output_result | result

        pickle_out_path = os.path.join(experiment_result_dir, 
                                       json_task_path.split('/')[-1].replace('.json', '_' + str(result['success']) + '.pickle'))

        with open(pickle_out_path, 'wb') as handle:
            pickle.dump(output_result, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    fire.Fire(solve_pose_estimation)