from pytorch3d import io
import torch
from pytorch3d.renderer.cameras import FoVPerspectiveCameras, PerspectiveCameras
from pytorch3d.renderer.points.rasterizer import PointsRasterizer, PointsRasterizationSettings
from pytorch3d.structures import Pointclouds
from pytorch3d.utils import cameras_from_opencv_projection 
from pytorch3d.renderer import (
    PointsRenderer,
    NormWeightedCompositor
)

from scene.dataset_readers import readColmapSceneInfo
from utils.camera_utils import loadCam
from collections import namedtuple
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import mrob
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import pickle
import cv2

def compute_point_cloud_camera_fraction(R, tvec, fx, fy, cx, cy, height, width, points, build_image=False):
    camera_matrix = torch.tensor([[fx, 0, cx],
                               [0, fy, cy],
                               [0,0,1]])
    # R = [camera.R for camera in cameras]
    # tvec = [camera.T for camera in cameras]

    camera_p3d = cameras_from_opencv_projection(R = torch.tensor(np.array(R)).unsqueeze(0).float(), 
                                                tvec = torch.tensor(np.array(tvec)).unsqueeze(0).float(), 
                                                camera_matrix = camera_matrix.unsqueeze(0).float(),
                                                image_size = torch.tensor([height, 
                                                                          width]).unsqueeze(0).float())
    
    raster_settings = PointsRasterizationSettings(
                    image_size=(height, 
                                width), 
                    radius = 0.025,
                    points_per_pixel = 1 
                    )

    # Create a points rasterizer
    

    rasterizer = PointsRasterizer(cameras=camera_p3d.cuda(), raster_settings=raster_settings)
    rasterized = rasterizer(points)

    image = None
    if build_image:
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            # Pass in background_color to the alpha compositor, setting the background color 
            # to the 3 item tuple, representing rgb on a scale of 0 -> 1, in this case blue
            compositor=NormWeightedCompositor()
        )

        image = renderer(points)

    fraction_set = set(torch.unique(rasterized.idx)[1:].tolist())

    return fraction_set, image

def compute_iou_2sets(set0, set1):

    intersection_indices = set0.intersection(set1)
    union_indices = set0.union(set1)
    iou = len(intersection_indices)/len(union_indices)

    return iou, list(intersection_indices), list(union_indices)


ply_path = "/mnt/sdb1/home/kbotashev/iros_paper/replica_dataset/office_4/office_4_base/sparse/0/points3D.ply"
task_dir_path = os.path.join(ply_path.replace('_base/sparse/0/points3D.ply', '_task'))
# task_dir_path = '/mnt/sdb1/home/kbotashev/iros_paper/replica_dataset/office_4/office_4_task/'
scene_info = readColmapSceneInfo(task_dir_path, 'images', eval=False)

pcd_o3d = o3d.io.read_point_cloud(ply_path)
pcd_o3d = pcd_o3d.voxel_down_sample(voxel_size=0.025)
points_tensor = torch.tensor(np.asarray(pcd_o3d.points)).unsqueeze(0).cuda().float()
colors_tensor = torch.tensor(np.asarray(pcd_o3d.colors)).unsqueeze(0).cuda().float()
points = Pointclouds(points_tensor, features=colors_tensor)

args = namedtuple('args', ['resolution', 'data_device'])
args = args(2, 'cuda:1')
pipe = namedtuple('pipe', ['convert_SHs_python', 'compute_cov3D_python', 'debug'])
pipe = pipe(False, False, False)
bg_color = [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

ids = np.arange(len(scene_info.train_cameras))

# scene_results = {}
# scene_results['point_cloud_path'] = ply_path
# scene_results['scene'] = ply_path.split('/')[-5]
# scene_results['frames'] = {}
for id in tqdm(ids):
    cam_info = scene_info.train_cameras[id]
    camera = loadCam(args=args, id = id, cam_info=cam_info, resolution_scale=1)

    height = camera.image_height
    width = camera.image_width
    fx = camera.fx
    fy = camera.fy
    cx = camera.cx
    cy = camera.cy
    scene_results = {}
    scene_results['point_cloud_path'] = ply_path
    scene_results['scene'] = ply_path.split('/')[-5]
    scene_results['img_name'] = cam_info.image_name
    frame_results = {np.round(iou_bin, 2):[] for iou_bin in np.linspace(0,1,21)}

    T_gt = mrob.geometry.SE3(mrob.geometry.SO3(camera.R), camera.T)
    set_gt, _ = compute_point_cloud_camera_fraction(**{'R':T_gt.R(), 'tvec':T_gt.t(), 
                            "fx":fx, 'fy':fy, 
                            'cx':cy, 'cy':cy, 
                            'height':height, 'width':width, 
                            'points':points})

    for noise_level in np.linspace(0,0.3,1000).tolist() + np.linspace(0.3,1,1000).tolist():
        tries = 1
        for num_try in range(tries):
            perturbation = np.random.randn(6)*noise_level
            T_init = mrob.geometry.SE3(perturbation).mul(T_gt)

            set_init, image_init = compute_point_cloud_camera_fraction(**{'R':T_init.R(), 'tvec':T_init.t(), 
                            "fx":fx, 'fy':fy, 
                            'cx':cy, 'cy':cy, 
                            'height':height, 'width':width, 
                            'points':points,
                            'build_image':True})
            
            norm_image = cv2.normalize(image_init[0].cpu().numpy(), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

            norm_image = norm_image.astype(np.uint8)

            iou, intersection_indices, union_indices = compute_iou_2sets(set_gt, set_init)

            frame_results[np.round(np.round(iou/0.05)*0.05,2)].append({'R_init':T_init.R(),
                                                           'T_init':T_init.t(),
                                                            'iou':iou,
                                                            'init_fraction_set_idx':list(set_init),
                                                            'intersection_idx':intersection_indices,
                                                            'union_idx':union_indices,
                                                            'init_image':norm_image
                                                            })
            
    init_frames_stats = {key:len(frame_results[key]) for key in frame_results.keys()}    
    
    scene_results['frames'] = {'img_path':cam_info.image_path,
                                'R':cam_info.R,
                                'T':cam_info.T,
                                'uid':id,
                                'FovY':cam_info.FovY,
                                'FovX':cam_info.FovX,
                                'image':np.array(cam_info.image),
                                'width':cam_info.width,
                                'height':cam_info.height,
                                'qvec':cam_info.qvec,
                                'cx':cam_info.cx,
                                'cy':cam_info.cy,
                                'gt_fraction_set_idx':list(set_gt),
                                'init_frames_iou_bins':frame_results,
                                'init_frames_iou_bins_stats':init_frames_stats
                                }
    
    os.makedirs(os.path.join(task_dir_path, 'images_pairs'), exist_ok = True)
    with open(os.path.join(task_dir_path, 'images_pairs', cam_info.image_name + '.pickle'), 'wb') as handle:
        pickle.dump(scene_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('---FINISHED---')

