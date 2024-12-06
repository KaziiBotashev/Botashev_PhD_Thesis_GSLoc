from glob import glob
import subprocess

import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import imageio.v3 as imageio
import json
import cv2
from tqdm import tqdm
import open3d as o3d
import shutil
import mrob
import subprocess
from joblib import Parallel, delayed

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
import fire

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

def create_point_actor(points, colors):
    """ open3d point cloud from numpy array """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    point_cloud.normals = o3d.utility.Vector3dVector(np.zeros_like(points))
    return point_cloud

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def run_extraction(room_name):
    print(room_name)

    replica_poses_basedir = '/mnt/sdb1/home/kbotashev/iros_paper/replica_dataset_1/prepared_scenes_poses_replica'

    output_basedir = '/mnt/sdb1/home/kbotashev/iros_paper/replica_dataset_1/scenes'

    room_output_dir = os.path.join(output_basedir, room_name)
    os.makedirs(room_output_dir, exist_ok=True)

    for replica_poses_dir in glob(os.path.join(replica_poses_basedir, room_name + '_*')):

        interpolation = 2 if '_base' in replica_poses_dir else 1

        scene_name = replica_poses_dir.split('/')[-1]

        print(scene_name)

        replica_output_dir = os.path.join(room_output_dir, scene_name, scene_name+'_raw')
        
        scene_output_dir = os.path.join(room_output_dir, scene_name)

        os.makedirs(replica_output_dir, exist_ok=True)

        print('### STARTED REPLICA EXTRACTOR FOR ' + scene_name)

        command = ['cd /mnt/sdb1/nerf_datasets/replica/replica-gen/ &&'
                    '/mnt/sdb1/home/vpyatov/anaconda3/envs/habitat/bin/python', 
                    '/mnt/sdb1/nerf_datasets/replica/replica-gen/src/example.py',
                    '--dataset=/mnt/sdb1/home/kbotashev/mip-nerf_projects/data/replica_dataset/replica.scene_dataset_config.json',
                    '--scene='+room_name,
                    '--depth_sensor',
                    '--height=1080',
                    '--width=1920',
                    '--interpolation='+str(interpolation),
                    '--poses='+replica_poses_dir,
                    '--output_path='+replica_output_dir]

        subprocess.run(' '.join(command), shell=True)

        #---------------------------------------------------------------------------------------
        print('### FINISHED REPLICA EXTRACTOR, STARTING FORMING PCs FOR ' + scene_name)
        new_dataset_dir = scene_output_dir


        with open(os.path.join(replica_output_dir, 'cam_params.json')) as fp:
            camera_params = json.load(fp)['camera']

        with open(os.path.join(replica_output_dir, 'transforms.json')) as fp:
            transforms = json.load(fp)    

        renamed_images_dir = os.path.join(new_dataset_dir, 'images')
        os.makedirs(renamed_images_dir, exist_ok=True)
        colmap_emulation_dir = os.path.join(new_dataset_dir, 'sparse', '0')
        os.makedirs(colmap_emulation_dir, exist_ok=True)
        images_txt_path = os.path.join(colmap_emulation_dir, 'images.txt')
        cameras_txt_path = os.path.join(colmap_emulation_dir, 'cameras.txt')

        K = np.array([[camera_params['fx'], 0, camera_params['cx']],
                    [0, camera_params['fy'], camera_params['cy']],
                    [0, 0, 1]])

        with open(cameras_txt_path, 'w') as camfwid:
            line_elems = ['1', 'PINHOLE', str(int(camera_params['w'])), str(int(camera_params['h'])), 
                        str(camera_params['fx']), str(camera_params['fy']),
                        str(camera_params['cx']), str(camera_params['cy'])]
            camfwid.write(' '.join(line_elems))

        keypoints_only = False
        decimation_factor = 100
        general_noise_intensity = 0
        peak_noise_intensity = 0
        peak_pose_noise = mrob.geometry.SE3(np.random.rand(6)*peak_noise_intensity)
        peak_noised_cam_indices = np.arange(40,50)

        poses = []
        colors = []
        pcs = []
        sift = cv2.SIFT_create()

        poses = {}
        with open(images_txt_path, "w") as fwid:
            for frame_info in tqdm(transforms['frames']):
                image_path = frame_info['file_path']
                depth_path = frame_info['depth_path']

                image = imageio.imread(os.path.join(replica_output_dir, image_path))/255
                depth = cv2.imread(os.path.join(replica_output_dir, depth_path), cv2.IMREAD_ANYDEPTH)/camera_params['scale']

                cam2world = np.array(frame_info['transform_matrix'])@np.diag([1,-1,-1,1])

                image_num_str = image_path.split('image.')[-1].split('.png')[0]

                poses[image_num_str + '.png'] = cam2world.tolist()

                if general_noise_intensity > 0 and int(image_num_str) not in peak_noised_cam_indices:
                    general_pose_noise = mrob.geometry.SE3(np.random.rand(6)*general_noise_intensity)
                    cam2world = general_pose_noise.mul(mrob.geometry.SE3(cam2world)).T()

                if int(image_num_str) in peak_noised_cam_indices and peak_noise_intensity > 0:
                    cam2world = peak_pose_noise.mul(mrob.geometry.SE3(cam2world)).T()

                world2cam = np.linalg.inv(cam2world)
                
                line_elems = [str(int(image_num_str))]
                tvec = world2cam[:-1,-1]
                rot = world2cam[:3,:3]
                line_elems += rotmat2qvec(rot).astype(str).tolist()
                line_elems += tvec.astype(str).tolist()
                line_elems.append(str(1))
                line_elems.append(image_num_str + '.png')
                fwid.write(' '.join(line_elems))
                fwid.write('\n')
                fwid.write(' '.join(['1', '2', '3']))
                fwid.write('\n')

                u = np.arange(image.shape[1])
                v = np.arange(image.shape[0])

                uu, vv = np.meshgrid(u, v)
                pixs = np.stack([uu, vv, np.ones_like(vv)],-1)

                Kinv_P = np.einsum('ij,bcj->bci', np.linalg.inv(K), pixs)
                D_Kinv_P = np.concatenate([np.multiply(np.expand_dims(depth,-1), Kinv_P), np.expand_dims(np.ones_like(depth),-1)], -1)
                pcd = np.einsum('ij,bcj->bci', cam2world, D_Kinv_P)[:,:,:-1]
                if keypoints_only:
                    image_gray = cv2.imread(os.path.join(replica_output_dir, image_path), cv2.IMREAD_GRAYSCALE)
                    kps, dess = sift.detectAndCompute(image_gray, None)
                    keypoint_coords = np.array([keypoint.pt[::-1] for keypoint in kps]).astype(int)
                    image = image[tuple(keypoint_coords.T)].tolist()
                    pcd = pcd[tuple(keypoint_coords.T)].tolist()
                    decimation_factor = 1
                else:
                    image = image.reshape(-1,3).tolist()[::decimation_factor]
                    pcd = pcd.reshape(-1,3).tolist()[::decimation_factor]
                colors += image
                pcs += pcd
                shutil.copy(os.path.join(replica_output_dir, image_path), os.path.join(renamed_images_dir, image_num_str + '.png'))

        with open(os.path.join(colmap_emulation_dir, "poses_c2w_gt.json"), "w") as outfile:
            json.dump(poses, outfile, indent = 4)
        result_point_actor = create_point_actor(np.array(pcs).reshape(-1, 3), np.array(colors).reshape(-1, 3))
        o3d.io.write_point_cloud(os.path.join(colmap_emulation_dir, 'points3D.ply'), result_point_actor.voxel_down_sample(voxel_size=0.025))
        print('### FINISHED FORMING PCs FOR ' + scene_name)

    #---------------------------------------------------------------------------------------
    
    print('### STARTED GENERATING CAMERA PAIRS FOR ' + room_name)
    ply_path = os.path.join(room_output_dir, room_name+'_base', 'sparse', '0', 'points3D.ply')
    task_dir_path = os.path.join(ply_path.replace('_base/sparse/0/points3D.ply', '_task'))
    # task_dir_path = '/mnt/sdb1/home/kbotashev/iros_paper/replica_dataset/office_4/office_4_task/'
    scene_info = readColmapSceneInfo(task_dir_path, 'images', eval=False)

    pcd_o3d = o3d.io.read_point_cloud(ply_path)
    pcd_o3d = pcd_o3d.voxel_down_sample(voxel_size=0.025)
    points_tensor = torch.tensor(np.asarray(pcd_o3d.points)).unsqueeze(0).cuda().float()
    colors_tensor = torch.tensor(np.asarray(pcd_o3d.colors)).unsqueeze(0).cuda().float()
    points = Pointclouds(points_tensor, features=colors_tensor)

    args = namedtuple('args', ['resolution', 'data_device'])
    args = args(2, 'cuda')
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
                                'cx':cx, 'cy':cy, 
                                'height':height, 'width':width, 
                                'points':points})

        for noise_level in np.linspace(0,0.3,1000).tolist() + np.linspace(0.3,1,1000).tolist():
            tries = 1
            for num_try in range(tries):
                perturbation = np.random.randn(6)*noise_level
                T_init = mrob.geometry.SE3(perturbation).mul(T_gt)

                set_init, image_init = compute_point_cloud_camera_fraction(**{'R':T_init.R(), 'tvec':T_init.t(), 
                                "fx":fx, 'fy':fy, 
                                'cx':cx, 'cy':cy, 
                                'height':height, 'width':width, 
                                'points':points,
                                'build_image':True})
                
                norm_image = cv2.normalize(image_init[0].cpu().numpy(), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

                norm_image = norm_image.astype(np.uint8)

                iou, intersection_indices, union_indices = compute_iou_2sets(set_gt, set_init)

                frame_results[np.round(np.round(iou/0.05)*0.05,2)].append({'R_init':T_init.R(),
                                                            'T_init':T_init.t(),
                                                                'iou':iou,
                                                                # 'init_fraction_set_idx':list(set_init),
                                                                'init_fraction_set_idx':None,
                                                                # 'intersection_idx':intersection_indices,
                                                                'intersection_idx':None,
                                                                # 'union_idx':union_indices,
                                                                'union_idx':None,
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
                                    # 'gt_fraction_set_idx':list(set_gt),
                                    'gt_fraction_set_idx':None,
                                    'init_frames_iou_bins':frame_results,
                                    'init_frames_iou_bins_stats':init_frames_stats
                                    }
        
        os.makedirs(os.path.join(task_dir_path, 'images_pairs'), exist_ok = True)
        with open(os.path.join(task_dir_path, 'images_pairs', cam_info.image_name + '.pickle'), 'wb') as handle:
            pickle.dump(scene_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('#####FINISHED#####')


#----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    fire.Fire(run_extraction)