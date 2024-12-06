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

room_name = 'apartment_1'

# replica_poses_basedir = '/mnt/sdb1/home/kbotashev/iros_paper/replica_dataset_1/prepared_scenes_poses_replica'

output_basedir = '/mnt/sdb1/home/kbotashev/iros_paper/replica_dataset_1/scenes'

room_output_dir = os.path.join(output_basedir, room_name)
os.makedirs(room_output_dir, exist_ok=True)

replica_poses_dir = '/mnt/sdb1/home/kbotashev/iros_paper/replica_dataset_1/scenes/apartment_1/apartment_1_rebase/apartment_1_rebase_raw_poses'

interpolation = 1 if '_rebase' in replica_poses_dir else 1

scene_name = replica_poses_dir.split('/')[-2]

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