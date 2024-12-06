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

from scene.cameras import CameraLie, CameraQuat, CameraLieTorch
from scene.cameras import CameraLie as Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov

from tqdm import tqdm
from glob import glob
from PIL import Image
import json
import os
from scene.colmap_loader import qvec2rotmat
from scene.colmap_loader import read_adj_text_to_dict

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, rescaled_size = 640, camera_type='lie'):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        scale = (resolution_scale * args.resolution)
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > rescaled_size:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>" + str(rescaled_size) + " pixels width), rescaling to it.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / rescaled_size
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
    if camera_type == 'lie':
        return CameraLie(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, qvec = cam_info.qvec,
                    FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                    image=gt_image, gt_alpha_mask=loaded_mask, cx = cam_info.cx/scale, cy = cam_info.cy/scale,
                    image_name=cam_info.image_name, uid=id, data_device=args.data_device)
    if camera_type == 'lietorch':
        return CameraLieTorch(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, qvec = cam_info.qvec,
                    FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                    image=gt_image, gt_alpha_mask=loaded_mask, cx = cam_info.cx/scale, cy = cam_info.cy/scale,
                    image_name=cam_info.image_name, uid=id, data_device=args.data_device)
    elif camera_type == 'qtvec':
        return CameraQuat(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, qvec = cam_info.qvec,
            FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
            image=gt_image, gt_alpha_mask=loaded_mask, cx = cam_info.cx/scale, cy = cam_info.cy/scale,
            image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []    
    # cam_infos = cam_infos[:20]

    for id, c in tqdm(enumerate(cam_infos), total = len(cam_infos)):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

# def cameraList_from_adjusted_txt(traj_txt_path, args):
#         path_train = os.path.join(args.model_path, 'adjusted_cameras_train.txt')
#         path_test = os.path.join(args.model_path, 'adjusted_cameras_test_final.txt')

#         full_traj = read_adj_text_to_dict(path_train)

#         full_traj.update(read_adj_text_to_dict(path_test))

#         images_dir = os.path.join(args.source_path, 'images')
    
#         for id, img_name in tqdm(enumerate(full_traj.keys()), total=len(full_traj.keys()), desc = 'Loading custom trajectory: '):

#             qvec = np.array(full_traj[img_name][:4])
#             t = np.array(full_traj[img_name][4:])

#             T = np.zeros((4,4))
#             T[:-1, :-1] = qvec2rotmat(qvec)
#             T[:-1,  -1] = t

#             T = np.linalg.inv(T)

#             R = T[:-1, :-1]
#             t = T[:-1,  -1]

#             image_path

#             image = PILtoTorch(Image.open(image_path))
#             height, width  = image.shape[1:]

#             fovx = np.deg2rad(cam_info['fov'])
#             # fx = width / (2 * np.arctan(fovx / 2))
#             fx = width / (2 * (np.tan(fovx/2)))

#             fovy = 2 * np.arctan(height/(2*fx))

#             camera_list.append(Camera(colmap_id=id, 
#                                         R=R, T=t, 
#                                         FoVx=fovx, FoVy=fovy, 
#                                         image=image, gt_alpha_mask=None,
#                                         image_name=image_name, uid=id, data_device=args.data_device))
#         return camera_list


def cameraList_from_custom_json_cameras_images(json_traj_path, args):
    camera_list = []
    image_paths = sorted(glob(json_traj_path.replace('json', 'mp4_frames') + '/*'))
    with open(json_traj_path) as f:
        traj = json.load(fp = f)   
    camera_traj = traj['camera_path']
    for id, cam_info in tqdm(enumerate(camera_traj), total=len(camera_traj), desc = 'Loading custom trajectory: '):
        T = np.linalg.inv(np.array(cam_info['camera_to_world']).reshape(4,4)@np.diag([1,-1,-1,1]))
        R = T[:-1, :-1]
        t = T[:-1,  -1]

        image_path = image_paths[id]
        image_name = image_path.split('/')[-1]

        image = PILtoTorch(Image.open(image_path))
        height, width  = image.shape[1:]

        fovx = np.deg2rad(cam_info['fov'])
        # fx = width / (2 * np.arctan(fovx / 2))
        fx = width / (2 * (np.tan(fovx/2)))

        fovy = 2 * np.arctan(height/(2*fx))

        camera_list.append(Camera(colmap_id=id, 
                                    R=R, T=t, 
                                    FoVx=fovx, FoVy=fovy, 
                                    image=image, gt_alpha_mask=None,
                                    image_name=image_name, uid=id, data_device=args.data_device))
    return camera_list


def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
