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

import os
import random
import json
import torch
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, cameraList_from_custom_json_cameras_images
from utils.general_utils import get_expon_lr_func, get_alternating_gaussians_pose_lrs
from utils.system_utils import mkdir_p
from scene.colmap_loader import read_adj_text_to_dict

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, 
                 shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if not args.custom_trajectory:

            self.train_cameras = {}
            self.test_cameras = {}

            if os.path.exists(os.path.join(args.source_path, "sparse")):
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
                                            #cameras_extrinsic_file_custom = args.cameras_extrinsic_file)
            elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
            else:
                assert False, "Could not recognize scene type!"


            print(scene_info.ply_path)
            if not self.loaded_iter:
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
                json_cams = []
                camlist = []
                if scene_info.test_cameras:
                    camlist.extend(scene_info.test_cameras)
                if scene_info.train_cameras:
                    camlist.extend(scene_info.train_cameras)
                for id, cam in enumerate(camlist):
                    json_cams.append(camera_to_JSON(id, cam))
                with open(os.path.join(self.model_path, "cameras_init.json"), 'w') as file:
                    json.dump(json_cams, file)

            if shuffle:
                random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
                random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

            self.cameras_extent = scene_info.nerf_normalization["radius"]

            for resolution_scale in resolution_scales:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

            if self.loaded_iter:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud.ply"))
            else:
                # print(scene_info.point_cloud)
                # self.gaussians.load_ply(scene_info.ply_path)
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)  


    
    def init_custom_trajectory(self, args : ModelParams, 
                               gaussians : GaussianModel, 
                               load_iteration=None,
                               ):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        custom_trajectory_json = args.custom_trajectory
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.test_cameras = {}
        self.train_cameras = {1:None}

        print("Loading Test Cameras")
        self.test_cameras[1] = cameraList_from_custom_json_cameras_images(custom_trajectory_json, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))


    def cam_adjustment_setup(self, training_args, split = 'train', scale=1.0):
        camera_params_opt = []
        if split == 'train':
            for camera in self.train_cameras[scale]:
                camera_params_opt.append(camera.world_view_transform)
        elif split == 'test':
            for camera in self.test_cameras[scale]:
                camera_params_opt.append(camera.world_view_transform)

        l = [{'params': camera_params_opt, 'lr': training_args.pose_lr_init, "name": "pose_lie"}]

        self.pose_optimizer = torch.optim.SGD(l, lr = training_args.pose_lr_init)
        # self.pose_optimizer = torch.optim.Adam(l, lr = training_args.pose_lr_init, eps=1e-8)
        if training_args.alternating_optimization:
            self.pose_scheduler_args, _ = get_alternating_gaussians_pose_lrs(training_args.pose_lr_init, training_args.pose_lr_final,
                                                                        training_args.alternating_optimization_pose_steps,
                                                                        training_args.alternating_optimization_pose_decay_ratio,
                                                                        training_args.position_lr_init, training_args.position_lr_final,
                                                                        training_args.alternating_optimization_gaussians_steps,
                                                                        training_args.alternating_optimization_gaussians_decay_ratio, 
                                                                        training_args.iterations)
        else:
            self.pose_scheduler_args = get_expon_lr_func(lr_init=training_args.pose_lr_init,
                                        lr_final=training_args.pose_lr_final,
                                        lr_delay_mult=training_args.pose_lr_delay_mult,
                                        max_steps=training_args.pose_lr_max_steps)


    def reset_pose_scheduler(self, training_args):
        self.pose_scheduler_args = get_expon_lr_func(lr_init=training_args.pose_lr_init,
                                lr_final=training_args.pose_lr_final,
                                lr_delay_mult=training_args.pose_lr_delay_mult,
                                max_steps=training_args.pose_lr_max_steps)
        
    def save_camera_poses(self, split, iteration, tag='', scale=1.0):
        if split == 'train':
            path1 = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration), 
                                'adjusted_cameras_' + split + tag + '.txt')
            path2 = os.path.join(self.model_path, 'adjusted_cameras_' + split + tag + '.txt')
        else:
            path1 = os.path.join(self.model_path, 'adjusted_cameras_' + split + tag + '.txt')
            path2 = os.path.join(self.model_path, 'adjusted_cameras_' + split + tag + '.txt')
        mkdir_p(os.path.dirname(path1))
        mkdir_p(os.path.dirname(path2))
        cam_params_dict = {'test' : self.test_cameras[scale],
                           'train': self.train_cameras[scale]}
        with open(path1, 'w') as fwid:
            for camera in cam_params_dict[split]:
                camera_params = list(camera.parameters())
                camera_params = [camera.uid, camera.image_name] + camera_params[0].detach().cpu().numpy().tolist() #+ \
                    # camera_params[1].detach().cpu().numpy().tolist()
                fwid.write(' '.join([str(camera_param) for camera_param in camera_params]))
                fwid.write('\n')
        with open(path2, 'w') as fwid:
            for camera in cam_params_dict[split]:
                camera_params = list(camera.parameters())
                camera_params = [camera.uid, camera.image_name] + camera_params[0].detach().cpu().numpy().tolist()# + \
                    # camera_params[1].detach().cpu().numpy().tolist()
                fwid.write(' '.join([str(camera_param) for camera_param in camera_params]))
                fwid.write('\n')

    def combine_adj_camera_poses(self, tag='', scale=1.0):
        path_train = os.path.join(self.model_path, 'adjusted_cameras_train.txt')
        path_test = os.path.join(self.model_path, 'adjusted_cameras_test_final.txt')

        full_traj = read_adj_text_to_dict(path_train)

        full_traj.update(read_adj_text_to_dict(path_test))
        
        path = os.path.join(self.model_path, 'images_adj' + tag + '.txt')
        with open(path, 'w') as fwid:
            for i, image_name in enumerate(sorted(full_traj.keys())):
                camera_params = full_traj[image_name]
                camera_params = [str(i)] + camera_params + ['1', image_name + '.png']
                fwid.write(' '.join([str(camera_param) for camera_param in camera_params]))
                fwid.write('\n')
                fwid.write(' '.join(['1', '2', '3']))
                fwid.write('\n')

    
    def combine_adj_camera_poses(self, tag='', scale=1.0):
        path_train = os.path.join(self.model_path, 'adjusted_cameras_train.txt')
        path_test = os.path.join(self.model_path, 'adjusted_cameras_test_final.txt')

        full_traj = read_adj_text_to_dict(path_train)

        full_traj.update(read_adj_text_to_dict(path_test))
        
        path = os.path.join(self.model_path, 'images_adj' + tag + '.txt')
        with open(path, 'w') as fwid:
            for i, image_name in enumerate(sorted(full_traj.keys())):
                camera_params = full_traj[image_name]
                camera_params = [str(i)] + camera_params + ['1', image_name + '.png']
                fwid.write(' '.join([str(camera_param) for camera_param in camera_params]))
                fwid.write('\n')
                fwid.write(' '.join(['1', '2', '3']))
                fwid.write('\n')


    def update_learning_rate_pose(self, iteration, factor = 1):
        ''' Learning rate scheduling per step '''
        for param_group in self.pose_optimizer.param_groups:
            if param_group["name"] == "pose_lie":
                if type(self.pose_scheduler_args) == list:
                    lr = self.pose_scheduler_args[iteration] * factor
                else:
                    lr = self.pose_scheduler_args(iteration) * factor
                param_group['lr'] = lr
                return lr

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]