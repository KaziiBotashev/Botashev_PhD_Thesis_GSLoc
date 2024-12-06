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

from argparse import ArgumentParser, Namespace
import sys
import os
from collections import namedtuple

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.custom_trajectory = ""
        self.cameras_extrinsic_file = ""
        self.img_tag = ""
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # self.iterations = 30_000
        # self.position_lr_init = 0.00016
        # self.position_lr_final = 0.0000016
        # self.position_lr_delay_mult = 0.01
        # self.position_lr_max_steps = 30_000
        # self.feature_lr = 0.0025
        # self.opacity_lr = 0.05
        # self.scaling_lr = 0.005
        # self.rotation_lr = 0.001
        # self.percent_dense = 0.01
        # self.lambda_dssim = 0.2
        # self.densification_interval = 100
        # self.opacity_reset_interval = 3000
        # self.densify_from_iter = 500
        # self.densify_until_iter = 15_000
        # self.densify_grad_threshold = 0.0002
        # self.pose_lr_init = 0.001
        # self.pose_lr_final = 0.0000001
        # self.pose_lr_delay_mult = 1
        # self.pose_lr_max_steps = 15_000
        # self.pose_adjustment_steps = 15_000
        # super().__init__(parser, "Optimization Parameters")

        self.iterations = 40_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        # self.position_lr_final = 0.000005
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 40_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 20_000
        self.densify_grad_threshold = 0.0002
        self.pose_lr_init = 0.0001
        self.pose_lr_final = 0.000001
        # self.pose_lr_init = 0.01
        # self.pose_lr_final = 0.0001
        self.pose_lr_delay_mult = 1
        self.pose_lr_max_steps = 20_000
        self.pose_adjustment_steps = 20_000
        # self.pose_qvec_lr = 0.01
        # self.pose_tvec_lr = 0.01 
        self.alternating_optimization = False
        self.alternating_optimization_pose_steps = 2000
        self.alternating_optimization_pose_decay_ratio = 0.01
        self.alternating_optimization_gaussians_steps = 6000
        self.alternating_optimization_gaussians_decay_ratio = 0.01

        self.adaptive_cam_choice = False
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

pose_estimation_params = namedtuple('pose_estimation_params', ['pose_lr_init', 'pose_lr_final', 'pose_lr_delay_steps', 'pose_lr_delay_mult', 'pose_lr_max_steps', 'alternating_optimization'])