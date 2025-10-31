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
        self.data_device = "cuda:0"
        self.eval = True

        self.qp = 16
        
        self.remove_outlier_points = True
        self.scale_point = 7500
        self.shift_center = True
        self.shift_left_corner = False


        self.color_latent_dim = 6
        self.color_hid_dim = 50
        self.color_latent_qp = 16


        self.scaling_qp = 16

        self.rotation_latent_dim = 2
        self.rotation_hid_dim = 50
        self.rotation_latent_qp = 16

        self.opacity_qp = 16
        '''
        decoder type 1: normal MLP
        decoder type 2: linear MLP
        '''
        self.decoder_type = 1
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
        self.iterations = 70_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 15_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100

        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500

        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False

        # unique_parameters
        self.unique = True
        self.unique_interval = 500
        self.unique_from_iter = 500
        self.unique_until_iter = 65_000

        # LQ-QGS related lr
        self.latent_lr = 0.000001
        self.scaling_addtional_mult = 0.2
        self.feature_xyz_latent_lr_decay = 1
        self.feature_xyz_latent_lr_init = 15000
        self.feature_xyz_latent_lr_end = 35_000
        self.feature_xyz_latent_lr_interval = 3000



        self.color_decoder_lr = 0.001
        self.color_decoder_adaptive_lr = 0.0001
        self.color_decoder_lr_decay = 0.95

        self.rotation_decoder_lr = 0.015
        self.rotation_decoder_adaptive_lr = 0.0001
        self.rotation_decoder_lr_decay = 0.95

        # prune parameters
        self.prune = True
        self.prune_interval = 2_500
        self.prune_from_iter = 36_000
        self.prune_until_iter = 66_000
        self.prune_percent = 0.1
        self.v_pow = 0.1
        self.prune_percent_decay = 1

        # feature_latent
        self.color_latent_flag = True
        self.feature_color_latent_lr = 0.001
        self.feature_color_latent_lr_decay = 0.95
        self.feature_color_latent_lr_init = 7000
        self.feature_color_latent_lr_end = 35000
        self.feature_color_latent_lr_interval = 3000
        self.feature_color_latent_quantization_flag = True



        self.opacity_quantization_flag = True

        self.scaling_quantization_flag = True
        self.scaling_lr_interval = 3000
        self.scaling_lr_decay = 0.95
        self.scaling_lr_init = 25000
        self.scaling_lr_end = 35000

        self.rotation_latent_flag = True
        self.feature_rotation_latent_lr = 0.005
        self.feature_rotation_latent_lr_init = 7000
        self.feature_rotation_latent_lr_end = 35000
        self.feature_rotation_latent_lr_interval = 3000
        self.feature_rotation_latent_lr_decay = 0.95
        self.feature_rotation_latent_quantization_flag = True


        self.adaptiveQP = False
        self.adaptiveQP_init = 16_000
        self.adaptiveQP_end = 66_000
        self.adaptiveQP_interval = 2500

        '''
        rate control model:
        0 - no control
        1 - model pruning
        2 - model reduce BD
        target rate: MB
        '''
        self.rateControl = 0
        self.targetRate = 1.5
        self.rateControl_iter = 200
        self.compression_ratio = 1.3


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
