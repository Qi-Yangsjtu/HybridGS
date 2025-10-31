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
from random import gauss

import torch
import numpy as np
# from pyexpat import features
# from torch.cuda import device
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import open3d as o3d
import sys
import math
from utils.quant_utils import Quantizer
from utils.Latent_decoder import Latent_decoder, Latent_decoder_linear
import torch_scatter
from icecream import ic
from torch.optim.lr_scheduler import ExponentialLR
import json

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        # new add
        self.qp = 0
        self.quantizer = None
        self._xyz_base_vector = torch.empty(0)
        self._xyz_latent_quan = torch.empty(0)
        self.optimizer_latent_quan = None

        self._opacity_rec = torch.empty(0)
        self._scaling_rec = torch.empty(0)


        # latent representation for color and rotation
        self._features_color_latent = torch.empty(0)
        self._features_rotation_latent = torch.empty(0)

        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            #     new add
            self._features_color_latent,
            self._features_rotation_latent
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale,
         self._features_color_latent,
         self._features_rotation_latent) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_scaling_rec(self):
        return self.scaling_activation(self._scaling_rec)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    # new add
    @property
    def get_color_features_latent(self):
        return self._features_color_latent

    @property
    def get_rotation_features_latent(self):
        return self._features_rotation_latent

    @property
    def get_sh(self):
        features_rest = self._features_rest
        return features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_opacity_rec(self):
        return self.opacity_activation(self._opacity_rec)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def get_covariance_rec(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling_rec, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, qp, remove_outlier_points,
                        feature_color_latent_dim=6, feature_rotation_latent_dim=1):
        self.spatial_lr_scale = spatial_lr_scale
        self.qp = qp
        self.quantizer = Quantizer()

        print('remove outlier points', remove_outlier_points)
        if remove_outlier_points:
            xyz = pcd.points
            rgb = pcd.colors
            normals = pcd.normals
            pc = o3d.geometry.PointCloud()

            pc.points = o3d.utility.Vector3dVector(xyz)
            pc_filtered, ind = pc.remove_statistical_outlier(nb_neighbors=50, std_ratio=2)
            xyz = xyz[ind]
            rgb = rgb[ind]
            normals = normals[ind]
            pcd = BasicPointCloud(xyz, rgb, normals)

        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        # new add
        feature_color_latent = torch.zeros((fused_point_cloud.shape[0], feature_color_latent_dim), device="cuda")
        feature_rotation_latent = torch.zeros((fused_point_cloud.shape[0], feature_rotation_latent_dim), device="cuda")
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        #     new add
        self._features_color_latent = nn.Parameter(feature_color_latent.requires_grad_(True))
        self._features_rotation_latent = nn.Parameter(feature_rotation_latent.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        color_lr = training_args.feature_color_latent_lr
        rotation_lr = training_args.feature_rotation_latent_lr
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            #     new add
            {'params': [self._features_color_latent], 'lr': color_lr,
             "name": "feature_color_latent"},
            {'params': [self._features_rotation_latent], 'lr': rotation_lr,
             "name": "feature_rotation_latent"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def shift_to_origin(self, shift_center, shift_corner):
        coord = self._xyz
        min_coord, _ = torch.min(coord, dim=0)
        max_coord, _ = torch.max(coord, dim=0)
        bbox_center = (max_coord + min_coord) / 2
        if shift_center and shift_corner:
            print("shift_center and shift_corner can not both be True!")
            sys.exit()
        if shift_center == True:
            translate = bbox_center
            self._xyz.data -= translate
            # -min_value - max_value --> -2^(n-1) - 2^(n-1) - 1
            max_value = self.get_max_value()
            max_int = 2 ** (self.qp - 1) - 1
            scale = max_int / max_value
        elif shift_corner == True:
            translate = min_coord
            self._xyz.data -= translate
            # 0 - max_value --> 0 - 2^(n)-1
            max_coord, _ = torch.max(self._xyz, dim=0)
            max_value = torch.max(max_coord)
            max_int = 2 ** self.qp - 1
            scale = max_int / max_value
        else:
            # both not True
            translate = torch.tensor([0, 0, 0], dtype=torch.float, device="cuda")
            self._xyz.data -= translate
            # -min_value - max_value --> -2^(n-1) - 2^(n-1) - 1
            max_value = self.get_max_value()
            max_int = 2 ** (self.qp - 1) - 1
            scale = max_int / max_value

        self.scale_center(scale)
        self.scale_scale(scale)

        self._xyz = torch.round(self._xyz)
        translate = translate.cpu().numpy()
        scale = scale.cpu().numpy()
        return -translate, scale

    def get_max_value(self):
        coord = self._xyz
        max_coord, _ = torch.max(coord, dim=0)
        max_max_coord = torch.max(max_coord)
        min_coord, _ = torch.min(coord, dim=0)
        max_min_coord = torch.max(torch.abs(min_coord))

        max_value = torch.max(max_max_coord, max_min_coord)
        return max_value

    def scale_center(self, scale):
        new_xyz = self._xyz * scale
        self._xyz.data = new_xyz

    def scale_scale(self, scale):
        new_scaling = self.scaling_inverse_activation(self.get_scaling * scale)
        self._scaling.data = new_scaling

    def adjust_scaling_lr(self, factor, additional_mult):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in ['scaling']:
                param_group['lr'] *= math.log(factor) * additional_mult

    def update_scaling_lr(self, iteration, scale_point, opt):
        if (iteration - scale_point)%opt.scaling_lr_interval ==0:
            for param_group in self.optimizer.param_groups:
                if param_group["name"] in ['scaling']:
                    init_lr = param_group['lr']
                    # ic(init_lr)
                    new_lr = init_lr * torch.tensor(opt.scaling_lr_decay)
                    param_group['lr'] = new_lr
                    # ic(new_lr)

    def generate_quant_xyz(self, latent_lr, shift_center, shift_corner):
        # _xyz --> [-1 0 1] triple code
        if shift_center:
            depth = self.qp - 1
        elif shift_corner:
            depth = self.qp
        else:
            depth = self.qp - 1
        # -2^(depth+1) - 2^(depth+1)
        self._xyz_base_vector = torch.flip(torch.pow(2, torch.arange(depth, dtype=torch.float)), dims=[0]).unsqueeze(
            1).cuda()
        coord = torch.round(self._xyz.data).cpu()
        coord_abs = coord.int().abs()
        # 2^(depth-1) 2^(depth-2) ... 2^1 2^0
        binary_repr = (coord_abs.unsqueeze(-1) >> torch.arange(depth - 1, -1, -1)) & 1
        negative_mask = coord < 0
        binary_repr[negative_mask] = -binary_repr[negative_mask]

        # add some noise: xyz = tanh(xyz)
        binary_xyz_latent = torch.atanh(binary_repr)
        random_zeros_replace = torch.rand(binary_xyz_latent.size()) - 0.5
        random_ones_replace = torch.rand(binary_xyz_latent.size()) * (1.49 - 0.55) + 0.55
        random_negones_replace = torch.rand(binary_xyz_latent.size()) * (1.49 - 0.55) - 1.49
        new_binary_xyz_latent = torch.where(binary_xyz_latent == 0, random_zeros_replace, binary_xyz_latent)
        new_binary_xyz_latent2 = torch.where(new_binary_xyz_latent == float('inf'), random_ones_replace,
                                             new_binary_xyz_latent)
        new_binary_xyz_latent3 = torch.where(new_binary_xyz_latent2 == float('-inf'), random_negones_replace,
                                             new_binary_xyz_latent2)
        self._xyz_latent_quan = nn.Parameter(new_binary_xyz_latent3.requires_grad_(True).cuda())
        self.optimizer_latent_quan = torch.optim.Adam([self._xyz_latent_quan], lr=latent_lr)
        self.optimizer_latent_quan.zero_grad()

    def log_point_number(self, iteration, log_file):
        counts = self._xyz.size(0)
        unique_elements = torch.unique(self._xyz, dim=0)
        unique_counts = unique_elements.size(0)
        with open(log_file, "a+") as f:
            f.write(f"iteration {iteration}: {counts} {unique_counts}\n")

    def mask_xyz_latent(self, mask):
        for group in self.optimizer_latent_quan.param_groups:
            stored_state = self.optimizer_latent_quan.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][~mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][~mask]

                del self.optimizer_latent_quan.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][~mask].requires_grad_(True)))
                self.optimizer_latent_quan.state[group['params'][0]] = stored_state

                self._xyz_latent_quan = group["params"][0]
            else:
                group["params"][0] = nn.Parameter((group["params"][0][~mask].requires_grad_(True)))
                self._xyz_latent_quan = group["params"][0]
    def unique_position_LQQGS(self, laten_lr, iterOverScalePoint=False):
        ellipse_volumn = self.get_scaling.prod(dim=1)
        latent_quant = self.quantizer(self._xyz_latent_quan)
        xyz_quant = torch.matmul(latent_quant, self._xyz_base_vector).squeeze(2)
        xyzQI = xyz_quant
        unique_xyz, indices = torch.unique(xyzQI, dim=0, return_inverse=True)

        _, max_size_indices_per_group = torch_scatter.scatter_max(ellipse_volumn, dim=0, index=indices)
        mask = torch.ones(xyzQI.size(0), dtype=torch.bool, device=self._xyz.device)
        mask[max_size_indices_per_group] = False
        self.prune_points(mask)
        if iterOverScalePoint:
            self.mask_xyz_latent(mask)


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr


    def update_rotation_latent_rate(self, iteration, opt):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "feature_rotation_latent":
                init_lr = torch.tensor(opt.feature_rotation_latent_lr)
                lr_decay = torch.tensor(opt.feature_rotation_latent_lr_decay)
                pow_num = torch.tensor(
                    (iteration - opt.feature_rotation_latent_lr_init) / opt.feature_rotation_latent_lr_interval)
                new_lr = init_lr * torch.pow(lr_decay, pow_num)
                param_group['lr'] = new_lr
                return new_lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def construct_list_of_latent_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_color_latent.shape[1]):
            l.append('color_latent_{}'.format(i))
        l.append('opacity')
        for i in range(self._features_scaling_latent.shape[1]):
            l.append('scale_latent_{}'.format(i))
        for i in range(self._features_rotation_latent.shape[1]):
            l.append('rot_latent_{}'.format(i))
        return l

    def construct_list_of_latent_attributes_2(self, scale_latent):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_color_latent.shape[1]):
            l.append('color_latent_{}'.format(i))
        l.append('opacity')
        for i in range(scale_latent.shape[1]):
            l.append('scale_latent_{}'.format(i))
        for i in range(self._features_rotation_latent.shape[1]):
            l.append('rot_latent_{}'.format(i))
        return l

    def save_ply(self, path, binary_flag = False):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        if binary_flag:
            asic_path = path.split('.ply')[0] + '_asic.ply'
            mkdir_p(os.path.dirname(asic_path))
            PlyData([el], text=True).write(asic_path)


    def save_latent_quan_ply(self, path, color, opacity, scaling, rotation):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        color_latent = color.detach().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scale_latent = scaling.detach().cpu().numpy()
        rotation_latent = rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_latent_attributes_2(scale_latent)]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, color_latent, opacities, scale_latent, rotation_latent), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        asic_path = path.split('.ply')[0] + '_asc.ply'
        mkdir_p(os.path.dirname(asic_path))
        PlyData([el], text=True).write(asic_path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # new add
        self._features_color_latent = optimizable_tensors["feature_color_latent"]
        self._features_rotation_latent = optimizable_tensors["feature_rotation_latent"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_feature_color_latent,
                              new_feature_rotation_latent):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation,
             # new add
             "feature_color_latent": new_feature_color_latent,
             "feature_rotation_latent": new_feature_rotation_latent, }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # new add
        self._features_color_latent = optimizable_tensors["feature_color_latent"]
        self._features_rotation_latent = optimizable_tensors["feature_rotation_latent"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        # new add
        new_feature_color_latent = self._features_color_latent[selected_pts_mask].repeat(N, 1)
        new_feature_rotation_latent = self._features_rotation_latent[selected_pts_mask].repeat(N, 1)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
                                   new_feature_color_latent, new_feature_rotation_latent)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        # new add
        new_feature_color_latent = self._features_color_latent[selected_pts_mask]
        new_feature_rotation_latent = self._features_rotation_latent[selected_pts_mask]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_feature_color_latent,
                                   new_feature_rotation_latent)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1

    def fix_para(self):
        print("fix all the attributes except scaling latent feature!")
        self._xyz.requires_grad_(False)
        self._features_dc.requires_grad_(False)
        self._features_rest.requires_grad_(False)
        self._opacity.requires_grad_(False)
        self._scaling.requires_grad_(False)
        self._rotation.requires_grad_(False)
        self._features_color_latent.requires_grad_(False)
        self._features_rotation_latent.requires_grad_(False)

    def active_para(self):
        print("activate all the attributes!")
        self._xyz.requires_grad_(True)
        self._features_dc.requires_grad_(True)
        self._features_rest.requires_grad_(True)
        self._opacity.requires_grad_(True)
        self._scaling.requires_grad_(True)
        self._rotation.requires_grad_(True)
        self._features_color_latent.requires_grad_(True)
        self._features_rotation_latent.requires_grad_(True)


    def prune_gaussians(self, percent, important_score, isLatentxyz = False):
        ic(important_score.shape)
        if percent !=0:
            sorted_tensor, _ = torch.sort(important_score, dim=0)
            index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
            value_nth_percentile = sorted_tensor[index_nth_percentile]
            prune_mask = (important_score <= value_nth_percentile).squeeze()
            self.prune_points(prune_mask)
            if isLatentxyz:
                self.mask_xyz_latent(prune_mask)


    def ini_color_latent(self, dataset, opt):
        degree = self.get_features.shape[1]
        if dataset.decoder_type==1:
            print("\n Normal decoder for color.")
            color_decoder = Latent_decoder(input_dim=dataset.color_latent_dim, output_dim=degree * 3,
                                       hidden_dim = dataset.color_hid_dim).cuda()
        elif dataset.decoder_type==2:
            print("\n Linear decoder for color.")
            color_decoder = Latent_decoder_linear(input_dim=dataset.color_latent_dim, output_dim=degree * 3,
                                       hidden_dim = dataset.color_hid_dim).cuda()
        else:
            sys.exit("\nDecoder type has to be 1 or 2!")

        optimizer_color_decoder = torch.optim.Adam(color_decoder.parameters(), lr=opt.color_decoder_lr)
        color_decoder.train()
        optimizer_color_decoder.zero_grad()
        optimizer_color_decoder_schedular = ExponentialLR(optimizer_color_decoder, gamma=opt.color_decoder_lr_decay)
        return degree, color_decoder, optimizer_color_decoder, optimizer_color_decoder_schedular


    def ini_rotation_latent(self, dataset, opt):
        rotation_degree = 4
        if dataset.decoder_type==1:
            print("\n Normal decoder for rotation.")
            rotation_decoder = Latent_decoder(input_dim=dataset.rotation_latent_dim, output_dim=rotation_degree,
                                         hidden_dim=dataset.rotation_hid_dim).cuda()
        elif dataset.decoder_type==2:
            print("\n Linear decoder for rotation.")
            rotation_decoder = Latent_decoder_linear(input_dim=dataset.rotation_latent_dim, output_dim=rotation_degree,
                                              hidden_dim=dataset.rotation_hid_dim).cuda()
        else:
            sys.exit("\nDecoder type has to be 1 or 2!")
        optimizer_rotation_decoder = torch.optim.Adam(rotation_decoder.parameters(), lr=opt.rotation_decoder_lr)
        rotation_decoder.train()
        optimizer_rotation_decoder.zero_grad()
        optimizer_rotation_decoder_schedular = ExponentialLR(optimizer_rotation_decoder, gamma=opt.rotation_decoder_lr_decay)
        return rotation_decoder, optimizer_rotation_decoder, optimizer_rotation_decoder_schedular

    def save_BD(self, color_BD, opacity_BD, scaling_BD, rotation_BD, model_path, iteration):
        data = {
            "color_BD": color_BD,
            "opacity_BD": opacity_BD,
            "scaling_BD": scaling_BD,
            "rotation_BD": rotation_BD,
        }
        root_path = os.path.join(model_path, "point_cloud/iteration_{}".format(iteration))
        path = os.path.join(root_path, "attr_BD.json")
        with open(path, 'w') as f:
            json.dump(data, f, indent = 4)