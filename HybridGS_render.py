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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_test
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.camera_utils import JSON_to_camera
from icecream import ic

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "{}".format(iteration), "gt")
    ic(render_path)
    ic(gts_path)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render_test(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                camera_json: str, point_cloud, img_path_name):
    llffhold = 8
    with torch.no_grad():
        dataset.camera_json = camera_json
        cameras = JSON_to_camera(dataset)
        cameras.sort(key=lambda x: x.image_name)
        if eval:
            train_cam_infos = [c for idx, c in enumerate(cameras) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cameras) if idx % llffhold == 0]
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        if point_cloud != "default":
            print("\nload file: ", point_cloud)
            scene.gaussians.load_ply(point_cloud)

        original_test_cameras = scene.getTestCameras()
        for idx in range(len(test_cam_infos)):
            original_test_cameras[idx].R = test_cam_infos[idx].R
            original_test_cameras[idx].T = test_cam_infos[idx].T
            original_test_cameras[idx].world_view_transform = test_cam_infos[idx].world_view_transform
            original_test_cameras[idx].projection_matrix = test_cam_infos[idx].projection_matrix
            original_test_cameras[idx].full_proj_transform = test_cam_infos[idx].full_proj_transform
            original_test_cameras[idx].camera_center = test_cam_infos[idx].camera_center

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # if not skip_train:
        #      render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            if img_path_name == "default":
                render_set(dataset.model_path, "test", scene.loaded_iter, original_test_cameras, gaussians, pipeline,
                           background)
            else:
                render_set(dataset.model_path, "test", img_path_name, original_test_cameras, gaussians, pipeline,
                           background)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--camera_json", type=str, default="")
    parser.add_argument("--point_cloud", type=str, default="default")
    parser.add_argument("--img_path_name", type=str, default="default")
    args = parser.parse_args()
    args.eval = True

    safe_state(args.quiet)
    torch.cuda.set_device(args.data_device)
    args.camera_json = os.path.join(args.model_path, "cameras.json")
    ic(args.camera_json)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
                args.camera_json, args.point_cloud, args.img_path_name)

