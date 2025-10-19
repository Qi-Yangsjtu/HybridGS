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
import torch
from random import randint

from torch.onnx.symbolic_opset9 import tensor

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.quant_utils import RoundSTE, LatentQuantizer, RobustQuantize
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import random
import numpy as np
import torch.nn as nn
from icecream import ic
from utils.prune import prune_list, calculate_v_imp_score
from torch.optim.lr_scheduler import ExponentialLR
import json
# import copy

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def save_args_to_json(args: ArgumentParser, file_path: str):
    #  args to dict
    args_dict = vars(args)

    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)
    # save as JSON
    file = os.path.join(file_path, 'Model Configuration.json')
    with open(file, 'w') as f:
        json.dump(args_dict, f, indent=4)
    print(f"Arguments saved to {file}")


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.set_device(dataset.data_device)
    scale_point = dataset.scale_point
    shift_center = dataset.shift_center
    shift_left_corner = dataset.shift_left_corner
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians)
    ic(gaussians._xyz.shape[0])
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)


    # initialize color latent feature decoder
    if opt.color_latent_flag:    
        color_degree, color_decoder, optimizer_color_decoder, optimizer_color_decoder_schedular = gaussians.ini_color_latent(dataset, opt)
        if opt.feature_color_latent_quantization_flag:
            color_latent_qp = dataset.color_latent_qp
            color_latent_quan = RobustQuantize()
            print("\nLatent color module with dim: ", dataset.color_latent_dim)
            print("\nColor qp: ", dataset.color_latent_qp)

    # initialize scaling quantizer
    if opt.scaling_quantization_flag:
        print("\nScaling quantization with qp: ", dataset.scaling_qp)
        scaling_qp = dataset.scaling_qp
        scaling_quan = RobustQuantize()

    # initialize rotation latent feature decoder
    if opt.rotation_latent_flag:
        rotation_decoder, optimizer_rotation_decoder, optimizer_rotation_decoder_schedular = gaussians.ini_rotation_latent(dataset, opt)
        if opt.feature_rotation_latent_quantization_flag:
            rotation_latent_qp = dataset.rotation_latent_qp
            rotation_latent_quan = RobustQuantize()
            print("\nLatent rotation module with dim: ", dataset.rotation_latent_dim)
            print("\nRotation qp: ", dataset.rotation_latent_qp)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if opt.unique:
            if iteration == opt.unique_until_iter + 1:
                gaussians.unique_position_LQQGS(opt.latent_lr, iteration > scale_point)
                gaussians._xyz_latent_quan.requires_grad_(False)
                print("\nStop update xyz!")


        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background


        if iteration > scale_point:
            latent_quant = gaussians.quantizer(gaussians._xyz_latent_quan)
            xyz_quant = torch.matmul(latent_quant, gaussians._xyz_base_vector).squeeze(2)
            gaussians._xyz = xyz_quant

        if opt.color_latent_flag:
            if opt.adaptiveQP:
                if (iteration >= opt.adaptiveQP_init and iteration <= opt.adaptiveQP_end
                        and (iteration - opt.adaptiveQP_init) % opt.adaptiveQP_interval == 0):
                    reduced_BD = distribution[0]
                    color_latent_qp = color_latent_qp - reduced_BD
                    print(f"\nRate control mode: iteration {iteration} reduce color latent qp to: {color_latent_qp}")
            color_feature_latent_int, color_feature_latent_rec = color_latent_quan.train(
                gaussians.get_color_features_latent, color_latent_qp)

            ste_color_feature_latent = color_feature_latent_rec

            color_decoded = color_decoder(ste_color_feature_latent)
            primitive_num = gaussians.get_color_features_latent.shape[0]
            color_decoded_reshape = color_decoded.reshape(primitive_num, 3, color_degree).permute(0, 2, 1)
            gaussians._features_dc = color_decoded_reshape[:, 0:1, :]
            gaussians._features_rest = color_decoded_reshape[:, 1:, :]


        # opacity quantizer
        if opt.opacity_quantization_flag:
            if iteration == first_iter:
                opacity_qp = dataset.opacity_qp              
                opacity_quantizer = RobustQuantize()
                gaussians._opacity_rec = gaussians._opacity
            else:
                if opt.adaptiveQP:
                    if (iteration >= opt.adaptiveQP_init and iteration <= opt.adaptiveQP_end
                            and (iteration - opt.adaptiveQP_init) % opt.adaptiveQP_interval == 0):
                        reduced_BD = distribution[0]
                        opacity_qp = opacity_qp - reduced_BD
                        print(f"\nRate control mode: iteration {iteration} reduce opacity qp to: {opacity_qp}")
                opacity_int, opacity_rec = opacity_quantizer.train(gaussians._opacity, opacity_qp)
                gaussians._opacity_rec = opacity_rec


        if opt.scaling_quantization_flag:
            if opt.adaptiveQP:
                if (iteration >= opt.adaptiveQP_init and iteration <= opt.adaptiveQP_end
                        and (iteration - opt.adaptiveQP_init) % opt.adaptiveQP_interval == 0):
                    reduced_BD = distribution[0]
                    scaling_qp = scaling_qp - reduced_BD
                    print(f"\nRate control mode: iteration {iteration} reduce scaling qp to: {scaling_qp}")

            scaling_int, scaling_rec = scaling_quan.train(gaussians._scaling, scaling_qp)
            gaussians._scaling_rec = scaling_rec


        if opt.rotation_latent_flag:
            if opt.adaptiveQP:
                if (iteration >= opt.adaptiveQP_init and iteration <= opt.adaptiveQP_end
                        and (iteration - opt.adaptiveQP_init) % opt.adaptiveQP_interval == 0):
                    reduced_BD = distribution[0]
                    rotation_latent_qp = rotation_latent_qp - reduced_BD
                    del distribution[0]
                    print(f"\nRate control mode: iteration {iteration} reduce rotation latent qp to: {rotation_latent_qp}")

            rotation_feature_latent_int, rotation_feature_latent_rec = rotation_latent_quan.train(
                gaussians.get_rotation_features_latent, rotation_latent_qp)
            ste_rotation_feature_latent = rotation_feature_latent_rec

            rotation_decoded = rotation_decoder(ste_rotation_feature_latent)
            gaussians._rotation = rotation_decoded


        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss.backward()

        iter_end.record()

        with (torch.no_grad()):
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if opt.color_latent_flag and opt.rotation_latent_flag and opt.scaling_quantization_flag:
                    if opt.feature_color_latent_quantization_flag and opt.feature_rotation_latent_quantization_flag:
                        
                        scene.save_latent_quan(iteration, color_feature_latent_int, opacity_int,
                                               scaling_int, rotation_feature_latent_int)                   
                        filePath = os.path.join(dataset.model_path, "point_cloud/iteration_{}".format(iteration))
                        color_latent_quan.save(os.path.join(filePath, "color_latent_quan_para.json"))
                        opacity_quantizer.save(os.path.join(filePath, "opacity_quan_para.json"))
                        scaling_quan.save(os.path.join(filePath, "scaling_quan_para.json"))
                        rotation_latent_quan.save(os.path.join(filePath, "rotation_latent_quan_para.json"))
                        gaussians.save_BD(color_latent_qp, opacity_qp, scaling_qp, rotation_latent_qp,
                                          dataset.model_path, iteration)

                    color_decoder.eval()
                    scene.saveColorFeatureLatent(color_decoder, iteration)
                    color_decoder.train()
                    rotation_decoder.eval()
                    scene.saveRotationFeatureLatent(rotation_decoder, iteration)
                    rotation_decoder.train()

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    # update latent code
                    if iteration > scale_point:
                        gaussians.generate_quant_xyz(opt.latent_lr, shift_center, shift_left_corner)
                        optimizer_latent_quan_schedule = ExponentialLR(gaussians.optimizer_latent_quan,
                                                                       gamma=opt.feature_xyz_latent_lr_decay)
                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if opt.prune:
                if iteration >= opt.prune_from_iter and iteration <= opt.prune_until_iter:
                    if (iteration - opt.prune_from_iter) % opt.prune_interval == 0:
                        if opt.prune_percent !=0:
                            print("\nPruning GS with importance score!")
                            gaussian_list, imp_list = prune_list(gaussians, scene, pipe, background)
                            v_list = calculate_v_imp_score(gaussians, imp_list, opt.v_pow)
                            no = (iteration - opt.prune_from_iter) / opt.prune_interval
                            prune_percent = opt.prune_percent * np.power(opt.prune_percent_decay, no)
                            ic(prune_percent)
                            gaussians.prune_gaussians(prune_percent, v_list, iteration > scale_point)


            if opt.unique:
                if iteration < opt.unique_until_iter and iteration > opt.unique_from_iter and iteration > scale_point:
                    if iteration % opt.unique_interval == 0:
                        before = gaussians._xyz.shape[0]
                        gaussians.unique_position_LQQGS(opt.latent_lr, iteration > scale_point)
                        after = gaussians._xyz.shape[0]
                        print("\n Unique GS: ", f"from {before} to {after}.")

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                if opt.color_latent_flag:
                    optimizer_color_decoder.step()
                    optimizer_color_decoder.zero_grad(set_to_none=True)
                    if opt.feature_color_latent_lr_init < iteration < opt.feature_color_latent_lr_end and \
                            iteration % opt.feature_color_latent_lr_interval == 0:
                        optimizer_color_decoder_schedular.step()
                if opt.scaling_quantization_flag and iteration >= opt.scaling_lr_init and iteration <= opt.scaling_lr_end:
                    gaussians.update_scaling_lr(iteration, opt.scaling_lr_init, opt)
                if opt.rotation_latent_flag:
                    optimizer_rotation_decoder.step()
                    optimizer_rotation_decoder.zero_grad(set_to_none=True)
                    if opt.feature_rotation_latent_lr_init < iteration < opt.feature_rotation_latent_lr_end and \
                            iteration % opt.feature_rotation_latent_lr_interval == 0:
                        optimizer_rotation_decoder_schedular.step()
                        gaussians.update_rotation_latent_rate(iteration, opt)
                if iteration > scale_point:
                    gaussians.optimizer_latent_quan.step()
                    gaussians.optimizer_latent_quan.zero_grad(set_to_none=True)
                    if opt.feature_xyz_latent_lr_init <= iteration <= opt.feature_xyz_latent_lr_end and \
                            iteration % opt.feature_xyz_latent_lr_interval == 0:
                        ini_lr = gaussians.optimizer_latent_quan.param_groups[0]['lr']
                        optimizer_latent_quan_schedule.step()
                        new_lr = gaussians.optimizer_latent_quan.param_groups[0]['lr']
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            # log point number
            gaussians.log_point_number(iteration, os.path.join(scene.model_path, "point_number.log"))

            if iteration == scale_point:
                translate, scale = gaussians.shift_to_origin(shift_center, shift_left_corner)
                scene.rescaleScene(dataset, translate, scale)
                gaussians.adjust_scaling_lr(scale, opt.scaling_addtional_mult)
                gaussians.generate_quant_xyz(opt.latent_lr, shift_center, shift_left_corner)
                optimizer_latent_quan_schedule = ExponentialLR(gaussians.optimizer_latent_quan,
                                                               gamma=opt.feature_xyz_latent_lr_decay)


            if iteration == opt.prune_from_iter-1 and opt.rateControl != 0:
                opt.rateControl_iter = opt.prune_from_iter
                num_primitive = gaussians._xyz.shape[0]
                ic(num_primitive)
                geo_bit = 3 * dataset.qp
                color_bit = dataset.color_latent_dim * color_latent_qp
                opacity_bit = opacity_qp
                scaling_bit = 3 * scaling_qp
                rotation_bit = dataset.rotation_latent_dim * rotation_latent_qp
                per_primitive_bit = geo_bit + color_bit + opacity_bit + scaling_bit + rotation_bit
                per_primitive_MB = ((per_primitive_bit / 8) / 1024) / 1024
                size = num_primitive * per_primitive_MB
                ic(size)
                delat_size = size / opt.compression_ratio - opt.targetRate
                ic(delat_size)
                if opt.rateControl == 1:
                    print("\nRate Control via Primitive Pruning.")
                    if delat_size > 0:
                        start_iter = opt.rateControl_iter
                        end_iter = opt.prune_until_iter
                        iterval = opt.prune_interval
                        pruning_times = np.floor((end_iter - start_iter)/iterval) + 1
                        ic(pruning_times)
                        target_num = np.floor(opt.targetRate / (per_primitive_MB/opt.compression_ratio))
                        ratio = 1 - (target_num/num_primitive)**(1/pruning_times)
                        ic(ratio)
                        opt.prune_percent = ratio
                    else:
                        print("\nMaximum size smaller than target size, do not need rate control!")
                        opt.prune_percent=0
                elif opt.rateControl == 2:
                    print("\nRate Control via Adaptive BD.")
                    if delat_size > 0:
                        start_iter = opt.rateControl_iter
                        opt.adaptiveQP_init = start_iter
                        end_iter = opt.adaptiveQP_end
                        iterval = opt.adaptiveQP_interval
                        reduce_times = int(np.floor((end_iter - start_iter) / iterval) + 1)
                        ic(reduce_times)
                        channels = dataset.color_latent_dim + 1 + 3 + dataset.rotation_latent_dim
                        delta_BD = int(np.ceil(delat_size * opt.compression_ratio / num_primitive/channels*  1024 * 1024 * 8))
                        ic(delta_BD)
                        base = delta_BD // reduce_times
                        extra = delta_BD % reduce_times
                        distribution = [base + 1 if i< extra else base for i in range(reduce_times)]
                        ic(distribution)
                        opt.adaptiveQP = True
                        opt.prune_percent = 0
                    else:
                        print("\nMaximum size smaller than target size, do not need rate control!")
                        opt.prune_percent = 0
                        opt.adaptiveQP = False



def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # print("log path: ", log)
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                p_log = "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test)
                log = os.path.join(scene.model_path, "point_cloud/iteration_{}/result.txt".format(iteration))
                log_dir = os.path.dirname(log)
                if not os.path.exists(log_dir):
                    print(p_log)
                    os.makedirs(log_dir)
                with open(log, "a") as file:
                    file.write(p_log)
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[ 50000, 70000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[ 50000, 70000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    set_random_seed(66)

    test_itera = [7000]
    for i in range(10000, args.iterations + 1, 2500):
        test_itera.append(i)
    print(test_itera)
    args.test_iterations = test_itera
    args.save_iterations = test_itera

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # save configuration
    save_args_to_json(args, args.model_path)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
