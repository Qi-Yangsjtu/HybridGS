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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
from typing import NamedTuple
import json
from PIL import Image

WARNED = False

class ImagePlaceholder:
    def __init__(self, mode, size):
        self.mode = mode
        self.size = size
        self.width, self.height = size
        
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    
def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
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

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

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

def JSON_to_camera(args):
    with open(args.camera_json, 'r') as file:
        camera_entries = json.load(file)
    cam_infos = []
    for camera_entry in camera_entries:
        id = camera_entry['id']
        img_name = camera_entry['img_name']
        width = camera_entry['width']
        height = camera_entry['height']
        # width = 1600
        # height = 1000
        
        #  TODO: load gt image
        img = Image.new('RGB', (width, height))
        
        position = np.array(camera_entry['position'])
        rotation = np.array(camera_entry['rotation'])
        W2C = np.zeros((4, 4))
        W2C[:3, :3] = rotation
        W2C[:3, 3] = position
        W2C[3, 3] = 1.0
        Rt = np.linalg.inv(W2C)
        R = np.transpose(Rt[:3, :3])
        T = Rt[:3, 3]
        
        focal_length_y = camera_entry['fy']
        focal_length_x = camera_entry['fx']
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
        
        cam_info = CameraInfo(uid=id, R=R, T=T, FovY=FovY, FovX=FovX, image=img,
                              image_path=None, image_name=img_name, width=width, height=height)
        cam_infos.append(cam_info)
    
    resolution_scales = [1.0]
    for resolution_scale in resolution_scales:
            render_cameras = cameraList_from_camInfos(cam_infos, resolution_scale, args)
    return render_cameras