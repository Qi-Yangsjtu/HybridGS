import json
import re
import sys
# sys.path.append("/")
import numpy as np
from plyfile import PlyData, PlyElement
import os
import argparse
from icecream import ic
import time
from utils.system_utils import mkdir_p
from utils.quant_utils import RobustQuantize
import torch
from scene import Scene, GaussianModel

def extract_features(plydata:PlyData, prefix = "color_latent_"):
    property_names = plydata.elements[0].data.dtype.names
    keys = [key for key in property_names if key.startswith(prefix)]
    if not keys:
        raise ValueError(f"No features found with prefix '{prefix}'")
    key_values = [np.array(plydata.elements[0][key]) for key in keys]
    values = np.stack(key_values, axis=1)
    return values



class LatentGS:
    def __init__(self):
        self._xyz = np.empty(0)
        self._color = np.empty(0)
        self._opacity = np.empty(0)
        self._scaling = np.empty(0)
        self._rotation = np.empty(0)
        self._matrix = np.empty(0)

    def load_ply(self, path, sort_by_xyz = True):
        print("GS path: ", path)
        plydata = PlyData.read(path)
        xyz = np.stack(
            (np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"])), axis = 1
        )
        opacity = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        color = extract_features(plydata, 'color_latent_')
        scaling = extract_features(plydata, 'scale_latent_')
        rotation = extract_features(plydata, 'rot_latent_')


        self._xyz = xyz
        self._color = color
        self._opacity = opacity
        self._scaling = scaling
        self._rotation = rotation
        self._matrix = np.hstack((xyz, color, opacity, scaling, rotation))
        if sort_by_xyz:
            self.sort_by_xyz()

    def save_ply(self, path, data, color_dim, binary_flag = False):
        self._xyz = data[:,0:3]
        self._color = data[:, 3:3+color_dim]
        self._opacity = data[:, 3+color_dim: 3+color_dim+1]
        self._scaling = data[:,3+ color_dim+1: 3+color_dim+4]
        self._rotation = data[:, 3+ color_dim+4:]

        normal = np.zeros(self._xyz.shape)
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_latent_attributes()]
        elements = np.empty(self._xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((self._xyz, normal, self._color, self._opacity, self._scaling, self._rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        mkdir_p(os.path.dirname(path))
        PlyData([el]).write(path)
        if not binary_flag:
            asic_path = path.split('.ply')[0] + '_asic.ply'
            mkdir_p(os.path.dirname(asic_path))
            PlyData([el], text=True).write(asic_path)


    def sort_by_xyz(self):
        xyz_np = self._xyz
        indices = np.lexsort((xyz_np[:,2], xyz_np[:,1], xyz_np[:,0]))
        sorted_matrix = self._matrix[indices]
        ic(self._matrix[0])
        ic(sorted_matrix[0])
        self._matrix = sorted_matrix
        self._xyz = self._xyz[indices]
        self._color = self._color[indices]
        self._opacity = self._opacity[indices]
        self._scaling = self._scaling[indices]
        self._rotation = self._rotation[indices]

    def get_feature_dim(self):
        return self._matrix.shape[1]

    def get_matrix(self):
        return self._matrix

    def get_color_dim(self):
        return self._color.shape[1]

    def construct_list_of_latent_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._color.shape[1]):
            l.append('color_latent_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_latent_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_latent_{}'.format(i))
        return l

    def get_geo(self):
        return self._xyz

    def get_color(self):
        return self._color

    def get_opacity(self):
        return self._opacity

    def get_scaling(self):
        return self._scaling

    def get_rotation(self):
        return self._rotation

class PointCloud:
    def __init__(self):
        self._xyz = np.empty(0)
        self._f = np.empty(0)
        self._matrix = np.empty(0)

    def load_ply(self, path, sort_by_xyz = True):
        print("Point cloud path: ", path)
        plydata = PlyData.read(path)
        xyz = np.stack(
            (np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"])), axis = 1
        )
        f = np.asarray(plydata.elements[0]["refc"])[..., np.newaxis]


        self._xyz = xyz
        self._f = f
        self._matrix = np.hstack((xyz, f))
        if sort_by_xyz:
            self.sort_by_xyz()

    def sort_by_xyz(self):
        xyz_np = self._xyz
        indices = np.lexsort((xyz_np[:,2], xyz_np[:,1], xyz_np[:,0]))
        sorted_matrix = self._matrix[indices]
        self._matrix = sorted_matrix
        self._xyz = self._xyz[indices]
        self._f = self._f[indices]


    def get_feature_dim(self):
        return self._matrix.shape[1]

    def get_matrix(self):
        return self._matrix

    def get_geo(self):
        return self._xyz
    def get_attr(self):
        return self._f


def saveXYZRToPLY(XYZR, ply_path, binary = False):
    dtype_full = [("x", "f4"), ("y", "f4"),("z", "f4"),("reflectance", "u2")]
    elements = np.empty(XYZR.shape[0], dtype= dtype_full)
    elements[:] = list(map(tuple, XYZR))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el], text=not binary).write(ply_path)

def Hybrid_Coding(source_path, iteration, cfg_path, binary_flag, result_path, GPCC_binary):
    print("\nHybridGS coding...")

    GS_data = LatentGS()
    GS_path = source_path + "point_cloud"+ f"/iteration_{iteration}" + "/point_cloud_latent_quan.ply"
    GS_data.load_ply(GS_path)
    gpcc_bitstream_path = os.path.join(result_path, f"iteration_{iteration}", 'bin')
    ic(gpcc_bitstream_path)
    gpcc_point_cloud_path = os.path.join(result_path, f"iteration_{iteration}", 'sub_point_cloud')
    ic(gpcc_point_cloud_path)
    if not os.path.exists(gpcc_bitstream_path):
        os.makedirs(gpcc_bitstream_path)
    if not os.path.exists(gpcc_point_cloud_path):
        os.makedirs(gpcc_point_cloud_path)

    feature_dim_full = GS_data.get_feature_dim()
    ic(feature_dim_full)
    matrix_quant = GS_data.get_matrix()

    geo_size = []
    geo_enc_time = []
    geo_dec_time = []
    total_enc_time = []

    attr_size = []
    attr_enc_time = []
    attr_dec_time = []
    total_dec_time = []

    # coding and decoding by GPCC
    for feature_index in range(0, feature_dim_full-3):
    # for feature_index in range(0, 1):
        xyz_feature = matrix_quant[:, [0,1,2, 3+ feature_index]]
        pc_xyz_feature_path = os.path.join(gpcc_point_cloud_path, f"point_cloud_feature_{feature_index}.ply")
        saveXYZRToPLY(xyz_feature, pc_xyz_feature_path, binary=binary_flag)
        pc_xyz_feature_bit_path = os.path.join(gpcc_bitstream_path, f"point_cloud_feature_{feature_index}.bin")
        enc_cfg = os.path.join(cfg_path, "encoder.cfg")
        dec_cfg = os.path.join(cfg_path, "decoder.cfg")
        enc_log = os.path.join(gpcc_bitstream_path, f"point_cloud_feature_{feature_index}_enc.log")
        dec_log = os.path.join(gpcc_bitstream_path, f"point_cloud_feature_{feature_index}_dec.log")
        pc_xyz_feature_path_rec = os.path.join(gpcc_point_cloud_path, f"point_cloud_feature_{feature_index}_rec.ply")
        encode_cmd = (f"{GPCC_binary} --uncompressedDataPath={pc_xyz_feature_path} --compressedStreamPath={pc_xyz_feature_bit_path}"
               f" --config={enc_cfg}>{enc_log}")
        ic(encode_cmd)
        os.system(encode_cmd)

        decode_cmd = (f"{GPCC_binary} --compressedStreamPath={pc_xyz_feature_bit_path} --reconstructedDataPath={pc_xyz_feature_path_rec}"
                      f" --config={dec_cfg} --outputBinaryPly={binary_flag} > {dec_log}")

        ic(decode_cmd)
        os.system(decode_cmd)

        with open(enc_log, 'r') as file:
            data = file.read()

        position_size = re.findall(r'positions bitstream size (\d+) B', data)
        position_time = re.findall(r'positions processing time \(user\): ([\d.]+) s', data)
        reflectances_size = re.findall(r'reflectances bitstream size (\d+) B', data)
        reflectances_time = re.findall(r'reflectances processing time \(user\): ([\d.]+) s', data)

        ic(position_size)

        position_size = sum(int(size) for size in position_size)

        position_time = sum(float(time) for time in position_time)
        reflectances_size = sum(int(size) for size in reflectances_size)
        reflectances_time = sum(float(time) for time in reflectances_time)

        geo_size.append(position_size)
        geo_enc_time.append(position_time)
        attr_size.append(reflectances_size)
        attr_enc_time.append(reflectances_time)

        with open(dec_log, 'r') as file:
            data = file.read()
        pattern = r'(positions|reflectances) processing time \(user\): ([\d.]+) s'
        matches = re.findall(pattern, data)
        ic(matches)
        geo_dec_sub_time = []
        attr_dec_sub_time = []
        for match in matches:
            type_, time  = match
            if type_ == 'positions':
                geo_dec_sub_time.append(float(time))
            elif type_ == 'reflectances':
                attr_dec_sub_time.append(float(time))
        geo_dec_time.append(sum(geo_dec_sub_time))
        attr_dec_time.append(sum(attr_dec_sub_time))
    geoSize = geo_size[0]
    geoEncTime = sum(geo_enc_time) / len(geo_enc_time)
    geoDecTime = sum(geo_dec_time) / len(geo_dec_time)
    attrSize = sum(attr_size)
    attrEncTime = sum(attr_enc_time)
    attrDecTime = sum(attr_dec_time)

    geo_size_mb = geoSize/(1024**2)
    attr_size_mb = attrSize/(1024**2)
    total_mb = geo_size_mb + attr_size_mb

    data= {
        "file": GS_path,
        "Geo_size": geo_size_mb,
        "Geo_enc_time": geoEncTime,
        "Geo_dec_time": geoDecTime,
        "Sub_Attr_size": attrSize,
        "Sub_Attr_enc_time": attr_enc_time,
        "Sub_Attr_dec_time": attr_dec_time,
        "Attr_size": attr_size_mb,
        "Attr_enc_time_serial": attrEncTime,
        "Attr_dec_time_serial": attrDecTime,
        "Total_size": total_mb,
        "Total_enc_time_serial": geoEncTime+attrEncTime,
        "Total_dec_time_serial": geoDecTime + attrDecTime,
    }
    with open(os.path.join(result_path, f"iteration_{iteration}", 'results2.json'), "w") as json_file:
        json.dump(data, json_file, indent=4)

    files = [f for f in os.listdir(gpcc_point_cloud_path) if f.endswith("_rec.ply")]
    files = sorted(files, key=lambda x: int(x.split("_")[-2]))
    attr_rec = []
    idx = 0

    for file in files:
        file_path = os.path.join(gpcc_point_cloud_path, file)
        rec_point = PointCloud()
        rec_point.load_ply(file_path, sort_by_xyz = True)
        if idx==0:
            geo = rec_point.get_geo()
            idx += 1
            gs_data = geo
        attr = rec_point.get_attr()
        attr_rec.append(attr)
        gs_data = np.hstack((gs_data, attr))

    ic(gs_data[0])
    rec_gs = LatentGS()
    gs_rec_path = os.path.join(result_path, f"iteration_{iteration}", 'decoded_GS','point_cloud_latent_quan.ply')

    rec_gs.save_ply(gs_rec_path, gs_data, color_dim=GS_data.get_color_dim(), binary_flag=False)

    return rec_gs

def HybridGS_dequantization(sample:LatentGS, metadata, iteration):
    print("\nHybridGS dequantization.")
    metadata_path = os.path.join(metadata, f"point_cloud/iteration_{iteration}")
    ic(metadata_path)
    color_decoder = metadata_path + '/color_latent_decoder.pth'
    color_deq_para = metadata_path + '/color_latent_quan_para.json'
    opacity_deq_para = metadata_path + '/opacity_quan_para.json'
    scaling_deq_para = metadata_path + '/scaling_quan_para.json'
    rotation_decoder = metadata_path + '/rotation_latent_decoder.pth'
    rotation_deq_para = metadata_path + '/rotation_latent_quan_para.json'

    color_decoder = torch.load(color_decoder,
                               map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    color_decoder.eval()

    rotation_decoder = torch.load(rotation_decoder,
                               map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    rotation_decoder.eval()




    device = next(color_decoder.parameters()).device
    color_quan = RobustQuantize()
    color_quan.load(color_deq_para)
    color_quan.toCUDA(device)
    color_latent = torch.tensor(sample.get_color(), device=device)

    start_time = time.time()
    color_latent_rec = color_quan.reconstruct_infer(color_latent)
    end_time = time.time()
    print("color deq: ", end_time - start_time)

    start_time = time.time()
    color_features = color_decoder(color_latent_rec)

    end_time = time.time()
    print("color MLP: ", end_time - start_time)
    ic(color_features.shape)
    color_degree = int(color_features.shape[1]/3)

    primitive_num = color_features.shape[0]
    color_decoded_reshape = color_features.reshape(primitive_num, 3, color_degree).permute(0, 2, 1)
    features_dc = color_decoded_reshape[:, 0:1, :]
    features_rest = color_decoded_reshape[:, 1:, :]

    device = next(rotation_decoder.parameters()).device
    rotation_quan = RobustQuantize()
    rotation_quan.load(rotation_deq_para)
    rotation_quan.toCUDA(device)
    rotation_latent = torch.tensor(sample.get_rotation(), device=device)
    rotation_latent_rec = rotation_quan.reconstruct_infer(rotation_latent)
    start_time = time.time()
    rotation_features = rotation_decoder(rotation_latent_rec)
    end_time = time.time()
    print("rotation MLP: ", end_time - start_time)

    opacity_quan = RobustQuantize()
    opacity_quan.load(opacity_deq_para)
    opacity_deq = opacity_quan.reconstruct_infer(sample.get_opacity())

    scaling_quan = RobustQuantize()
    scaling_quan.load(scaling_deq_para)
    scaling_deq = scaling_quan.reconstruct_infer(sample.get_scaling())


    Vanilla_GS = GaussianModel(3)
    Vanilla_GS._xyz = torch.tensor(sample.get_geo(), device=device)
    Vanilla_GS._features_dc = features_dc
    Vanilla_GS._features_rest = features_rest
    Vanilla_GS._opacity = torch.tensor(opacity_deq, device=device)
    Vanilla_GS._scaling = torch.tensor(scaling_deq, device=device)
    Vanilla_GS._rotation = rotation_features

    output_path = os.path.join(source_path, f"deq_point_cloud/iteration_{iteration}/point_cloud.ply")
    Vanilla_GS.save_ply(output_path, binary_flag = True)

    return output_path

def HybridGS_metric(model_path, source_path, iteration, save_path, img_path_name, render, scene = 'other'):
    print("\nHybridGS metric testing.")
    project_dir = os.path.dirname(__file__)
    ic(project_dir)
    render_script = os.path.join(project_dir, "HybridGS_render.py")
    metric_script = os.path.join(project_dir, "metrics.py")
    device = "cuda:0"

    if render and scene == 'mipNerf360_outdoor':
        cmd_render = f"python {render_script} --iteration {iteration} -s {source_path} -m {model_path} --point_cloud={save_path} --img_path_name={img_path_name} --eval --skip_train --data_device={device} --images=images_4"
        cmd_metric = f"python {metric_script} -m {model_path}  --data_device={device} --iteration={iteration} --img={img_path_name}"
        os.system(cmd_render)
        os.system(cmd_metric)
    elif render and scene == 'mipNerf360_indoor':
        cmd_render = f"python {render_script} --iteration {iteration} -s {source_path} -m {model_path} --point_cloud={save_path} --img_path_name={img_path_name} --eval --skip_train --data_device={device} --images=images_2"
        cmd_metric = f"python {metric_script} -m {model_path}  --data_device={device} --iteration={iteration} --img={img_path_name}"
        os.system(cmd_render)
        os.system(cmd_metric)
    else:
        cmd_render = f"python {render_script} --iteration {iteration} -s {source_path} -m {model_path} --point_cloud={save_path} --img_path_name={img_path_name} --eval --skip_train --data_device={device}"
        cmd_metric = f"python {metric_script} -m {model_path}  --data_device={device} --iteration={iteration} --img={img_path_name}"
        os.system(cmd_render)
        os.system(cmd_metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HybridGS coding.")
    parser.add_argument("--Gaussian_Splatting_Path", type = str,
            default='./output/dance/')
    parser.add_argument("--GPCC_Binary", type = str, default='./GPCCforGS/tmc3_v25.exe')
    parser.add_argument("--Results_Path", type= str, default='./output/bitstream/')
    parser.add_argument("--Iteration", nargs="+", type=int,  default=[50000, 70000])
    parser.add_argument("--Cfg_Path", type = str, default='./GPCCforGS/GPCC_cgfs/xyz_reflectance/')
    parser.add_argument("--BinaryFlag", type = str, default=0)
    parser.add_argument("--Ground_Truth", type = str, default=".../...") # set source path
    parser.add_argument("--scene", type= str, default='other')


    args = parser.parse_args()

    source_path = args.Gaussian_Splatting_Path
    iterations = args.Iteration
    cfg_path = args.Cfg_Path
    binary_flag = args.BinaryFlag
    result_path = args.Results_Path
    GPCC_binary = args.GPCC_Binary
    ground_truth = args.Ground_Truth
    scene = args.scene
    for iteration in iterations:
        rec_gs = Hybrid_Coding(source_path, iteration, cfg_path, binary_flag, result_path, GPCC_binary)
        deq_GS_path = HybridGS_dequantization(rec_gs, source_path, iteration)
        render_path = f"render_{iteration}"
        HybridGS_metric(source_path, ground_truth, iteration, deq_GS_path, render_path, render = True, scene=scene)

