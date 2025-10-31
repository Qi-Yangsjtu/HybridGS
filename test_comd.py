import os
import numpy as np
from icecream import ic
import subprocess
train_script = 'train.py'
import os

# Generate explicit compact GS sample with HybridGS first part
source_path = "E:/ProjectGS/GS_dataset/dance"
model_path = "./output/dance"
qp = '16'
cmd = 'python ' + train_script + ' --data_device="cuda:0" '
cmd = cmd + ' --source_path='+source_path
cmd = cmd + ' --model_path=' + model_path
cmd = cmd + ' --qp='+qp
cmd = cmd + ' --color_latent_dim=6'
cmd = cmd + ' --color_latent_qp=' + qp
cmd = cmd + ' --opacity_qp=' + qp
cmd = cmd + ' --scaling_qp=' + qp
cmd = cmd + ' --rotation_latent_dim=2'
cmd = cmd + ' --rotation_latent_qp=' + qp
log =  model_path + f'/log_BD{qp}.txt'
os.makedirs(os.path.dirname(log), exist_ok=True)
print(cmd)
print(log)
with open(log, 'w') as log_file:
        subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)

# Using GPCC to compress HybridGS samples, and testing metric

GPCC_script = 'HybridGS_coding.py'
result_path = source_path+'/bitstream'
GPCC_cfg = './GPCCforGS/GPCC_cgfs/xyz_reflectance/'
iteration = '70000'
GPCC_binary = 'E:/BaiduSyncdisk/GPCC/tmc3_v25.exe'
cmd = 'python '+ GPCC_script + ' --Gaussian_Splatting_Path=' + model_path + '/'
cmd = cmd + ' --Results_Path='+ result_path
cmd = cmd + ' --Iteration='+iteration
cmd = cmd + ' --Cfg_Path='+GPCC_cfg
cmd = cmd + ' --Ground_Truth='+source_path
cmd = cmd + ' --GPCC_Binary='+ GPCC_binary
cmd = cmd + ' --scene=other'

os.system(cmd)