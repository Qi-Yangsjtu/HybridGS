# HybridGS


This project is based on the official implementation associated with the paper "3D Gaussian Splatting for Real-Time Radiance Field Rendering", which can be found [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).


Abstract: *Most existing 3D Gaussian Splatting (3DGS) compression schemes focus on producing compact 3DGS representation via implicit data embedding. They have long coding time and highly customized data format, making it difficult for standardization efforts and widespread deployment. This paper presents a new 3DGS compression framework called HybridGS, which takes advantage of both compact generation and standard point cloud data encoding. HybridGS first generates compact and explicit 3DGS data. A dual-channel sparse representation is introduced to supervise the primitive position and feature bit depth. It then utilizes a canonical point cloud encoder to perform further data compression and form standard output bitstream. A simple and effective rate control scheme is proposed to pivot the interpretable data compression scheme. At the current stage, HybridGS does not include any modules aimed at improving the 3DGS quality during generation. But experiment results show that it still provides comparable reconstruction performance against state-of-the-art methods, with evidently higher encoding and decoding speed. *


### Setup

```

### Install
```shell
unzip submodules.zip
SET DISTUTILS_USE_SDK=1 # Windows only
conda env create --file environment.yml
conda activate HybridGS
```

### Running

To run the HybridGS, check test_comd.py

There are several new parameters can be used besides vanilla GS

--qp: position bit depth

--color\_latent\_dim: dimension of color latent feature

--color\_latent\_qp: color bit depth

--opacity_qp: opacity bit depth

--scaling_qp: scaling bit depth

--rotation\_latent\_dim: dimension of rotation latent feature

--rotation\_latent\_qp: rotation bit depth

--rateControl : rate control model

    0-no rate control
    1-Rate Control via Primitive Pruning
    2-Rate Control via Adaptive BD

--targetRate (for rateControl=1/2,): target bitstream of rate control

    --targetRate = xx(MB)

--compression_ratio: estimated lossless compression ratio for GPCC, default is 1.3

you will obtain several compact GS samples after running train.py.

The results folder is like the following: 
-point_cloud

    -iteration_50000
    -iteration_70000
        - point_cloud.py (vanilla GS sample for quick check)
        - point_cloud_latent_quan.ply (HybridGS samples in binary coding, GPCC will compress this file)
        - point_cloud_latent_quan_asc.ply (ASCALL version)
        - xxx_quan_para.json: dequantization metadata
        - xxx_latent_decoder.pth: latent feature decoder

Then you can use HybridGS_coding.py for GPCC compression and test metrics.

--Gaussian\_Splatting\_Path: path for compact GS

--GPCC\_Binary: path for GPCC binary

--Cfg\_Path: GPCC configuration file 

--Iteration: which iteration of compact GS needs to be compressed by GPCC

--Results_Path: path for bitstream and metric

--Ground_Truth: ground truth path, for metric calculation

--scene: dataset flag, for image resolution

    - mipNerf360_outdoor, using ''images=images_4''
    - mipNerf360_indoor, using ''images=images_2''
    - other, using ''images=images''

The results after HybriGS_coding.py


    -bitstream
        -iteration_xxx
            -bin: GPCC bitstream
            -decoded_GS: decoded GS sample
            -sub_point_cloud: input of GPCC after split GS sample, see paper section 3.2.1
            -results2.json: bitstream and coding time log
    -deq_point_cloud
        -iteration_xxx
            -point_cloud.ply: dequantized samples after GPCC decoding
    -test
        -render_xxx
            -gt: ground truth image
            -renders: rendered image
            -per_view.json: per view metric
            -results.json: metric results
