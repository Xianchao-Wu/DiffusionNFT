#########################################################################
# File Name: 0.download.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Wed Mar  4 08:19:23 2026
#########################################################################
#!/bin/bash

mkdir reward_ckpts
cd reward_ckpts
# Aesthetic
wget https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/refs/heads/main/sac+logos+ava1-l14-linearMSE.pth
# GenEval
wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth
# ClipScore
wget https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin
# HPSv2.1
wget https://huggingface.co/xswu/HPSv2/resolve/main/HPS_v2.1_compressed.pt
cd ..
