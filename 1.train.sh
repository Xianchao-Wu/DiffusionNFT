#########################################################################
# File Name: 1.train.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Wed Mar  4 08:37:41 2026
#########################################################################
#!/bin/bash

#export WANDB_API_KEY=xxx
#export WANDB_ENTITY=xxx

# GenEval
#torchrun --nproc_per_node=8 scripts/train_nft_sd3.py --config config/nft.py:sd3_geneval
CUDA_VISIBLE_DEVICES=7 python -m ipdb scripts/train_nft_sd3.py \
	--config config/nft.py:sd3_ocr
#--config config/nft.py:sd3_geneval

# Multi-reward
#torchrun --nproc_per_node=8 scripts/train_nft_sd3.py --config config/nft.py:sd3_multi_reward
