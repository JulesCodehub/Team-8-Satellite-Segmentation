#!/bin/bash
#SBATCH --job-name=unetImageSegBaseline
#SBATCH --output=logs/unetIS_output_%j.txt
#SBATCH --error=logs/unetIS_error_%j.txt
#SBATCH --time=12:00:00
#SBATCH --partition=kimq
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

echo "Activating My custom environment"
source ~/myEnvs/tf_env/bin/activate

cd ~/SwinTransformer

echo "Running Unet Image Segmentation Team 8"
python tools/train.py <CONFIG_FILE> --options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]
