#!/bin/bash
#SBATCH --job-name=unetTest
#SBATCH --output=job%j.txt
#SBATCH --error=error%j.txt
#SBATCH -N 1
#SBATCH --time=12:00:00
#SBATCH --partition=kimq
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

echo "Activating My custom environment"
source ~/myEnvs/tf_env/bin/activate

cd ~/SwinTransformer

echo "Running T8 Test"
python tools/train.py configs/pspnet_unet_s5-d16_64x64_40k_drive.py --options model.pretrained=swin_transformer.py [model.backbone.use_checkpoint=True]

echo "Deactivating TensorFlow-2.6.2 environment"
deactivate

echo "Done."