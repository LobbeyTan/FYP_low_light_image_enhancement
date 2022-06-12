#!/bin/sh

#SBATCH --partition=gpu-v100s
#SBATCH --job-name=LLIE_23
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=21G
#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tancheelam2@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread

source /home/user/lobbeytan/FYP_low_light_image_enhancement/venv/bin/activate

python enlighten_train.py
