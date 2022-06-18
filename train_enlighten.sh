#!/bin/sh

#SBATCH --partition=gpu-titan
#SBATCH --job-name=Enlightening
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=85G
#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tancheelam2@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread

source /home/user/lobbeytan/FYP_low_light_image_enhancement/venv/bin/activate

cd /home/user/lobbeytan/FYP_low_light_image_enhancement/prior_works/enlightenGAN

python script.py --train