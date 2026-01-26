#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu 4G
#SBATCH -J pcs

cd /mnt/home/user/ibo-hpc

source /mnt/home/user/.bashrc
source /mnt/home/user/ibo-hpc/.venv/bin/activate

python main.py
