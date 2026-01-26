#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu 4G
#SBATCH -J pcs

cd /mnt/home/luser/ibo-hpc

source /mnt/home/luser/.bashrc
source /mnt/home/luser/ibo-hpc/.venv/bin/activate

python main.py
