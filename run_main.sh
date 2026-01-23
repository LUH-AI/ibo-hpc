#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu 4G
#SBATCH -J pcs

cd /mnt/home/lfehring/ibo-hpc

source /mnt/home/lfehring/.bashrc
source /mnt/home/lfehring/ibo-hpc/.venv/bin/activate

python main.py
