#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# Job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --nodes=1   # Number of nodes to use
#SBATCH --ntasks-per-node=8   # Use 8 processor cores per node
#SBATCH --time=1-0:0:0   # Walltime limit (DD-HH:MM:SS)
#SBATCH --mem=256G   # Maximum memory per node
#SBATCH --gres=gpu:a100:1   # Required GPU hardware
#SBATCH --mail-user=email@insti.edu   # Email address
#SBATCH --mail-type=BEGIN   # Send an email when the job starts
#SBATCH --mail-type=END   # Send an email when the job ends
#SBATCH --mail-type=FAIL   # Send an email if the job fails

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

cd /work-ir/
module load micromamba
export MAMBA_ROOT_PREFIX=/work-dir/micromamba
eval "$(micromamba shell hook --shell=bash)"
micromamba activate moe
which python
which pip
cd sidaMoE/SiDA-MoE/
torchrun --nproc_per_node=1 src/finetune.py --benchmark super_glue --model switch-base-8 --task mrpc --batch_size 64
torchrun --master_port=29601 --nproc_per_node=1 src/ft_accuracy.py --model switch-base-8 --dataset mrpc --eval_bs 64 --n_experts 8 --verbose --KD 
