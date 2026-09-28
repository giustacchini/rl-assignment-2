#!/usr/bin/env bash
#SBATCH --job-name=lunar-dqn-baseline
#SBATCH --time=0-01:00:00
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --partition priority


echo "Running on node: ${SLURMD_NODENAME:-local}"
echo "Job ID: ${SLURM_JOB_ID:-local}"

nvidia-smi

echo "now processing task id:: ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
mkdir "log_${SLURM_JOB_ID}"
python3 src/lunar_env.py > output_${SLURM_JOB_ID}.txt 2>&1

echo "Finished task ${SLURM_JOB_ID:-local}"
