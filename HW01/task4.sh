#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -t 0-00:00:30

#SBATCH --job-name=FirstSlurm
#SBATCH --output=FirstSlurm.out --error=FirstSlurm.err

#SBATCH -c 2

hostname
