#!/bin/bash

#SBATCH --job-name=mix3r
#SBATCH --account=p33_tsd
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=accel
#SBATCH --gres=gpu:1

set -e # Exit the script on any error
set -u # Treat any unset variables as an error
module --quiet purge  # Reset the modules to the system default
source $HOME/py_envs/mm/bin/activate

CONFIG=$1

python mix3r_int.py --config ${CONFIG}

