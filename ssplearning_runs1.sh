#!/bin/bash
#SBATCH --account=def-celiasmi   # Replace with your account
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G                 # memory per node
#SBATCH --time=0-12:00           # 3 hours
#SBATCH --array=0-5              # Adjust the array range as needed

module load python/3.10 scipy-stack
source ~/projects/def-celiasmi/ns2dumon/rlenv2/bin/activate
cd ~/projects/def-celiasmi/ns2dumon/rl-baselines3-zoo
module load python/3.10 scipy-stack mujoco


for seed in $(seq $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID)
do
    for env in "HalfCheetah-v4" "Ant-v4" "Hopper-v4" "Walker2d-v4" "Swimmer-v4"
    do
        echo "Env: $env"
       	python train.py --env $env --algo sac --seed $seed --verbose 0 --conf-file  hyperparams/sac.yml --track --wandb-project-name classiccontrol --wandb-entity nicole-s-dumont --wandb-tags $env default-obs 
        python train.py --env $env --algo sac --seed $seed --verbose 0 --conf-file  hyperparams/learnssp/sac_ssp.yml --track --wandb-project-name classiccontrol --wandb-entity nicole-s-dumont --wandb-tags $env learnssp-obs
    done

done
