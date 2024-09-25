#!/bin/bash
#SBATCH --account=def-celiasmi   # Replace with your account
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G                 # memory per node
#SBATCH --time=0-6:00           # 3 hours
#SBATCH --array=0-5              # Adjust the array range as needed

module load python/3.10 scipy-stack

source ~/projects/def-celiasmi/ns2dumon/rlenv2/bin/activate
cd ~/projects/def-celiasmi/ns2dumon/rl-baselines3-zoo

for seed in $(seq $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID)
do
    for env in "ContinuousMaze-5x5-v0"  "ContinuousMazeBlocks-5x5-v0"  "ContinuousMaze-9x9-v0" "ContinuousMazeBlocks-9x9-v0"
    do
        echo $env
        python train.py --env $env --algo ppo --seed $seed --verbose 0 --gym-packages gym_continuous_maze hrr_gym_wrappers --conf-file hyperparams/learnssp/ppo_ssp.yml  --track --wandb-project-name ssp-continuous-maze --wandb-entity nicole-s-dumont --wandb-tags learnssp-obs
        python train.py --env $env --algo ppo --seed $seed --verbose 0 --gym-packages gym_continuous_maze --conf-file hyperparams/ppo.yml  --track --wandb-project-name ssp-continuous-maze --wandb-entity nicole-s-dumont --wandb-tags default-obs
    done
done