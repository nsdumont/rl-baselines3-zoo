#!/bin/bash
#SBATCH --account=def-celiasmi   
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G                 
#SBATCH --time=0-6:00           #
#SBATCH --array=0-49        


module load python/3.11 scipy-stack mujoco swig mysql
source ~/projects/def-celiasmi/ns2dumon/rlzoo/bin/activate
cd ~/projects/def-celiasmi/ns2dumon/rl-baselines3-zoo

project=minigrid3
sspdim=393

seeds=(0 1 2 3 4)
environments=("MiniGrid-Empty-8x8-v0" "MiniGrid-Empty-16x16-v0" \
               "MiniGrid-DoorKey-5x5-v0" "MiniGrid-DoorKey-8x8-v0" \
               "MiniGrid-LavaGapS5-v0" "MiniGrid-LavaGapS6-v0" \
               "MiniGrid-FourRooms-v0" "MiniGrid-MultiRoom-N2-S4-v0"
               "MiniGrid-KeyCorridorS3R2-v0" "MiniGrid-KeyCorridorS4R3-v0")
timesteps=(5e4 1e5 \
           1e5 1e6 \
           3e4 6e4 \
           1e6 1e5
           1e5 1e5)

seed_id=$(($SLURM_ARRAY_TASK_ID % ${#seeds[@]}))
env_id=$(($SLURM_ARRAY_TASK_ID / ${#seeds[@]}))

seed="${seeds[$seed_idx]}"
env="${environments[$env_id]}"
ntimesteps="${timesteps[$env_id]}"

echo "Running environment: $env with timesteps: $ntimesteps"
        
python create_temp_ssp_config.py --env $env --n $ntimesteps --type default

python train.py --env $env --algo ppo --seed $seed --verbose 0 --gym-packages minigrid --conf-file temp.yml \
     --track --wandb-project-name $project --wandb-entity nicole-s-dumont --wandb-tags default-obs 
     
python create_temp_ssp_config.py --env $env --n $ntimesteps --ssp-dim $sspdim  --type view
     
python train.py --env $env --algo ppo --seed $seed --verbose 0 --gym-packages minigrid --conf-file temp.yml \
    --track --wandb-project-name $project --wandb-entity nicole-s-dumont --wandb-tags learnsspview-obs
    
python create_temp_ssp_config.py --env $env --n $ntimesteps --ssp-dim $sspdim --type xy
    
python train.py --env $env --algo ppo --seed $seed --verbose 0 --gym-packages minigrid --conf-file temp.yml \
    --track --wandb-project-name $project --wandb-entity nicole-s-dumont --wandb-tags learnsspxy-obs
    
python create_temp_ssp_config.py --env $env --n $ntimesteps --ssp-dim $sspdim  --type view2
     
python train.py --env $env --algo ppo --seed $seed --verbose 0 --gym-packages minigrid --conf-file temp.yml \
    --track --wandb-project-name $project --wandb-entity nicole-s-dumont --wandb-tags learnsspview2-obs
