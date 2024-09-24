#!/bin/bash
#SBATCH --account=def-celiasmi   # Replace with your account
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G                 # memory per node
#SBATCH --time=0-12:00           # 3 hours
#SBATCH --array=0-5              # Adjust the array range as needed

module load python/3.10 scipy-stack

source ~/projects/def-celiasmi/ns2dumon/rlenv2/bin/activate
cd ~/projects/def-celiasmi/ns2dumon/rl-baselines3-zoo

for seed in $(seq $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID)
do
    for env in "Acrobot-v1" "Pendulum-v1" "CartPole-v1" "MountainCar-v0"  "LunarLander-v2" 
    do
        echo "Env: $env"
        
        #python train.py --env $env --algo ppo --seed $seed --verbose 0 --conf-file  hyperparams/ppo.yml --track --wandb-project-name classiccontrol --wandb-entity nicole-s-dumont --wandb-tags $env default-obs 
        python train.py --env $env --algo ppo --seed $seed --verbose 0 --conf-file  hyperparams/learnssp/ppo_ssp.yml --track --wandb-project-name classiccontrol --wandb-entity nicole-s-dumont --wandb-tags $env learnssp-obs
        python train.py --env $env --algo ppo --seed $seed --verbose 0 --conf-file  hyperparams/ssp/ppo_ssp.yml --track --wandb-project-name classiccontrol --wandb-entity nicole-s-dumont --wandb-tags $env ssp-obs
        
    done

    for env in "MountainCarContinuous-v0"  "HalfCheetahBulletEnv-v0" "AntBulletEnv-v0" "HopperBulletEnv-v0" "Walker2DBulletEnv-v0" "ReacherBulletEnv-v0"
    do
        echo "Env: $env"
       	python train.py --env $env --algo sac --seed $seed --verbose 0 --conf-file  hyperparams/sac.yml --track --wandb-project-name classiccontrol --wandb-entity nicole-s-dumont --wandb-tags $env default-obs 
        python train.py --env $env --algo sac --seed $seed --verbose 0 --conf-file  hyperparams/learnssp/sac_ssp.yml --track --wandb-project-name classiccontrol --wandb-entity nicole-s-dumont --wandb-tags $env learnssp-obs
    done

    echo "Beginning high freq eval with early stopping runs"
    echo "Env: CartPole-v1"
	python train.py --env CartPole-v1 --algo ppo --seed $seed --verbose 0 --eval-freq 10 --eval-episodes 100 --stop-reward-threshold 500.0 --conf-file  hyperparams/ppo.yml --track --wandb-project-name classiccontrol --wandb-entity nicole-s-dumont --wandb-tags CartPole-v1-2 default-obs 
    python train.py --env CartPole-v1 --algo ppo --seed $seed --verbose 0 --eval-freq 10 --eval-episodes 100 --stop-reward-threshold 500.0 --conf-file  hyperparams/learnssp/ppo_ssp.yml --track --wandb-project-name classiccontrol --wandb-entity nicole-s-dumont --wandb-tags CartPole-v1-2 learnssp-obs

    echo "Env: MountainCar-v0"
	python train.py --env MountainCar-v0 --algo ppo --seed $seed --verbose 0 --eval-freq 10 --eval-episodes 100 --stop-reward-threshold -110.0 --conf-file  hyperparams/ppo.yml --track --wandb-project-name classiccontrol --wandb-entity nicole-s-dumont --wandb-tags MountainCar-v0-2 default-obs 
    python train.py --env MountainCar-v0 --algo ppo --seed $seed --verbose 0 --eval-freq 10 --eval-episodes 100 --stop-reward-threshold -110.0 --conf-file  hyperparams/learnssp/ppo_ssp.yml --track --wandb-project-name classiccontrol --wandb-entity nicole-s-dumont --wandb-tags MountainCar-v0-2 learnssp-obs

    echo "Env: MountainCarContinuous-v0"
	python train.py --env MountainCarContinuous-v0 --algo sac --seed $seed --verbose 0 --eval-freq 10 --eval-episodes 100 --stop-reward-threshold 90.0 --conf-file  hyperparams/sac.yml --track --wandb-project-name classiccontrol --wandb-entity nicole-s-dumont --wandb-tags MountainCarContinuous-v0-2 default-obs 
    python train.py --env MountainCarContinuous-v0 --algo sac --seed $seed --verbose 0 --eval-freq 10 --eval-episodes 100 --stop-reward-threshold 90.0 --conf-file  hyperparams/learnssp/sac_ssp.yml --track --wandb-project-name classiccontrol --wandb-entity nicole-s-dumont --wandb-tags MountainCarContinuous-v0-2 learnssp-obs

done
