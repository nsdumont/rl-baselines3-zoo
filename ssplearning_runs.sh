
for env in "Acrobot-v1" "Pendulum-v1" "CartPole-v1" "MountainCar-v0"  "LunarLander-v2" 
do
    echo "Env: $env"
    for (( seed = 0 ; seed < $1 ; ++seed ))
    do
        echo "Seed: $seed"
    	python train.py --env $env --algo ppo --seed $seed --verbose 0 --conf-file  hyperparams/ppo.yml --track --wandb-project-name classiccontrol --wandb-entity nicole-s-dumont --wandb-tags $env default-obs 
        python train.py --env $env --algo ppo --seed $seed --verbose 0 --conf-file  hyperparams/learnssp/ppo_ssp.yml --track --wandb-project-name classiccontrol --wandb-entity nicole-s-dumont --wandb-tags $env ssp-obs
    done
done

for env in "MountainCarContinuous-v0" "LunarLanderContinuous-v2" "HalfCheetahBulletEnv-v0" "AntBulletEnv-v0" "HopperBulletEnv-v0" "Walker2DBulletEnv-v0" "ReacherBulletEnv-v0"
do
    echo "Env: $env"
    for (( seed = 0 ; seed < $1 ; ++seed ))
    do
        echo "Seed: $seed"
    	python train.py --env $env --algo sac --seed $seed --verbose 0 --conf-file  hyperparams/sac.yml --track --wandb-project-name classiccontrol --wandb-entity nicole-s-dumont --wandb-tags $env default-obs 
        python train.py --env $env --algo sac --seed $seed --verbose 0 --conf-file  hyperparams/learnssp/sac_ssp.yml --track --wandb-project-name classiccontrol --wandb-entity nicole-s-dumont --wandb-tags $env ssp-obs
    done
done

echo "Beginning high freq eval with early stopping runs"
echo "Env: CartPole-v1"
for (( seed = 0 ; seed < $1 ; ++seed ))
do
    echo "Seed: $seed"
	python train.py --env CartPole-v1 --algo ppo --seed $seed --verbose 0 --eval-freq 10 --eval-episodes 100 --stop-reward-threshold 500.0 --conf-file  hyperparams/ppo.yml --track --wandb-project-name classiccontrol --wandb-entity nicole-s-dumont --wandb-tags CartPole-v1-2 default-obs 
    python train.py --env CartPole-v1 --algo ppo --seed $seed --verbose 0 --eval-freq 10 --eval-episodes 100 --stop-reward-threshold 500.0 --conf-file  hyperparams/learnssp/ppo_ssp.yml --track --wandb-project-name classiccontrol --wandb-entity nicole-s-dumont --wandb-tags CartPole-v1-2 ssp-obs
done

echo "Env: MountainCar-v0"
for (( seed = 0 ; seed < $1 ; ++seed ))
do
    echo "Seed: $seed"
	python train.py --env MountainCar-v0 --algo ppo --seed $seed --verbose 0 --eval-freq 10 --eval-episodes 100 --stop-reward-threshold -110.0 --conf-file  hyperparams/ppo.yml --track --wandb-project-name classiccontrol --wandb-entity nicole-s-dumont --wandb-tags MountainCar-v0-2 default-obs 
    python train.py --env MountainCar-v0 --algo ppo --seed $seed --verbose 0 --eval-freq 10 --eval-episodes 100 --stop-reward-threshold -110.0 --conf-file  hyperparams/learnssp/ppo_ssp.yml --track --wandb-project-name classiccontrol --wandb-entity nicole-s-dumont --wandb-tags MountainCar-v0-2 ssp-obs
done

echo "Env: LunarLander-v2"
for (( seed = 0 ; seed < $1 ; ++seed ))
do
    echo "Seed: $seed"
	python train.py --env LunarLander-v2 --algo ppo --seed $seed --verbose 0 --eval-freq 10 --eval-episodes 100 --stop-reward-threshold 200.0 --conf-file  hyperparams/ppo.yml --track --wandb-project-name classiccontrol --wandb-entity nicole-s-dumont --wandb-tags LunarLander-v2-2 default-obs 
    python train.py --env LunarLander-v2 --algo ppo --seed $seed --verbose 0 --eval-freq 10 --eval-episodes 100 --stop-reward-threshold 200.0 --conf-file  hyperparams/learnssp/ppo_ssp.yml --track --wandb-project-name classiccontrol --wandb-entity nicole-s-dumont --wandb-tags LunarLander-v2-2 ssp-obs
done

echo "Env: MountainCarContinuous-v0"
for (( seed = 0 ; seed < $1 ; ++seed ))
do
    echo "Seed: $seed"
	python train.py --env MountainCarContinuous-v0 --algo sac --seed $seed --verbose 0 --eval-freq 10 --eval-episodes 100 --stop-reward-threshold 90.0 --conf-file  hyperparams/sac.yml --track --wandb-project-name classiccontrol --wandb-entity nicole-s-dumont --wandb-tags MountainCarContinuous-v0-2 default-obs 
    python train.py --env MountainCarContinuous-v0 --algo sac --seed $seed --verbose 0 --eval-freq 10 --eval-episodes 100 --stop-reward-threshold 90.0 --conf-file  hyperparams/learnssp/sac_ssp.yml --track --wandb-project-name classiccontrol --wandb-entity nicole-s-dumont --wandb-tags MountainCarContinuous-v0-2 ssp-obs
done

echo "Env: LunarLanderContinuous-v2"
for (( seed = 0 ; seed < $1 ; ++seed ))
do
    echo "Seed: $seed"
	python train.py --env LunarLanderContinuous-v2 --algo sac --seed $seed --verbose 0 --eval-freq 10 --eval-episodes 100 --stop-reward-threshold 200.0 --conf-file  hyperparams/sac.yml --track --wandb-project-name classiccontrol --wandb-entity nicole-s-dumont --wandb-tags LunarLanderContinuous-v22 default-obs 
    python train.py --env LunarLanderContinuous-v2 --algo sac --seed $seed --verbose 0 --eval-freq 10 --eval-episodes 100 --stop-reward-threshold 200.0 --conf-file  hyperparams/learnssp/sac_ssp.yml --track --wandb-project-name classiccontrol --wandb-entity nicole-s-dumont --wandb-tags LunarLanderContinuous-v2 ssp-obs
done
