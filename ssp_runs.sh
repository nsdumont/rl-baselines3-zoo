for env in "CartPole-v1" "Pendulum-v1" "MountainCar-v0" "MountainCarContinuous-v0" "LunarLander-v2" "LunarLanderContinuous-v2" "Acrobot-v1"
do
    for (( seed = 0 ; seed < $1 ; ++seed ))
    do
        echo "Seed: $seed"
    	python train.py --env $env --algo ppo --seed $seed --verbose 0 --track --wandb-project-name ssp-rl --wandb-entity nicole-s-dumont --wandb-tags $env default-obs 
        python train.py --env $env --algo ppo --seed $seed --verbose 0 --conf-file  hyperparams/ssp/ppo_ssp.yml --track --wandb-project-name ssp-rl --wandb-entity nicole-s-dumont --wandb-tags $env ssp-obs
        python train.py --env $env --algo ppo --seed $seed --verbose 0 --conf-file hyperparams/python/ppo_rand.py --track --wandb-project-name ssp-rl --wandb-entity nicole-s-dumont --wandb-tags $env rand-obs  
    done
done