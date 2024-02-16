for env in "MiniGrid-FourRooms-v0" "MiniGrid-DoorKey-5x5-v0" "MiniGrid-MultiRoom-N4-S5-v0" "MiniGrid-Fetch-5x5-N2-v0" "MiniGrid-GoToDoor-5x5-v0" "MiniGrid-PutNear-6x6-N2-v0" "MiniGrid-RedBlueDoors-6x6-v0" "MiniGrid-KeyCorridorS3R1-v0" "MiniGrid-Unlock-v0" 
do
    for (( seed = 0 ; seed < $1 ; ++seed ))
    do
        echo "Seed: $seed"
    	python train.py --env $env --algo ppo --seed $seed --verbose 0 --gym-packages minigrid --conf-file hyperparams/ppo.yml  --track --wandb-project-name ssp-rl --wandb-entity nicole-s-dumont --wandb-tags $env default-obs 
        python train.py --env $env --algo ppo --seed $seed --verbose 0 --gym-packages minigrid --conf-file hyperparams/ssp/ppo_ssp.yml --track --wandb-project-name ssp-rl --wandb-entity nicole-s-dumont --wandb-tags $env ssp-obs
    done
done

python train.py --env MiniGrid-DoorKey-5x5-v0 --algo ppo --seed 0 --verbose 0 --gym-packages minigrid --conf-file hyperparams/ssp/ppo_ssp.yml --track --wandb-project-name ssp-rl --wandb-entity nicole-s-dumont --wandb-tags MiniGrid-DoorKey-5x5-v0 ssp-obs
