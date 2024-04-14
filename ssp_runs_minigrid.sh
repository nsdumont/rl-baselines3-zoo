for env in "MiniGrid-Empty-8x8-v0" "MiniGrid-Empty-16x16-v0" "MiniGrid-Unlock-v0"  "MiniGrid-DoorKey-5x5-v0" "MiniGrid-DoorKey-6x6-v0"  #"MiniGrid-PutNear-6x6-N2-v0"  
do
    echo $env
    for (( seed = 0 ; seed < $1 ; ++seed ))
    do
        echo "Seed: $seed"
    	python train.py --env $env --algo ppo --seed $seed --verbose 0 --gym-packages minigrid --conf-file hyperparams/ppo.yml  --track --wandb-project-name sb3 --wandb-entity nicole-s-dumont --wandb-tags default-obs 
        python train.py --env $env --algo ppo --seed $seed --verbose 0 --gym-packages minigrid --conf-file hyperparams/ssp/ppo_ssp.yml --track --wandb-project-name sb3 --wandb-entity nicole-s-dumont --wandb-tags ssp-obs
    done
done

#"MiniGrid-FourRooms-v0" "MiniGrid-MultiRoom-N4-S5-v0" "MiniGrid-GoToDoor-5x5-v0" 