project=minigrid
sspdim=201
environments=("MiniGrid-Unlock-v0" "MiniGrid-Empty-8x8-v0" "MiniGrid-Empty-16x16-v0" \
               "MiniGrid-DoorKey-5x5-v0" "MiniGrid-DoorKey-8x8-v0" \
               "MiniGrid-LavaGapS5-v0" "MiniGrid-LavaGapS6-v0" \
               "MiniGrid-FourRooms-v0" "MiniGrid-MultiRoom-N2-S4-v0"
               "MiniGrid-MultiRoom-N4-S5-v0")

# Define corresponding timesteps for each environment
timesteps=(1e5 5e4 1e5 \
           1e5 1e6 \
           3e4 6e4 \
           2e6 1e6 3e6)

for (( seed = 0 ; seed < $1 ; ++seed ))
do
    for i in "${!environments[@]}"
    do
        env="${environments[$i]}"
        ntimesteps="${timesteps[$i]}"
    
        # Now you can use $env and $ntimesteps inside the loop
        echo "Running environment: $env with timesteps: $ntimesteps"
        
        python create_temp_ssp_config.py --env $env --n $ntimesteps --type default
    
        python train.py --env $env --algo ppo --seed $seed --verbose 0 --gym-packages minigrid2 --conf-file temp.yml \
             --track --wandb-project-name $project --wandb-entity nicole-s-dumont --wandb-tags default-obs 
             
        python create_temp_ssp_config.py --env $env --n $ntimesteps --ssp-dim $sspdim  --type view
             
        python train.py --env $env --algo ppo --seed $seed --verbose 0 --gym-packages minigrid2 --conf-file temp.yml \
            --track --wandb-project-name $project --wandb-entity nicole-s-dumont --wandb-tags learnsspview-obs
            
        python create_temp_ssp_config.py --env $env --n $ntimesteps --ssp-dim $sspdim --type xy
            
        python train.py --env $env --algo ppo --seed $seed --verbose 0 --gym-packages minigrid2 --conf-file temp.yml \
            --track --wandb-project-name $project --wandb-entity nicole-s-dumont --wandb-tags learnsspxy-obs
            
        python create_temp_ssp_config.py --env $env --n $ntimesteps --ssp-dim $sspdim  --type view2
             
        python train.py --env $env --algo ppo --seed $seed --verbose 0 --gym-packages minigrid2 --conf-file temp.yml \
            --track --wandb-project-name $project --wandb-entity nicole-s-dumont --wandb-tags learnsspview2-obs
    done
done
