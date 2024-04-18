
for env in "ContinuousMaze-5x5-v0"  "ContinuousMazeBlocks-5x5-v0"  "ContinuousMaze-9x9-v0" "ContinuousMazeBlocks-9x9-v0"
do
    echo $env
    for (( seed = 0 ; seed < $1 ; ++seed ))
    do
        echo "Seed: $seed"
        python train.py --env $env --algo ppo --seed $seed --verbose 0 --gym-packages gym_continuous_maze hrr_gym_wrappers --conf-file hyperparams/ssp/ppo_ssp.yml  --track --wandb-project-name sb3 --wandb-entity nicole-s-dumont --wandb-tags ssp-obs new
    	python train.py --env $env --algo ppo --seed $seed --verbose 0 --gym-packages gym_continuous_maze --conf-file hyperparams/ppo.yml  --track --wandb-project-name sb3 --wandb-entity nicole-s-dumont --wandb-tags default-obs new
    done
done