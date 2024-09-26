for env in "ContinuousMaze-5x5-v0" "ContinuousMaze-7x7-v0" "ContinuousMaze-8x8-v0" "ContinuousMaze-10x10-v0"  "ContinuousMaze-12x12-v0" "ContinuousMaze-13x13-v0" "ContinuousMaze-15x15-v0" 
do
    echo $env
    for (( seed = 0 ; seed < $1 ; ++seed ))
    do
        echo "Seed: $seed"
        python train.py --env $env --algo ppo --seed $seed --verbose 0 --gym-packages gym_continuous_maze hrr_gym_wrappers --conf-file hyperparams/learnssp/ppo_ssp.yml --hyperparams learning_rate:0.001 --track --wandb-project-name ssp-continuous-maze --wandb-entity nicole-s-dumont --wandb-tags learnssp-obs
    	python train.py --env $env --algo ppo --seed $seed --verbose 0 --gym-packages gym_continuous_maze hrr_gym_wrappers --conf-file hyperparams/ssp/ppo_ssp.yml --hyperparams learning_rate:0.001 --track --wandb-project-name ssp-continuous-maze --wandb-entity nicole-s-dumont --wandb-tags ssp-obs2
    	python train.py --env $env --algo ppo --seed $seed --verbose 0 --gym-packages gym_continuous_maze --conf-file hyperparams/ppo.yml --track --wandb-project-name ssp-continuous-maze --wandb-entity nicole-s-dumont --wandb-tags default-obs 
    done
done