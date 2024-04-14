for env in "maze-sample-5x5-v0" #"maze-sample-6x6-v0" "maze-sample-7x7-v0" "maze-sample-8x8-v0" "maze-sample-9x9-v0" "maze-sample-10x10-v0"
do
    echo $env
    for (( seed = 0 ; seed < $1 ; ++seed ))
    do
        echo "Seed: $seed"
        python train.py --env $env --algo ppo --seed $seed --verbose 0 --gym-packages gym_maze --conf-file hyperparams/ssp/ppo_ssp.yml --track --wandb-project-name ssp-rl --wandb-entity nicole-s-dumont --wandb-tags $env ssp-obs
    	python train.py --env $env --algo ppo --seed $seed --verbose 0 --gym-packages gym_maze --conf-file hyperparams/ppo.yml  --track --wandb-project-name ssp-rl --wandb-entity nicole-s-dumont --wandb-tags $env default-obs 
    done
done