for env in "maze-random-5x5-v0" "maze-random-6x6-v0" "maze-sample-7x7-v0" "maze-sample-8x8-v0" "maze-sample-9x9-v0" "maze-sample-9x9-v0"
do
    python train.py --env $env --algo dqn --seed 1 --verbose 0 --gym-packages gym_maze --conf-file hyperparams/ssp/dqn_ssp.yml --track --wandb-project-name ssp-rl --wandb-entity nicole-s-dumont --wandb-tags $env ssp-obs
	python train.py --env $env --algo dqn --seed 1 --verbose 0 --gym-packages gym_maze --conf-file hyperparams/dqn.yml  --track --wandb-project-name ssp-rl --wandb-entity nicole-s-dumont --wandb-tags $env default-obs 
done