
for (( seed = 1 ; seed < $1 ; ++seed ))
do
    echo "Seed: $seed"
    python train.py --env MountainCarContinuous-v0 --algo sac --seed $seed --verbose 0 --conf-file  hyperparams/ssp/sac_ssp.yml --track --wandb-project-name ssp-rl --wandb-entity nicole-s-dumont --wandb-tags $env ssp-obs sac
done

