for env in "LunarLanderContinuous-v2"
do
    for (( seed = 0 ; seed < $1 ; ++seed ))
    do
        echo "Seed: $seed"
        python train.py --env $env --algo sac --seed $seed --verbose 0 --gym-packages hrr_gym_wrappers --conf-file hyperparams/ssp/sac_ssp.yml --track --wandb-project-name ssp-rl --wandb-entity nicole-s-dumont --wandb-tags $env ssp-obs
        python train.py --env $env --algo sac --seed $seed --verbose 0 --conf-file hyperparams/sac.yml --track --wandb-project-name ssp-rl --wandb-entity nicole-s-dumont --wandb-tags $env default-obs
    done
done

