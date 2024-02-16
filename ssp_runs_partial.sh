for env in "Acrobot-v1"
do
    for (( seed = 0 ; seed < $1 ; ++seed ))
    do
        echo "Seed: $seed"
        python train.py --env $env --algo ppo --seed $seed --verbose 0 --gym-packages hrr_gym_wrappers --conf-file hyperparams/ssp/ppo_ssp.yml --track --wandb-project-name ssp-rl --wandb-entity nicole-s-dumont --wandb-tags $env new-ssp-obs
    done
done

