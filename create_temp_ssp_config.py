import argparse
import yaml
import gymnasium as gym
import minigrid
import numpy as np

def create_yml(args):
    env = gym.make(args.env)
    pos_ls = np.min([env.unwrapped.height,env.unwrapped.width])/10.
    dir_ls = 4./10.
    if args.type=='view':
        content = {
            args.env: {
                "env_wrapper": [
                    {
                        "hrr_gym_wrappers.PrepMiniGridViewFlatWrapper": {
                            "shape_out": args.ssp_dim
                        }
                    }
                ],
                "n_envs": 8,
                "n_timesteps": float(args.n),
                "policy": "MlpPolicy",
                "n_steps": 128,
                "batch_size": 64,
                "gae_lambda": 0.95,
                "gamma": 0.99,
                "n_epochs": 10,
                "ent_coef": 0.0,
                "learning_rate": 2.5e-4,
                "clip_range": 0.2,
                "policy_kwargs": f"dict(features_extractor_class=hrr_gym_wrappers.SSPMiniGridViewProcesser, features_extractor_kwargs=dict(features_dim={args.ssp_dim}, ssp_h=[{pos_ls},{pos_ls},{dir_ls}]))"
            }
        }
    elif args.type=='view2':
        content = {
            args.env: {
                "env_wrapper": [
                    {
                        "hrr_gym_wrappers.PrepMiniGridViewFlatWrapper2": {
                            "shape_out": args.ssp_dim
                        }
                    }
                ],
                "n_envs": 8,
                "n_timesteps": float(args.n),
                "policy": "MlpPolicy",
                "n_steps": 128,
                "batch_size": 64,
                "gae_lambda": 0.95,
                "gamma": 0.99,
                "n_epochs": 10,
                "ent_coef": 0.0,
                "learning_rate": 2.5e-4,
                "clip_range": 0.2,
                "policy_kwargs": f"dict(features_extractor_class=hrr_gym_wrappers.SSPMiniGridViewProcesser, features_extractor_kwargs=dict(features_dim={args.ssp_dim}, ssp_h=[{pos_ls},{pos_ls},{dir_ls}]))"
            }
        }
    elif args.type=='xy':
        content = {
            args.env: {
                "env_wrapper": [
                    "hrr_gym_wrappers.MiniGridXYFlatWrapper"
                ],
                "n_envs": 8,
                "n_timesteps": float(args.n),
                "policy": "MlpPolicy",
                "n_steps": 128,
                "batch_size": 64,
                "gae_lambda": 0.95,
                "gamma": 0.99,
                "n_epochs": 10,
                "ent_coef": 0.0,
                "learning_rate": 2.5e-4,
                "clip_range": 0.2,
                "policy_kwargs": f"dict(features_extractor_class=hrr_gym_wrappers.SSPProcesser, features_extractor_kwargs=dict(features_dim={args.ssp_dim}, ssp_h=[{pos_ls},{pos_ls},{dir_ls}]))"
            }
        }
    else:
        content = {
            args.env: {
                "env_wrapper": [
                    "minigrid.wrappers.FlatObsWrapper"
                ],
                "n_envs": 8,
                "n_timesteps": float(args.n),
                "policy": "MlpPolicy",
                "n_steps": 128,
                "batch_size": 64,
                "gae_lambda": 0.95,
                "gamma": 0.99,
                "n_epochs": 10,
                "ent_coef": 0.0,
                "learning_rate": 2.5e-4,
                "clip_range": 0.2
            }
        }
        
        
    # if (args.env=="MiniGrid-FourRooms-v0") :
    #     if (args.type == "default"):
    #         content[args.env].update( { 'n_steps': 512 })
    #         content[args.env]['policy_kwargs'] = "dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))"
        
    #     else:
    #         content[args.env].update( {
    #                 'batch_size': 16,
    #                 'n_steps': 512,
    #                 'gamma': 0.995,
    #                 'learning_rate': 0.0002667131432316635,
    #                 'ent_coef': 0.00014739249767857526,
    #                 'clip_range': 0.1,
    #                 'n_epochs': 10,
    #                 'gae_lambda': 0.99,
    #                 'max_grad_norm': 0.6,
    #                 'vf_coef': 0.4834382740594195} )
    #         content[args.env]['policy_kwargs'] = content[args.env]['policy_kwargs'][:-1] + ", activation_fn=nn.ReLU, net_arch=dict(pi=[256, 256], vf=[256, 256]))"
        

    # Save the content to a YAML file named "temp.yml"
    with open("temp.yml", "w") as file:
        yaml.dump(content, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a YAML config file.")
    parser.add_argument('--env', type=str, required=True, help="The environment name.")
    parser.add_argument('--n',  required=True, help="Number of timesteps.")
    parser.add_argument('--ssp-dim', type=int, required=False, help="Dimension of SSP.")
    parser.add_argument('--type', type=str)

    args = parser.parse_args()
    
    create_yml(args)
