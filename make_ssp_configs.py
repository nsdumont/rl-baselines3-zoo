
import yaml

import gymnasium as gym

algos = ['ppo', 'a2c', 'dqn','qrdqn','sac','td3','tqc','trpo']
keep_envs = ["CartPole-v1", "Pendulum-v1", "MountainCar-v0" ,"MountainCarContinuous-v0", "LunarLander-v2","LunarLanderContinuous-v2", "Acrobot-v1"]
# len_scale=[array([0.19399145]),
#  array([0.03160698]),
#  array([0.03160698]),
#  array([0.03160698]),
#  array([0.19854607]),
#  array([0.23778243]),
#  array([0.03160698])]

if False:
    for algo in algos:
        with open("hyperparams/" + algo + ".yml", 'r') as stream:
            hyperparams = yaml.safe_load(stream)
            
        kremove=[]
        for k in hyperparams.keys():
            if not(k in keep_envs):
                kremove.append(k)
                continue
            env=gym.make(k)
            d = env.observation_space.shape[0]
            ssp_dim = 2*(d+1)*(5**2) + 1
            # hyperparams[k]['gym-packages']="hrr_gym_wrappers"
            hyperparams[k]['env_wrapper']=[{"hrr_gym_wrappers.SSPObsWrapper": {"shape_out": ssp_dim }}]
        
        for k in kremove:
            hyperparams.pop(k, None);
            
        with open('hyperparams/learnssp/' + algo + '_ssp.yml', 'w') as outfile:
            yaml.dump(hyperparams, outfile, default_flow_style=False)
else:
    for algo in algos:
        with open("hyperparams/ssp/" + algo + "_ssp.yml", 'r') as stream:
            hyperparams = yaml.safe_load(stream)
            
        kremove=[]
        for k in hyperparams.keys():
            if k in keep_envs:
                if 'policy_kwargs' in hyperparams[k]:
                    existing_policy = hyperparams[k]['policy_kwargs'][:-1]
                else:
                    existing_policy = ''
                ssp_dim = hyperparams[k]['env_wrapper'][0]['hrr_gym_wrappers.SSPObsWrapper']['shape_out']
                if 'length_scale' in hyperparams[k]['env_wrapper'][0]['hrr_gym_wrappers.SSPObsWrapper']:
                    ls = hyperparams[k]['env_wrapper'][0]['hrr_gym_wrappers.SSPObsWrapper']['length_scale']
                    hyperparams[k]['policy_kwargs']= existing_policy+\
                        ',features_extractor_class=hrr_gym_wrappers.SSPProcesser,features_extractor_kwargs=dict(features_dim=' + str(ssp_dim) + ',ssp_h='\
                            + str(ls) + '))'
                else:
                    hyperparams[k]['policy_kwargs']=existing_policy+\
                        ',features_extractor_class=hrr_gym_wrappers.SSPProcesser,features_extractor_kwargs=dict(features_dim=' + str(ssp_dim) + '))'
                hyperparams[k].pop('env_wrapper', None);
            else:
                kremove.append(k)
                continue
            
        
        for k in kremove:
            hyperparams.pop(k, None);
            
        with open('hyperparams/learnssp/' + algo + '_ssp.yml', 'w') as outfile:
            yaml.dump(hyperparams, outfile, default_flow_style=False)