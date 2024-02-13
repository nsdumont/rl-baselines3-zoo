import argparse
import importlib
import os
import sys

import numpy as np
import torch as th
import yaml
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.callbacks import tqdm
from stable_baselines3.common.utils import set_random_seed

import rl_zoo3.import_envs  # noqa: F401 pylint: disable=unused-import
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.load_from_hub import download_from_hub
from rl_zoo3.utils import StoreDict, get_model_path

# from sklearn.neighbors import KernelDensity
from sklearn.gaussian_process import GaussianProcessRegressor 

import hrr_gym_wrappers
import gymnasium as gym


from sklearn.gaussian_process.kernels import Hyperparameter, Kernel, StationaryKernelMixin, NormalizedKernelMixin, _check_length_scale

def d_sinc(x):
    x = np.asanyarray(x)
    y = np.where(x == 0, 1.0e-20, x)
    return (np.cos(np.pi * y)/y) - (np.sin(np.pi * y)/(np.pi * y**2))

class SincKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    '''
    A sinc-function kernel. 

    The SincKernel is a stationary kernel.  It is parameterized by a
    length scale parameter 

    Parameters:
    -----------
    length_scale: float or ndarray of shape (n_features,), default=1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used.  If an array, an anisotropic kernel is used where each 
        dimension of l defines the length-scale of the respective feature
        dimensions

    length_scale_bounds: pair of floats >= 0 or 'fixed', default=(1e-5,1e5)
        The lower and upper bound of 'length_scale'. If set to 'fixed', 
        length_scale cannot be changed during hyperparameter tuning.
    '''

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5,1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter(
                'length_scale', 
                'numeric', 
                self.length_scale_bounds, 
                len(self.length_scale)
            )
        return Hyperparameter('length_scale', 'numeric', self.length_scale_bounds)
    def __call__(self, X, Y=None, eval_gradient=False):
        '''
        Return the kernel k(X,Y) and optimally it's gradient

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
        Y : ndarray of shape (n_samples_Y, n_features). If None, k(X,X) 
            evaluated instead.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims)
        '''

        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)

        if Y is None:
            dists = (X[:,None,:] - X[None,:,:]) / length_scale
            K = np.prod(np.sinc(dists), axis=-1)
        else:
            if eval_gradient:
                raise ValueError('Gradient can only be evaluted when Y is None')
            dists = (X[:,None,:] - Y[None,:,:]) / length_scale
            K = np.prod(np.sinc(dists), axis=-1)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = np.zeros((X.shape[0], 
                                       X.shape[0], 
                                       1))
                sinc_mat = np.sinc(dists) 
                d_sinc_mat = d_sinc(dists)
                for x_idx in range(X.shape[1]):
                    temp = np.prod(sinc_mat[:,:,:x_idx], axis=-1)
                    temp *= np.prod(sinc_mat[:,:,x_idx+1:], axis=-1)
                    temp *= d_sinc_mat[:,:,x_idx] * (-dists[:,:,x_idx]/(length_scale**2))
                    K_gradient[:,:,0] += temp
                return K, K_gradient
            elif self.anisotropic:
                # we need to compute the pairwise dimension-wise distances.
                K_gradient = np.zeros((X.shape[0],
                                       X.shape[0],
                                       len(length_scale)))
                sinc_mat = np.sinc(dists)
                d_sinc_mat = d_sinc(dists)
                for l_idx in range(len(length_scale)):
                    K_gradient[:,:,l_idx] = np.prod(sinc_mat[:,:,:l_idx], axis=-1)
                    K_gradient[:,:,l_idx] *= np.prod(sinc_mat[:,:,l_idx+1:], axis=-1)
                    K_gradient[:,:,l_idx] *= d_sinc_mat[:,:,l_idx]
                    K_gradient[:,:,l_idx] *= -dists[:,:,l_idx]/(length_scale[l_idx]**2)
                ### end for
                return K, K_gradient
            ### end if

        else:
            return K
    ### end __call__

    def __repr__(self):
        if self.anisotropic:
            return '{0}(length_scale=[{1}]'.format(
                self.__class__.__name__,
                ', '.join(map('{0:.3g}'.format, self.length_scale)),
            )
        else:
            return '{0}(length_scale={1:.3g}'.format(
               self.__class__.__name__, np.ravel(self.length_scale)[0]
            )

keep_envs = ["CartPole-v1", "Pendulum-v1", "MountainCar-v0" ,"MountainCarContinuous-v0", "LunarLander-v2","LunarLanderContinuous-v2", "Acrobot-v1"]
len_scales = []
# array([0.19399145]),
#  array([0.03160698]),
#  array([0.03160698]),
#  array([0.03160698]),
#  array([0.19854607]),
#  array([0.23778243]),
#  array([0.03160698])]
    
# with open("hyperparams/a2c.yml", 'r') as stream:
#     hyperparams = yaml.safe_load(stream)
    
# 
# for k in hyperparams.keys():
#     if not(k in keep_envs):
#         hyperparams.pop(k, None);
#         continue
#     env=gym.make(k)
#     d = env.observation_space.shape[0]
#     ssp_dim = 2*(d+1)*(5**2) + 1
#     hyperparams[k]['env_wrapper']=[{"hrr_gym_wrappers.SSPObsWrapper": {"shape_out": ssp_dim, }}]
    
timesteps = 1000
algo = 'a2c'
folder = 'rl-trained-agents'
for env_name in  keep_envs:
    
    _, model_path, log_path = get_model_path(
        0,
        folder,
        algo,
        env_name,
    )
    
    # Off-policy algorithm only support one env for now
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]
    
    seed=0
    set_random_seed(seed)
    
    
    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=True)
    
    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    
    log_dir = None
    
    env = create_test_env(
        env_name,
        n_envs=1,
        stats_path=maybe_stats_path,
        seed=seed,
        log_dir=log_dir,
        should_render=False,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )
    d= env.observation_space.shape[0]
    
    kwargs = dict(seed=0)
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))
        # Hack due to breaking change in v1.6
        # handle_timeout_termination cannot be at the same time
        # with optimize_memory_usage
        if "optimize_memory_usage" in hyperparams:
            kwargs.update(optimize_memory_usage=False)
    
    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8
    
    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
    
    if "HerReplayBuffer" in hyperparams.get("replay_buffer_class", ""):
        kwargs["env"] = env
    
    model = ALGOS[algo].load(model_path, custom_objects=custom_objects, device="auto", **kwargs)
    obs = env.reset()
    
    stochastic = False
    deterministic = not stochastic
    
    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    # For HER, monitor success rate
    successes = []
    lstm_states = None
    episode_start = np.ones((env.num_envs,), dtype=bool)
    
    generator = range(timesteps)
    obss = np.zeros((timesteps, d))
    rs = np.zeros((timesteps,1))
    istart=0
    for i in generator:
        action, lstm_states = model.predict(
            obs,  # type: ignore[arg-type]
            state=lstm_states,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        obs, reward, done, infos = env.step(action)
    
        if done:
            env.reset()
            istart=1
    
        obss[i] = obs
        rs[istart:i] += reward
       
    
    fit_gp = GaussianProcessRegressor(
                kernel=SincKernel(
                    length_scale_bounds=(
                        1/np.sqrt(obss.shape[0]+1), 
                        1e5)
                    ),
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=20,
                random_state=0,
            )
    fit_gp.fit(obss, rs)
    lenscale = np.exp(fit_gp.kernel_.theta)
    len_scales.append(lenscale)
