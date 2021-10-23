import numpy as np
from tensorboardX import SummaryWriter
from torch.multiprocessing import Pipe

from agents import *
from config import *
from envs import *
from utils import *

import rl_trainer


def main():
    # reading config for parameters
    print({section: dict(config[section]) for section in config.sections()})
    train_method = default_config['TrainMethod']
    env_id = default_config['EnvID']
    env_type = default_config['EnvType']

    # discrete or continuous action space
    if default_config['Discrete'] == "True":
        discrete = True
    else:
        discrete = False

    if env_type == 'mario':
        # default way of creating a mario environment in gym
        env = JoypadSpace(gym_super_mario_bros.make(env_id), COMPLEX_MOVEMENT)
    elif env_type == 'atari' or env_type == 'mujoco':
        # atari and mujoco share the same initialization
        env = gym.make(env_id)
    else:
        raise NotImplementedError
        
    # input size: observation space shape of the input
    # output_size: action space size of the input
    input_size = env.observation_space.shape  # 4
    output_size = env.action_space.n if discrete else env.action_space.shape[0]

    if 'Breakout' in env_id:
        output_size -= 1

    # close the viewers
    env.close()

    # TODO: change load_model and is_render to be configurable in config.conf
    is_load_model = False

    # visualize the environment
    if default_config['Render'] == "True":
        is_render = True
    else:
        is_render = False
    
    # paths for models
    # TODO: figure out what this is saving
    model_path = 'models/{}.model'.format(env_id)
    icm_path = 'models/{}.icm'.format(env_id)

    # tensorboard log writer
    writer = SummaryWriter()

    # config files
    use_cuda = default_config.getboolean('UseGPU')
    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet') # reduce overfitting ?

    # lam: lambda; GAE discount factor
    lam = float(default_config['Lambda'])
    # num_workers: number of workers for RL agent
    #   runs {num_worker} instances of the environment for trajectory collection
    num_worker = int(default_config['NumEnv'])

    histsize = int(default_config['StateStackSize'])
    pph = int(default_config['PreProcHeight'])
    ppw = int(default_config['PreProcWidth'])

    # number of steps per rollout
    num_step = int(default_config['NumStep'])

    # PPO epsilon. TODO: does this work for continuous action spaces
    ppo_eps = float(default_config['PPOEps'])
    # number of epochs
    epoch = int(default_config['Epoch'])
    # mini_batch: determines batch size
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    # learning rate
    learning_rate = float(default_config['LearningRate'])
    # TODO: where is this used
    entropy_coef = float(default_config['Entropy'])
    # discount factor
    gamma = float(default_config['Gamma'])
    # Curiosity parameter; scaling factor for curiosity intrinsic loss.
    eta = float(default_config['ETA'])

    # TODO: what is this
    clip_grad_norm = float(default_config['ClipGradNorm'])

    # number of steps used to gain normalizing constants for observations
    pre_obs_norm_step = int(default_config['ObsNormStep'])

    # method for training: TODO: actor critic and distillation
    if train_method == "ICM":
        agent = ICMAgent
    elif train_method == "AC":
        agent = ACAgent
    else:
        raise NotImplementedError

    # switch for creating different environments
    # each type of environment has a defined initializer in `envs.py`
    # TODO: need to change so that each environment can switch reward functions
    if default_config['EnvType'] == 'atari':
        env_type = AtariEnvironment
    elif default_config['EnvType'] == 'mario':
        env_type = MarioEnvironment
    elif default_config['EnvType'] == 'mujoco':
        env_type = MujocoEnvironment
    else:
        raise NotImplementedError

    # creating the agent
    agent = agent(
        input_size,
        output_size,
        discrete,
        num_worker,
        num_step,
        gamma,
        lam=lam,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        clip_grad_norm=clip_grad_norm,
        epoch=epoch,
        batch_size=batch_size,
        ppo_eps=ppo_eps,
        eta=eta,
        use_cuda=use_cuda,
        use_gae=use_gae,
        use_noisy_net=use_noisy_net
    )

    # load a previously created model from model_path
    if is_load_model:
        if use_cuda:
            agent.model.load_state_dict(torch.load(model_path))
        else:
            agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # create distributed RL workers
    # parent connection: controller
    # child connection: worker
    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe() # creates separate process for work.
        # creates an environment, which is a python PROCESS;
        work = env_type(env_id, is_render, idx, child_conn, history_size=histsize, h=pph, w=ppw) # TODO: change to take in a dictionary of other configurable params; kwargs ?
        work.start() # start the environment PROCESS
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    # TODO: chage so not hardcoded history_size, height, width
    states = np.zeros([num_worker, histsize, pph, ppw])

    # TODO: explain all variables below
    sample_episode = 0 #
    sample_rall = 0 # rall = overall reward
    sample_step = 0 # 
    sample_env_idx = 0 # not changed, what does it do ?
    sample_i_rall = 0 # sampled reward for worker i
    global_update = 0 # number of iterations taken, incremented when all workers finish a step
    global_step = 0 # number of steps taken overall, aggregated over all workers

    # get observation normalizing parameters
    obs_rms = rl_trainer.get_env_normalize_params(pre_obs_norm_step, 
                                        output_size, 
                                        num_worker, 
                                        parent_conns,
                                        histsize,
                                        pph,
                                        ppw
                                        )

    if train_method == "ICM":
        rl_trainer.train_icm(
            writer,
            agent,
            obs_rms,
            parent_conns,
            output_size,
            num_worker,
            histsize,
            pph,
            ppw,
            num_step,
            gamma,
            model_path,
            icm_path
        )

    if train_method == "AC":
        rl_trainer.train_ac(
            writer,
            agent,
            obs_rms,
            parent_conns,
            output_size,
            num_worker,
            histsize,
            pph,
            ppw,
            num_step,
            gamma,
            model_path
        )

if __name__ == '__main__':
    main()
