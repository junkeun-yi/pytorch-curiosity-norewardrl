import numpy as np
from tensorboardX import SummaryWriter
from torch.multiprocessing import Pipe

from agents import *
from config import *
from envs import *
from utils import *


def main():
    # reading config for parameters
    print({section: dict(config[section]) for section in config.sections()})
    train_method = default_config['TrainMethod']
    env_id = default_config['EnvID']
    env_type = default_config['EnvType']

    if env_type == 'mario':
        # default way of creating a mario environment in gym
        env = JoypadSpace(gym_super_mario_bros.make(env_id), COMPLEX_MOVEMENT)
    elif env_type == 'atari' or env_type == 'mujoco':
        # atari and mujoco share the same initialization
        env = gym.make(env_id)
    else:
        raise NotImplementedError
    # TODO: explain what the bottom few lines d
    # input size: observation space shape of the input
    #   because the input is TODO:
    # output_size: action space size of the input
    #   because the output is TODO:
    input_size = env.observation_space.shape  # 4
    if env_type != 'mujoco':
        # discrete action space
        output_size = env.action_space.n  # 2
    else:
        # continuous action space
        # action space for continuous is a torch box, does not have explicit action_space.n like discrete
        # TODO: fix this ? not sure what this is supposed to be
        output_size = 1

    if 'Breakout' in env_id:
        output_size -= 1

    # close the viewers
    env.close()

    # TODO: change load_model and is_render to be configurable in config.conf
    is_load_model = False
    
    if default_config['Discrete'] == "True"
        discrete = True
    else:
        discrete = False

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

    # mean and standard of obseravtions, in order to normalize future observations
    # TODO: make this a parameter
    # why do we need to normalize the observations ?
    # the normalizing is to make sure that the observations from the parallel workers are aggregated correctly ?
    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 4, 84, 84))

    # number of steps used to gain normalizing constants for observations
    pre_obs_norm_step = int(default_config['ObsNormStep'])
    discounted_reward = RewardForwardFilter(gamma)

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
        work = env_type(env_id, is_render, idx, child_conn) # TODO: change to take in a dictionary of other configurable params; kwargs ?
        work.start() # start the environment PROCESS
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    # TODO: chage so not hardcoded history_size, height, width
    states = np.zeros([num_worker, 4, 84, 84])

    # TODO: explain all variables below
    sample_episode = 0 #
    sample_rall = 0 # rall = overall reward
    sample_step = 0 # 
    sample_env_idx = 0 # not changed, what does it do ?
    sample_i_rall = 0 # sampled reward for worker i
    global_update = 0 # number of iterations taken, incremented when all workers finish a step
    global_step = 0 # number of steps taken overall, aggregated over all workers

    # TODO: should have this saved somewhere else, or at least not run every time.
    # Also, should not visualize this part of the environment ...
    # normalize obs
    print('Start to initailize observation normalization parameter.....')
    next_obs = []
    steps = 0
    while steps < pre_obs_norm_step:
        # for pre_obs_norm_steps, run the environment and receive observations.
        # these observations are used to determine mean/std of state in order to normalize it.
        steps += num_worker
        # discrite case
        # TODO: make one for the continuous case.
        actions = np.random.randint(0, output_size, size=(num_worker,))

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        for parent_conn in parent_conns:
            s, r, d, rd, lr = parent_conn.recv()
            next_obs.append(s[:])

    next_obs = np.stack(next_obs)
    obs_rms.update(next_obs) # update observation running mean startdard
    print('End to initalize...')

    # playing loop
    # TODO: clean this up into a function in the agent instead of if statements
    if train_method == "ICM":
        # playing loop
        while True:
            total_state, total_reward, total_done, total_next_state, total_action, total_int_reward, total_next_obs, total_values, total_policy = \
                [], [], [], [], [], [], [], [], []
            global_step += (num_worker * num_step) # increase global step
            global_update += 1 # global 

            # Step 1. n-step rollout
            for _ in range(num_step):
                # get new action from normalized observations
                actions, value, policy = agent.get_action((states - obs_rms.mean) / np.sqrt(obs_rms.var))

                # send action to every worker
                for parent_conn, action in zip(parent_conns, actions):
                    parent_conn.send(action) # see envs.py, search for child_conn.recv

                # receive the next state, observations, and metrics from the workers
                next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
                for parent_conn in parent_conns:
                    s, r, d, rd, lr = parent_conn.recv()
                    next_states.append(s)
                    rewards.append(r)
                    dones.append(d)
                    real_dones.append(rd)
                    log_rewards.append(lr)

                # concatenate the observations
                next_states = np.stack(next_states)
                rewards = np.hstack(rewards)
                dones = np.hstack(dones)
                real_dones = np.hstack(real_dones)

                # curiosity:
                # total reward = intrinsic reward
                # TODO: add hyperparameter for normalization or not
                intrinsic_reward = agent.compute_intrinsic_reward(
                    (states - obs_rms.mean) / np.sqrt(obs_rms.var),
                    (next_states - obs_rms.mean) / np.sqrt(obs_rms.var),
                    actions)
                sample_i_rall += intrinsic_reward[sample_env_idx]

                # TODO: add in extrinsic reward for no curiosity teacher. TODO: make it a hyperparameter.

                total_int_reward.append(intrinsic_reward)
                total_state.append(states)
                total_next_state.append(next_states)
                total_reward.append(rewards)
                total_done.append(dones)
                total_action.append(actions)
                total_values.append(value)
                total_policy.append(policy)

                states = next_states[:, :, :, :]

                # actual rewards for state
                sample_rall += log_rewards[sample_env_idx]

                sample_step += 1
                if real_dones[sample_env_idx]:
                    sample_episode += 1
                    writer.add_scalar('data/reward_per_epi', sample_rall, sample_episode)
                    writer.add_scalar('data/reward_per_rollout', sample_rall, global_update)
                    writer.add_scalar('data/step', sample_step, sample_episode)
                    sample_rall = 0
                    sample_step = 0
                    sample_i_rall = 0

            # calculate last next value
            _, value, _ = agent.get_action((states - obs_rms.mean) / np.sqrt(obs_rms.var))
            total_values.append(value)
            # --------------------------------------------------

            total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84]) # what is this ????
            total_next_state = np.stack(total_next_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
            total_action = np.stack(total_action).transpose().reshape([-1])
            total_done = np.stack(total_done).transpose()
            total_values = np.stack(total_values).transpose()
            total_logging_policy = torch.stack(total_policy).view(-1, output_size).cpu().numpy()

            # Step 2. calculate intrinsic reward
            # running mean intrinsic reward
            total_int_reward = np.stack(total_int_reward).transpose()
            total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                            total_int_reward.T])
            mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
            reward_rms.update_from_moments(mean, std ** 2, count)

            # normalize intrinsic reward
            total_int_reward /= np.sqrt(reward_rms.var)
            writer.add_scalar('data/int_reward_per_epi', np.sum(total_int_reward) / num_worker, sample_episode)
            writer.add_scalar('data/int_reward_per_rollout', np.sum(total_int_reward) / num_worker, global_update)
            # -------------------------------------------------------------------------------------------

            # logging Max action probability
            writer.add_scalar('data/max_prob', softmax(total_logging_policy).max(1).mean(), sample_episode)

            # Step 3. make target and advantage
            target, adv = make_train_data(total_int_reward,
                                        np.zeros_like(total_int_reward),
                                        total_values,
                                        gamma,
                                        num_step,
                                        num_worker)

            adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
            # -----------------------------------------------

            # Step 5. Training!
            agent.train_model((total_state - obs_rms.mean) / np.sqrt(obs_rms.var),
                            (total_next_state - obs_rms.mean) / np.sqrt(obs_rms.var),
                            target, total_action,
                            adv,
                            total_policy)

            if global_step % (num_worker * num_step * 100) == 0:
                print('Now Global Step :{}'.format(global_step))
                torch.save(agent.model.state_dict(), model_path)
                torch.save(agent.icm.state_dict(), icm_path)
    elif train_method == "AC":
        while True:
            total_state, total_reward, total_done, total_next_state, total_action, total_int_reward, total_next_obs, total_values, total_policy = \
                [], [], [], [], [], [], [], [], []
            global_step += (num_worker * num_step) # increase global step
            global_update += 1 # global 

            # Step 1. n-step rollout
            for _ in range(num_step):
                # get new action from normalized observations
                actions, value, policy = agent.get_action((states - obs_rms.mean) / np.sqrt(obs_rms.var))

                print(actions)

                # send action to every worker
                for parent_conn, action in zip(parent_conns, actions):
                    parent_conn.send(action) # see envs.py, search for child_conn.recv

                # receive the next state, observations, and metrics from the workers
                next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
                for parent_conn in parent_conns:
                    s, r, d, rd, lr = parent_conn.recv()
                    next_states.append(s)
                    rewards.append(r)
                    dones.append(d)
                    real_dones.append(rd)
                    log_rewards.append(lr)

                # concatenate the observations
                next_states = np.stack(next_states)
                rewards = np.hstack(rewards)
                dones = np.hstack(dones)
                real_dones = np.hstack(real_dones)

                # curiosity:
                # total reward = intrinsic reward
                # TODO: add hyperparameter for normalization or not
                # intrinsic_reward = agent.compute_intrinsic_reward(
                #     (states - obs_rms.mean) / np.sqrt(obs_rms.var),
                #     (next_states - obs_rms.mean) / np.sqrt(obs_rms.var),
                #     actions)
                # sample_i_rall += intrinsic_reward[sample_env_idx]

                # total_int_reward.append(intrinsic_reward)
                total_state.append(states)
                total_next_state.append(next_states)
                total_reward.append(rewards)
                total_done.append(dones)
                total_action.append(actions)
                total_values.append(value)
                total_policy.append(policy)

                states = next_states[:, :, :, :]

                # actual rewards for state
                sample_rall += log_rewards[sample_env_idx]

                sample_step += 1
                if real_dones[sample_env_idx]:
                    sample_episode += 1
                    writer.add_scalar('data/reward_per_epi', sample_rall, sample_episode)
                    writer.add_scalar('data/reward_per_rollout', sample_rall, global_update)
                    writer.add_scalar('data/step', sample_step, sample_episode)
                    sample_rall = 0
                    sample_step = 0
                    # sample_i_rall = 0

            # calculate last next value
            _, value, _ = agent.get_action((states - obs_rms.mean) / np.sqrt(obs_rms.var))
            total_values.append(value)
            # --------------------------------------------------

            total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84]) # what is this ????
            total_next_state = np.stack(total_next_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
            total_action = np.stack(total_action).transpose().reshape([-1])
            total_done = np.stack(total_done).transpose()
            total_values = np.stack(total_values).transpose()
            total_logging_policy = torch.stack(total_policy).view(-1, output_size).cpu().numpy()

            # Step 2. calculate intrinsic reward
            # running mean intrinsic reward
            # total_int_reward = np.stack(total_int_reward).transpose()
            # total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
            #                                 total_int_reward.T])
            total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                            total_reward])
            mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
            reward_rms.update_from_moments(mean, std ** 2, count)

            # normalize intrinsic reward
            # total_int_reward /= np.sqrt(reward_rms.var)
            # writer.add_scalar('data/int_reward_per_epi', np.sum(total_int_reward) / num_worker, sample_episode)
            # writer.add_scalar('data/int_reward_per_rollout', np.sum(total_int_reward) / num_worker, global_update)
            # -------------------------------------------------------------------------------------------

            # logging Max action probability
            writer.add_scalar('data/max_prob', softmax(total_logging_policy).max(1).mean(), sample_episode)

            # Step 3. make target and advantage
            # target, adv = make_train_data(total_int_reward,
            #                             np.zeros_like(total_int_reward),
            #                             total_values,
            #                             gamma,
            #                             num_step,
            #                             num_worker)
            target, adv = make_train_data(np.array(total_reward),
                                        np.zeros_like(total_reward),
                                        total_values,
                                        gamma,
                                        num_step,
                                        num_worker)

            # normalize advantage
            adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
            # -----------------------------------------------

            # Step 5. Training!
            agent.train_model((total_state - obs_rms.mean) / np.sqrt(obs_rms.var),
                            (total_next_state - obs_rms.mean) / np.sqrt(obs_rms.var),
                            target, total_action,
                            adv,
                            total_policy)

            if global_step % (num_worker * num_step * 100) == 0:
                print('Now Global Step :{}'.format(global_step))
                torch.save(agent.model.state_dict(), model_path)
                torch.save
    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()
