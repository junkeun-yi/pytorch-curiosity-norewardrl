import numpy as np
from tensorboardX import SummaryWriter
from torch.multiprocessing import Pipe

from agents import *
from utils import *

# find mean and std of observations to normalize.
def get_env_normalize_params(pre_obs_norm_step, 
                            output_size, 
                            num_worker, 
                            parent_conns,
                            histsize,
                            pph,
                            ppw):
    obs_rms = RunningMeanStd(shape=(1, histsize, pph, ppw))
    print('Start to initailize observation normalization parameter.....')
    next_obs = []
    steps = 0
    while steps < pre_obs_norm_step:
        # for pre_obs_norm_steps, run the environment and receive observations.
        # these observations are used to determine mean/std of state in order to normalize it.
        steps += num_worker
        # discrete case
        actions = np.random.randint(0, output_size, size=(num_worker,))

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        for parent_conn in parent_conns:
            s, r, d, rd, lr = parent_conn.recv()
            next_obs.append(s[:])

    next_obs = np.stack(next_obs)
    obs_rms.update(next_obs) # update observation running mean startdard
    print('End to initalize...')
    return obs_rms


def train_icm(writer, agent, obs_rms, parent_conns, output_size, num_worker, histsize, pph, ppw, num_step, gamma, model_path, icm_path):
    sample_episode = 0 #
    sample_rall = 0 # rall = overall reward
    sample_step = 0 # 
    sample_env_idx = 0 # not changed, what does it do ?
    sample_i_rall = 0 # sampled reward for worker i
    global_update = 0 # number of iterations taken, incremented when all workers finish a step
    global_step = 0 # number of steps taken overall, aggregated over all workers

    states = np.zeros([num_worker, histsize, pph, ppw])

    reward_rms = RunningMeanStd()
    discounted_reward = RewardForwardFilter(gamma)

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

        print(f"{num_step}-step rollout done")

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
        target, adv = make_train_data(np.array(total_reward) + np.transpose(np.array(total_int_reward)),
                                    np.zeros_like(total_reward), #np.zeros_like(total_int_reward),
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


def train_ac(writer, agent, obs_rms, parent_conns, output_size, num_worker, histsize, pph, ppw, num_step, gamma, model_path):
    sample_episode = 0 #
    sample_rall = 0 # rall = overall reward
    sample_step = 0 # 
    sample_env_idx = 0 # not changed, what does it do ?
    sample_i_rall = 0 # sampled reward for worker i
    global_update = 0 # number of iterations taken, incremented when all workers finish a step
    global_step = 0 # number of steps taken overall, aggregated over all workers

    states = np.zeros([num_worker, histsize, pph, ppw])

    reward_rms = RunningMeanStd()
    discounted_reward = RewardForwardFilter(gamma)

    # training_loop
    while True:
        total_state, total_reward, total_done, total_next_state, total_action, total_int_reward, total_next_obs, total_values, total_policy = \
            [], [], [], [], [], [], [], [], []
        global_step += (num_worker * num_step) # increase global step
        global_update += 1 # global 

        # n_step rollout
        for _ in range(num_step):
            # get new action from normalized observations
            actions, value, policy = agent.get_action((states - obs_rms.mean) / np.sqrt(obs_rms.var))

            # send action to every worker
            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)
            
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
            
        print(f"{num_step}-step rollout done")

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

        total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                        total_reward])
        mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
        reward_rms.update_from_moments(mean, std ** 2, count)

        # log max action probability
        writer.add_scalar('data/max_prob', softmax(total_logging_policy).max(1).mean(), sample_episode)

        """
        target, adv = make_train_data2(np.array(total_reward),
                        np.zeros_like(total_reward),
                        total_values,
                        gamma,
                        num_step,
                        num_worker)
        """

        print(total_reward)
        print(total_values)

        # """
        target, adv = make_train_data(np.array(total_reward),
                        np.zeros_like(total_reward),
                        total_values,
                        gamma,
                        num_step,
                        num_worker)
        # """

        # normalize advantage
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)

        # Step 5. Training! 
        agent.train_model((total_state - obs_rms.mean) / np.sqrt(obs_rms.var),
                        (total_next_state - obs_rms.mean) / np.sqrt(obs_rms.var),
                        target, total_action,
                        adv,
                        total_policy)

        if global_step % (num_worker * num_step * 100) == 0:
            print('Now Global Step :{}'.format(global_step))
            torch.save(agent.model.state_dict(), model_path)