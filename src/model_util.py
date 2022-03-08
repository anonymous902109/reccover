import csv
import os
from collections import namedtuple

import copy
import math
import random
from itertools import product

import torch
import numpy as np

EvalStat = namedtuple('EvalStat', ('episode', 'env', 'mask', 'avg_return'))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mask_state(obs, mask):
    obs = torch.tensor(obs)
    obs = obs.to(DEVICE)
    mask = mask.to(DEVICE)

    if len(obs.shape) == 1:
        obs = obs.unsqueeze(dim=0)

    if mask is None:
        return obs

    mask = mask.expand_as(obs)
    obs = torch.cat([obs * mask, mask], dim=1)
    return obs


def run_masked_episode(env, model, mask, start_obs=None, k=None):
    done = False
    total_reward = 0.0
    if start_obs is None:
        obs = env.reset()
    else:
        env.timesteps = 0
        env.set_state(copy.copy(start_obs))
        obs = env.get_state()

    i = 0
    while not done:
        i += 1
        obs = torch.from_numpy(obs)
        obs = obs.float()

        masked_obs = mask_state(obs, mask)

        action = model.predict(masked_obs)

        obs, reward, done, _ = env.step(action)
        total_reward += reward

        if not done and k is not None and i >= k:
            done = True

    return total_reward


def run_all_masks(env, model, masks, eval_path, train_ep=0, num_episodes=10, start_obs=None, k=None):
    eval_stats = []
    mask_returns = {}
    for mask in masks:
        mask_return = 0.0
        for i_episode in range(num_episodes):
            mask_return += run_masked_episode(env, model, mask, start_obs, k)

        mask_returns[str(mask.tolist())] = mask_return/num_episodes
        eval_stats.append((train_ep, mask, mask_return/num_episodes))

    write_csv(eval_stats, eval_path, header=['Episode', 'Mask', 'Average Return'])
    return mask_returns


def run_episode(env, model=None, render=False, verbose=False, device=torch.device('cpu')):
    '''Runs one episode. Actions are chosen based on the provided model or randomly'''
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        obs = torch.from_numpy(obs)
        obs = obs.float()
        obs = obs.to(device)

        if model is None:
            action = env.action_space.sample()
        else:
            if len(obs.shape) == 1:
                obs = obs.unsqueeze(0)

            obs = obs.to(device)
            action = model.predict(obs)

        obs, reward, done, _ = env.step(action)

        total_reward += reward

        if done and verbose:
            print('Total reward: {}'.format(total_reward))

    return total_reward


def eval_model(env, model, num_episodes=100, device=torch.device('cpu')):
    total_reward = 0.0
    for i in range(num_episodes):
        total_reward += run_episode(env, model, device=device)

    print('Average reward for env {}: {}'.format(env.name, total_reward/num_episodes))
    return total_reward/num_episodes


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def write_csv(data, csv_path, header=['Episode', 'Env', 'Mask', 'Average Return']):
    '''
    Writes evaluation stats into a csv file
    :param data: list of tuples
    :param csv_path: path to csv file
    :return:
    '''
    if os.path.exists(csv_path):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    with open(csv_path, append_write) as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if append_write == 'w': # if writing for the first time, write headers
            writer.writerow(header)
        for stat in data:
            writer.writerow(stat)


def get_all_masks_of_len(length):
    all_masks = list(product(range(2), repeat=length))
    all_masks.reverse()  # so that graphs with more features are evaluated first

    all_masks = torch.from_numpy(np.array(all_masks))
    print('Found {} masks'.format(all_masks.shape[0]))
    return all_masks


def get_Q_val(policy, obs):
    obs = copy.copy(obs)
    obs = torch.tensor(obs)
    obs = obs.to(DEVICE)

    return policy.policy_net.forward(obs).flatten()


def get_V_val(policy, obs, mask=None):
    obs = copy.copy(obs)
    if mask is not None:
        obs = mask_state(obs, mask)

    obs = torch.tensor(obs)
    obs = obs.to(DEVICE)

    Q_vals = get_Q_val(policy, obs)

    return max(Q_vals)


def is_local_maxima(obs, policy, env, mask=None):
    obs = copy.copy(obs)
    neigh_V_vals = []

    if mask is not None:
        masked_obs = torch.tensor(obs)
        masked_obs = mask_state(masked_obs, mask)
        V_val = get_V_val(policy, masked_obs).item()
    else:
        V_val = get_V_val(policy, obs)

    actions = env.num_actions

    for a in range(actions):
        next_V_val = get_avg_V_val(policy, env, obs, a, mask=mask)
        if next_V_val > V_val:
            return False

        neigh_V_vals.append(next_V_val)

    if (len(neigh_V_vals) > 0) and (V_val >= max(neigh_V_vals)):
        return True
    else:
        return False


def get_avg_V_val(policy, env, obs, action, n_ep=500, mask=None):
    V_vals = []
    for ep_id in range(n_ep):
        env.reset()
        env.set_state(obs)

        next_state, rew, _, _ = env.step(action)
        if next_state is not None:
            next_V_val = get_V_val(policy, next_state, mask=mask).item()
        else:
            next_V_val = 0

        V_vals.append(next_V_val)

    return np.mean(V_vals)


def is_local_minima(obs, policy, env):
    obs = copy.copy(obs)

    neigh_V_vals = []
    V_val = get_V_val(policy, obs)
    actions = env.num_actions

    for a in range(actions):
        env.reset()
        env.set_state(obs)

        next_state, rew, done, _ = env.step(a)
        if next_state is None:
            next_V_val = 0
        else:
            next_V_val = get_V_val(policy, next_state)

        neigh_V_vals.append(next_V_val)

    if V_val <= min(neigh_V_vals):
        return True
    else:
        return False


def count_occurences(state, data):
    state_dim = state.shape[0]
    tmp = np.sum(data == state, axis=-1) # feature-wise comparison
    n_s = np.sum(tmp == state_dim) # when all features are the same

    return n_s


def get_novelty(state, data):
    n_s = count_occurences(state, data)
    novelty = 1.0 / math.sqrt(n_s) if n_s > 0 else 1
    return novelty

def get_feature_importance(policy, state, f_id, f_values, env, n_ep=100, k=3):
    rews_orig = []

    for i in range(n_ep):
        done = False
        env.reset()
        env.set_state(copy.copy(state))
        obs = copy.copy(state)
        ep_rew = 0
        l = 0

        while not done and l < k:
            obs = torch.tensor([obs], device=DEVICE)
            action = policy.predict(obs)
            obs, rew, done, _ = env.step(action)

            ep_rew += rew
            l += 1

        rews_orig.append(ep_rew)

    r_orig = np.mean(rews_orig)

    rews_pert = []
    for j in range(n_ep):
        pert_f = random.choice(f_values)

        env.reset()
        env.set_state(copy.copy(state))
        obs = copy.copy(state)
        obs[f_id] = pert_f
        ep_rew = 0
        l = 0
        done = False

        while not done and l < k:
            obs = torch.tensor([obs], device=DEVICE)
            action = policy.predict(obs)
            obs, rew, done, _ = env.step(action)

            pert_f = random.choice(f_values)
            if not done:
                obs[f_id] = pert_f

            ep_rew += rew
            l += 1

        rews_pert.append(ep_rew)

    r_pert = np.mean(rews_pert)

    importance = r_orig - r_pert

    return importance



