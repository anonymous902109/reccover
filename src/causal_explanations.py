import copy
import torch

from src.model_util import run_all_masks, is_local_maxima, get_novelty, get_feature_importance, DEVICE
from envs.taxi import TaxiEnv
from envs.gridworld_traffic_env import GridworldTrafficEnv
import numpy as np


def eval_policy(policy, env, n_ep=100, k=None, start_state=None):
    """ Evaluates policy in the environment """
    total_rew = 0.0
    for i_ep in range(n_ep):
        ep_rew = unroll_episode(policy, env, k=k, start_state=start_state)
        total_rew += ep_rew

    return total_rew / n_ep


def unroll_episode(policy, env, start_state=None, k=None):
    """ Unrolls single episode in the environment, starting in start state and for k steps"""
    if start_state is None:
        obs = env.reset()
    else:
        env.timesteps = 0
        env.set_state(copy.copy(start_state))
        obs = env.get_state()

    done = False
    total_rew = 0
    i=0

    while not done:
        i += 1
        obs = torch.tensor([obs], device=DEVICE)
        action = policy.predict(obs)
        obs, rew, done, _ = env.step(action)

        if k is not None and i >= k:
            done = True

        total_rew += rew

    return total_rew


def get_agent_transitions(policy, env, n_episodes=100):
    '''
    Gathers transition data from interactions with the environment
    '''
    data = []
    for e_id in range(n_episodes):
        obs = env.reset()

        data.append(list(obs))

        done = False
        while not done:
            obs = torch.tensor([obs])
            obs = obs.to(DEVICE)
            action = policy.predict(obs)
            obs, rew, done, _ = env.step(action)

            if obs is not None:
                data.append(list(obs))

    print('Collected {} states'.format(len(data)))

    return np.vstack(data)


def get_critical_states(data, policy, env):
    """
    Extracts critical states
    """

    # generating local maxima
    local_maxima = []
    for s in data:
        local_maxima.append(is_local_maxima(s, policy, env))

    print('Collected {} local maxima states'.format(data[local_maxima].shape[0]))

    critical_states = np.vstack([data[local_maxima]])
    critical_states = np.unique(critical_states, axis=0)  # remove duplicates

    print('Collected {} critical states.'.format(critical_states.shape[0]))
    print('Critical state: {}'.format(critical_states))

    return critical_states


def gen_alt_worlds(state, data, env, features, feature_values, env_name, threshold=0.9):
    """
    Generates a set of alternative environments starting in state by intervening on its features
    """

    alt_worlds = []
    alt_params = []

    for i, (f_name, f_id) in enumerate(features.items()):
        for val in feature_values[i]:
            # import module where class is
            klass = globals()[env_name]

            # intervene on the feature
            env.reset()
            env.set_state(copy.copy(state))
            env.set_up_intervention(f_id, f_name, val)
            env.intervene()
            intervened_s = env.get_state()

            # is valid state
            valid = env.is_valid_state(intervened_s)

            # check novelty of the state
            nov_s = get_novelty(intervened_s, data)

            if (nov_s > threshold) and valid:
                alt_w = klass(alt=True, feature=f_id, f_name=f_name, value=val)
                alt_worlds.append(alt_w)
                alt_params.append((f_name, val))

    return alt_worlds


def eval_alt_worlds(s, policy, graph_policy, alt_worlds, eval_path, full_mask, k):
    """ Evaluates the policy and graph-policy in alternative environments """
    all_masks = graph_policy.masks
    causal_counter = 0

    for i, w in enumerate(alt_worlds):
        w.set_state(copy.copy(s))
        w.intervene()
        mask_returns = run_all_masks(w,
                                     graph_policy,
                                     all_masks,
                                     num_episodes=1,
                                     eval_path=eval_path.format(w.f_name, w.value, str(w.get_state())),
                                     start_obs=w.get_state(),
                                     k=k)

        # evaluate original policy in alt world
        w.reset()
        w.set_state(copy.copy(s))
        w.intervene()
        agent_return = eval_policy(policy, w, n_ep=10, k=k, start_state=w.get_state())

        mask_returns[str(full_mask)] = agent_return

        causal_confusion = test_causal_confusion(mask_returns, full_mask)
        if causal_confusion is not None:
            causal_counter += 1
            print('Causal confusion for state {} feature :{} value:{} : {}'.format(s, w.f_name, w.value, causal_confusion))

    if causal_counter > 0:
        return True
    else:
        print('No causal confusion found in state: {}'.format(s))
        return False


def test_causal_confusion(mask_returns, full_mask):
    """
    Tests for causal confusion by comparing performance of policies
    relying on different feature subsets in alternative environments
    """
    full_mask_reward = mask_returns[str(full_mask)]
    better_than_full = {str(mask): (full_mask_reward < reward) for (mask, reward) in mask_returns.items()}

    better_masks = {str(mask): mask_returns[str(mask)] for (mask, better) in better_than_full.items() if better}

    if len(better_masks) > 0:
        max_value = max(better_masks.values()) # select only masks with highest return
        better_masks = {k: v for (k, v) in better_masks.items() if v == max_value}

        better_masks[str(full_mask)] = full_mask_reward  # add the full mask
        return better_masks
    else:
        return None


def reccover(policy, graph_param_policy, env, features, feature_values, eval_path, env_name, full_mask, k):
    """ ReCCoVER algorithm for detecting and correcting causal confusion """
    # generate transition data
    data = get_agent_transitions(policy, env)

    # get critical states
    unique_data = np.unique(data, axis=0)
    critical_states = get_critical_states(unique_data, policy, env)

    # create alternative worlds
    num_alt_states = []
    num_causal_confusion = 0
    for s in critical_states:
        alt_worlds = gen_alt_worlds(s, data, env, features, feature_values, env_name)
        num_alt_states.append(len(alt_worlds))

        # evaluate in the alternative worlds
        detected = eval_alt_worlds(s,
                                   policy,
                                   graph_param_policy,
                                   alt_worlds,
                                   eval_path,
                                   full_mask,
                                   k)

        num_causal_confusion += detected

    print('Average number of alt worlds per state:{}'.format(np.mean(num_alt_states)))
    print('Number of states in which causal confusion is detected: {}'.format(num_causal_confusion))


