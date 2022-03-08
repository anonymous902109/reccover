import copy
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from algs.simple.ddpg import Actor, Critic
from algs.parametrized.replay_memory_parametrized import ReplayMemoryParametrized, ParametrizedTransition
from src.util import get_all_masks_of_len, run_all_masks, mask_state, write_csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# [Not the implementation used in the TD3 paper]

EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 50000


class DDPGParametrized(object):
    def __init__(self, state_dim, action_dim, max_action, num_features, discount=0.99, tau=0.001):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)

        self.discount = discount
        self.tau = tau
        self.num_features = num_features

        self.memory = ReplayMemoryParametrized(capacity=5000)
        self.steps_done = 0

        self.masks = get_all_masks_of_len(self.num_features)

        self.actor_losses = np.zeros((100, ))
        self.critic_losses = np.zeros((100, ))
        self.actor_loss_pos = 0
        self.critic_loss_pos = 0

    def select_action(self, obs, env):
        ''' Selects action based on the epsilon-greedy strategy '''
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)

        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                state = torch.FloatTensor(obs.cpu().reshape(1, -1)).to(device)
                return self.actor(state).cpu().data.numpy().flatten()
        else:
            return env.action_space.sample()

    def optimize(self, mask, i_episode, batch_size=512):
        if len(self.memory) < batch_size:
            return

        # Sample replay buffer
        transitions = self.memory.sample(batch_size, mask)  # this is array of tuples
        batch = ParametrizedTransition(*zip(*transitions))

        state = torch.stack(batch.state)
        action = torch.from_numpy(np.stack(batch.action))
        next_state = torch.stack(batch.next_state)
        reward = torch.from_numpy(np.stack(batch.reward))
        not_done = torch.from_numpy(np.stack(batch.not_done))
        state, action, next_state, reward, not_done = state.to(device), action.to(device), next_state.to(device), reward.to(device), not_done.to(device)

        # Compute the target Q value
        target_Q = self.critic_target(next_state.squeeze(), self.actor_target(next_state.squeeze()))
        target_Q = reward.squeeze() + (not_done * self.discount * target_Q.squeeze()).detach()

        # Get current Q estimate
        current_Q = self.critic(state.squeeze(), action).squeeze()

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state.squeeze(), self.actor(state.squeeze())).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        stop = self.check_loss_convergence(i_episode, actor_loss.item(), critic_loss.item())
        return stop

    def train(self, env, save_path, eval_path, num_episodes=100000):
        i_episode = 0
        stop = False

        while not stop:
            i_episode += 1

            obs = env.reset()
            obs = torch.from_numpy(obs)

            done = False

            mask = self.get_random_mask()
            while not done:
                obs = obs.float()
                obs = mask_state(obs, mask)

                action = self.select_action(obs, env)
                new_obs, reward, done, _ = env.step(action.item())

                reward = torch.from_numpy(np.array([reward]))
                reward = reward.float()
                new_obs = torch.from_numpy(new_obs)
                new_obs = new_obs.float()
                new_obs_encoded = mask_state(new_obs, mask)

                # Save transition
                self.memory.push(obs, action, new_obs_encoded, reward, not done, mask)

                # Move to a new state
                obs = new_obs

                # Optimize model
                stop = self.optimize(mask, i_episode=i_episode)

                if done:
                    obs = env.reset()

            if i_episode % 100 == 0:
                with torch.no_grad():
                    print('Finished episode {}'.format(i_episode))
                    run_all_masks(env, self.actor, self.masks, eval_path, train_ep=i_episode)
                    self.save(save_path)

    def predict(self, state):
        with torch.no_grad():
            return self.actor.predict(state)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic.pt")
        torch.save(self.actor.state_dict(), filename + "_actor.pt")

    def load(self, actor_filename, critic_filename):
        self.critic.load_state_dict(torch.load(critic_filename))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(actor_filename))
        self.actor_target = copy.deepcopy(self.actor)

    def get_random_mask(self):
        rand_mask = np.random.choice(2, size=(1, self.num_features), p=[0.2, 0.8])
        return torch.from_numpy(rand_mask)

    def check_loss_convergence(self, i_episode, actor_loss, critic_loss):
        self.actor_losses[self.actor_loss_pos % 100] = actor_loss
        self.actor_loss_pos += 1
        self.critic_losses[self.critic_loss_pos% 100] = critic_loss
        self.critic_loss_pos += 1

        actor_loss_range = np.max(self.actor_losses) - np.min(self.actor_losses)
        critic_loss_range = np.max(self.critic_losses) - np.min(self.critic_losses)

        write_csv([(i_episode, actor_loss_range, critic_loss_range)],
                  'src/results/losses.csv',
                  header=['Episode', 'Actor loss range', 'Critic loss range'])

        if i_episode < 100:
            return False

        if actor_loss_range < 0.5 and critic_loss_range < 0.5:
            return True
        else:
            return False