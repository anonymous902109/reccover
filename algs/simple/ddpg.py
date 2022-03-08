import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algs.simple.replay_memory import ReplayMemory, Transition
from src.util import eval_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# [Not the implementation used in the TD3 paper]

EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 10000

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

    def predict(self, state):
        with torch.no_grad():
            return self.forward(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400 + action_dim, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(torch.cat([q, action], 1)))
        return self.l3(q)


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.001):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)

        self.discount = discount
        self.tau = tau

        self.memory = ReplayMemory(capacity=50000)
        self.steps_done = 0

    def select_action(self, obs, env):
        ''' Selects action based on the epsilon-greedy strategy '''
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)

        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                state = torch.FloatTensor(obs.reshape(1, -1)).to(device)
                return self.actor(state).cpu().data.numpy().flatten()
        else:
            return env.action_space.sample()

    def optimize(self, batch_size=512):
        if len(self.memory) < batch_size:
            return

        # Sample replay buffer
        transitions = self.memory.sample(batch_size)  # this is array of tuples
        batch = Transition(*zip(*transitions))

        state = torch.from_numpy(np.stack(batch.state))
        action = torch.from_numpy(np.stack(batch.action))
        next_state = torch.from_numpy(np.stack(batch.next_state))
        reward = torch.from_numpy(np.stack(batch.reward))
        not_done = torch.from_numpy(np.stack(batch.not_done))

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward.squeeze() + (not_done * self.discount * target_Q.squeeze()).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action).squeeze()

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self, env, save_path, eval_path, num_episodes=100000):
        i_episode = 0

        while i_episode < num_episodes:
            i_episode += 1

            obs = env.reset()
            obs = torch.from_numpy(obs)

            done = False

            while not done:
                obs = obs.float()

                action = self.select_action(obs, env)
                new_obs, reward, done, _ = env.step(action.item())

                reward = torch.from_numpy(np.array([reward]))
                reward = reward.float()
                new_obs = torch.from_numpy(new_obs)
                new_obs = new_obs.float()

                # Save transition
                self.memory.push(obs, action, new_obs, reward, not done)

                # Move to a new state
                obs = new_obs

                # Optimize model
                self.optimize()

                if done:
                    obs = env.reset()

            if i_episode % 100 == 0:
                with torch.no_grad():
                    print('Finished episode {}'.format(i_episode))
                    eval_model(env, self.actor, num_episodes=10)
                    self.save('src/trained_models/cont/taxi_cont_ddpg', i_episode)

    def predict(self, state):
        with torch.no_grad():
            return self.actor.predict(state)

    def save(self, filename, i_episode):
        torch.save(self.critic.state_dict(), filename + "_critic_{}.pt".format(i_episode))
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer_{}".format(i_episode))

        torch.save(self.actor.state_dict(), filename + "_actor_{}.pt".format(i_episode))
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer_{}".format(i_episode))

    def load(self, actor_filename, critic_filename):
        self.critic.load_state_dict(torch.load(critic_filename))
        # self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer_{}".format(i_episode)))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(actor_filename))
        # self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer_{}".format(i_episode)))
        self.actor_target = copy.deepcopy(self.actor)
