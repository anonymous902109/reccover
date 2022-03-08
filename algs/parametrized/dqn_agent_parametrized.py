import math
import random
from collections import namedtuple

import torch
from torch import optim, nn
from torch.optim.lr_scheduler import StepLR
import numpy as np

from algs.parametrized.dqn_parametrized import DQNParametrized
from algs.parametrized.replay_memory_parametrized import ReplayMemoryParametrized, ParametrizedTransition
from src.model_util import get_all_masks_of_len, mask_state, run_all_masks, write_csv

BATCH_SIZE = 1024
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 50000
TARGET_UPDATE = 100
SAVE_FREQ = 100

EvalStat = namedtuple('EvalStat', ('episode', 'env', 'mask', 'avg_return'))


class DQNAgentParametrized:
    ''' DQN learning agent '''

    def __init__(self, input_size, hidden_size, num_actions, num_features, device, masks):
        super(DQNAgentParametrized, self).__init__()
        self.input_size = input_size
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.num_features = num_features

        self.policy_net = DQNParametrized(input_size, self.hidden_size, num_actions).to(device)
        self.target_net = DQNParametrized(input_size, self.hidden_size, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemoryParametrized(10000, num_masks=2**self.num_features)
        self.timesteps = 0

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.scheduler = StepLR(self.optimizer, gamma=0.1, step_size=200*4000)
        self.criterion = nn.MSELoss()

        self.losses = np.zeros((100,))
        self.loss_pos = 0

        self.masks = masks

        self.device = device

    def select_action(self, obs):
        ''' Selects action based on the epsilon-greedy strategy '''
        sample = random.random()
        random_steps = 200 * 5000

        self.timesteps += 1

        if self.timesteps < random_steps:  # approximately first 10000 episodes, play randomly
            return torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.long, device=self.device)

        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * (self.timesteps - random_steps) / EPS_DECAY)

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(obs).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.long, device=self.device)

    def optimize_dqn(self, mask, i_episode):
        ''' Performs one batch gradient update '''
        if len(self.memory) < BATCH_SIZE:
            return False

        transitions = self.memory.sample(BATCH_SIZE, mask=mask)  # this is array of tuples
        batch = ParametrizedTransition(*zip(*transitions))  # turns into tuple of arrays

        # separate those that are not final states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool,
                                      device=self.device)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action).squeeze()
        reward_batch = torch.stack(batch.reward)

        state_action_values = self.policy_net(state_batch).squeeze().gather(1, action_batch.unsqueeze(1))
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).squeeze().max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch.squeeze()

        loss = self.criterion(state_action_values.squeeze(), expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)  # clip gradients
        self.optimizer.step()
        self.scheduler.step()

        return self.check_loss_convergence(i_episode, loss.item())

    def train_dqn(self, env, save_path, eval_path, num_episodes=100000):
        ''' Training DQN '''
        stop = False
        i_episode = 0

        while not stop:
            i_episode += 1

            obs = env.reset()
            obs = torch.from_numpy(obs)

            mask = self.get_rand_mask()
            done = False

            while not done:
                obs = obs.float()
                obs = obs.to(self.device)
                obs = mask_state(obs, mask)

                action = self.select_action(obs)
                new_obs, reward, done, _ = env.step(action.item())

                if done:
                    new_obs = None
                    new_obs_encoded = None
                else:
                    new_obs = torch.from_numpy(new_obs)
                    new_obs = new_obs.float()
                    new_obs_encoded = new_obs.to(self.device)
                    new_obs_encoded = mask_state(new_obs, mask)

                reward = torch.tensor([reward], device=self.device)
                reward = reward.float()

                # Save transition
                self.memory.push(obs, action, new_obs_encoded, reward, not done, mask)

                # Move to a new state
                obs = new_obs

                # Optimize model
                stop = self.optimize_dqn(mask=mask, i_episode=i_episode)

                if i_episode > num_episodes:
                    stop = True

                if done:
                    mask = self.get_rand_mask()
                    obs = env.reset()

            if i_episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if i_episode % 1000 == 0:
                with torch.no_grad():
                    print('Finished episode {}'.format(i_episode))
                    run_all_masks(env, self.policy_net,
                                  self.masks, eval_path + '_{}.pt'.format(i_episode), i_episode, num_episodes=10)
                    self.save(save_path)

    def predict(self, obs):
        with torch.no_grad():
            return self.policy_net(obs).max(1)[1].view(1, 1)

    def save(self, save_path):
        torch.save(self.policy_net.state_dict(), save_path)

    def load(self, path, from_cuda=False):
        if from_cuda:
            state_dict = torch.load(path, map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(path)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def get_rand_mask(self):
        rand_mask = np.random.choice(2, (1, self.num_features), p=[0.2, 0.8])
        rand_mask = torch.from_numpy(rand_mask)
        rand_mask = rand_mask.to(self.device)

        return rand_mask

    def check_loss_convergence(self, i_episode, loss):
        self.losses[self.loss_pos % 100] = loss
        self.loss_pos += 1
        loss_range = np.max(self.losses) - np.min(self.losses)

        write_csv([(i_episode, loss, loss_range)], 'src/results/taxi_discrete_loss.csv',
                  ['Episode', 'Loss', 'Loss range'])

        if i_episode < 5000:
            return False

        if loss_range < 0.5:
            return True
        else:
            return False