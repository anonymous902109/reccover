import math
import random
from collections import namedtuple

import torch
from torch import optim, nn
from torch.optim.lr_scheduler import StepLR

from src.causal_explanations import eval_policy
from algs.simple.dqn import DQN
from algs.simple.replay_memory import ReplayMemory, Transition

BATCH_SIZE = 1024
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 100000
TARGET_UPDATE = 10
SAVE_FREQ = 100

EvalStat = namedtuple('EvalStat', ('episode', 'env', 'mask', 'avg_return'))

class DQNAgent:
    ''' DQN learning agent '''

    def __init__(self, input_size, hidden_size, num_actions, num_features, device):
        super(DQNAgent, self).__init__()
        self.input_size = input_size
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.num_features = num_features

        self.policy_net = DQN(input_size, self.hidden_size, num_actions).to(device)
        self.target_net = DQN(input_size, self.hidden_size, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(50000)
        self.steps_done = 0

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.scheduler = StepLR(self.optimizer, gamma=0.1, step_size=200 * 4000)
        self.criterion = nn.MSELoss()

        self.device = device

    def select_action(self, obs):
        ''' Selects action based on the epsilon-greedy strategy '''
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)

        random_timesteps = 200 * 500

        if self.steps_done < random_timesteps:
            return torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.long, device=self.device)

        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(obs).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.long, device=self.device)

    def optimize_dqn(self):
        ''' Performs one batch gradient update '''
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)  # this is array of tuples
        batch = Transition(*zip(*transitions))  # turns into tuple of arrays

        # separate those that are not final states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool, device=self.device)
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

    def train_dqn(self, env, save_path, num_episodes=100000):
        ''' Training DQN '''

        stop = False
        i_episode = 0

        while i_episode < num_episodes:
            i_episode += 1

            obs = env.reset()
            obs = torch.from_numpy(obs)

            done = False
            num_timesteps = 0

            while not done and not stop:
                obs = obs.float()
                obs = obs.to(self.device)

                obs = obs.unsqueeze(0)
                action = self.select_action(obs)
                new_obs, reward, done, _ = env.step(action.item())

                if done:
                    new_obs = None
                else:
                    new_obs = torch.from_numpy(new_obs)
                    new_obs = new_obs.float()
                    new_obs = new_obs.to(self.device)

                reward = torch.tensor([reward], device=self.device)
                reward = reward.float()

                # Save transition
                self.memory.push(obs, action, new_obs, reward, not done)

                # Move to a new state
                obs = new_obs

                # Optimize model
                self.optimize_dqn()

                num_timesteps += 1

                if done:
                    obs = env.reset()

            if i_episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if i_episode % 100 == 0:
                with torch.no_grad():
                    print('Finished episode {}'.format(i_episode))
                    total_correct = eval_policy(self.policy_net, env, n_ep=10)
                    print('Confused = False: {}'.format(total_correct))
                    env.set_confused(None)
                    total_correct = eval_policy(self.policy_net, env, n_ep=10)
                    print('Confused = None: {}'.format(total_correct))
                    env.set_confused(False)
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

