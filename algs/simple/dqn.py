import torch
import torch.nn as nn


class DQN(nn.Module):
    ''' DQN model '''

    def __init__(self, input_size, hidden_size, num_actions):
        super(DQN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        x = x.float()

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.output(x)

        return x

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x).max(1)[1].view(1, 1)



