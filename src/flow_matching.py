import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import generate_noisy_data, get_target_vf

class FlowMatching:
    def __init__(self):
        self.model = VectorFieldNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def generate_two_circles_targets(self):
        # Generate target points from two circles
        theta = np.linspace(0, 2 * np.pi, 100)
        circle1 = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        circle2 = np.stack([2 * np.cos(theta), 2 * np.sin(theta)], axis=1)
        target_points = np.concatenate([circle1, circle2], axis=0)
        return torch.tensor(target_points, dtype=torch.float32)

    def train_step(self, target_data, conditioning_info):
        timestep = torch.rand(1).item()
        noisy_data = generate_noisy_data(target_data, timestep)
        target_vf = get_target_vf(noisy_data, target_data, timestep)

        input_data = torch.cat([noisy_data, conditioning_info.repeat(noisy_data.size(0), 1)], dim=1)
        predicted_vf = self.model(input_data)

        loss = self.loss_fn(predicted_vf, target_vf)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

class VectorFieldNet(nn.Module):
    def __init__(self):
        super(VectorFieldNet, self).__init__()
        self.fc1 = nn.Linear(4, 64)  # Input: 2D data + 2D conditioning
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)  # Output: 2D vector field

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)