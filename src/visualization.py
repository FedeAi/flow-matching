import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import generate_noisy_data

def visualize_flow_matching(model, target_data, step):
    # Generate grid for visualization
    x, y = np.meshgrid(np.linspace(-3, 3, 20), np.linspace(-3, 3, 20))
    grid_points = torch.tensor(np.stack((x.flatten(), y.flatten()), axis=1), dtype=torch.float32)

    with torch.no_grad():
        conditioning_info = torch.zeros(2)
        input_grid = torch.cat([grid_points, conditioning_info.repeat(grid_points.size(0), 1)], dim=1)
        vector_field = model.model(input_grid).numpy()

    plt.figure(figsize=(6, 6))
    plt.quiver(grid_points[:, 0], grid_points[:, 1], vector_field[:, 0], vector_field[:, 1], color='blue')
    plt.scatter(target_data[:, 0], target_data[:, 1], s=10, c='red')
    plt.title(f'Learned Vector Field at Step {step}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'images/vector_field_step_{step}.png')
    plt.close()

def visualize_flow(model, target_data, conditioning_info):
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame):
        timestep = frame / 100
        noisy_data = generate_noisy_data(target_data, timestep)
        ax.clear()
        ax.scatter(noisy_data[:, 0], noisy_data[:, 1], s=10, c='blue')
        ax.set_title(f'Timestep {timestep:.2f}')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)

    ani = FuncAnimation(fig, update, frames=100, interval=50)
    return ani