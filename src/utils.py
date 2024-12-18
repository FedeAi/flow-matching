import torch
import numpy as np

def generate_noisy_data(data, timestep):
    noise_scale = 1.0 - timestep
    noise = torch.randn_like(data) * noise_scale
    return data * timestep + noise

def get_target_vf(noisy_data, target_data, timestep):
    return (target_data - noisy_data) / (1.0 - timestep)

def generate_dynamic_conditioning(step, total_steps):
    """Simulate a changing conditioning vector over time."""
    progress = step / total_steps
    x = -1.0 + 2.0 * progress  # Linear transition from -1 to 1
    y = 0.5 * np.sin(2 * np.pi * progress)  # Sinusoidal variation
    return torch.tensor([x, y], dtype=torch.float32)