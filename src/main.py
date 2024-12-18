import torch
import numpy as np
from tqdm import tqdm
from flow_matching import FlowMatching
from utils import generate_dynamic_conditioning
from visualization import visualize_flow_matching, visualize_flow

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize model and target data
    model = FlowMatching()
    target_data = model.generate_two_circles_targets()

    # Training loop
    n_steps = 1500
    for step in tqdm(range(n_steps)):
        # Simulate dynamic conditioning that changes over time
        conditioning_info = generate_dynamic_conditioning(step, n_steps)
        loss = model.train_step(target_data, conditioning_info)

        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}, Loss: {loss:.4f}")
            visualize_flow_matching(model, target_data, step + 1)

    # Generate animations
    ani = visualize_flow(model, target_data, generate_dynamic_conditioning(n_steps - 1, n_steps))
    ani.save('images/flow_matching_dynamic_conditions.gif', writer='pillow')

    ani = visualize_flow(model, target_data, generate_dynamic_conditioning(1, n_steps))
    ani.save('images/flow_matching_dynamic_conditions_1.gif', writer='pillow')