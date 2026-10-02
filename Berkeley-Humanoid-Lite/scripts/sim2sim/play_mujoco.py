# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.


import os
import platform
from pathlib import Path

# WSLg may fall back to software rendering unless Mesa's D3D12 driver is
# selected before MuJoCo creates its OpenGL context.
if (
    "microsoft" in platform.release().lower()
    and Path("/dev/dxg").exists()
    and Path("/usr/lib/x86_64-linux-gnu/dri/d3d12_dri.so").exists()
):
    os.environ.setdefault("GALLIUM_DRIVER", "d3d12")

import numpy as np
import torch

from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
from berkeley_humanoid_lite.environments import MujocoSimulator, Cfg


# Load configuration
cfg = Cfg.from_arguments()

if not cfg:
    raise ValueError("Failed to load config.")


# Main execution block
def main():
    """Main execution function for the MuJoCo simulation environment."""
    # Initialize environment
    robot = MujocoSimulator(cfg)
    obs = robot.reset()

    # Initialize and start policy controller
    controller = RlController(cfg)
    controller.load_policy()

    # Default actions for fallback
    default_actions = np.array(cfg.default_joint_positions, dtype=np.float32)[robot.cfg.action_indices]

    # Main control loop
    while robot.mj_viewer.is_running():
        # Send observations and receive actions
        actions = controller.update(obs.numpy())

        # Use default actions if no actions received
        if actions is None:
            actions = default_actions

        # Execute step
        actions = torch.tensor(actions)
        obs = robot.step(actions)


if __name__ == "__main__":
    main()
