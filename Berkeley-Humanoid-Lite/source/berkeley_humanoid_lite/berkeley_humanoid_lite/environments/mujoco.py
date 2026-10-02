
import os
import time
import threading
import tempfile
from pathlib import Path
from collections import deque

import numpy as np
import torch
import mujoco
import mujoco.viewer

from berkeley_humanoid_lite_lowlevel.policy.config import Cfg
from .keyboard import KeyboardCommandController


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion.

    Args:
        q (torch.Tensor): Quaternion [w, x, y, z]
        v (torch.Tensor): Vector to rotate

    Returns:
        torch.Tensor: Rotated vector
    """
    q_w = q[0]
    q_vec = q[1:4]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (torch.dot(q_vec, v)) * 2.0
    return a - b + c


class MujocoEnv:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg

        # Load appropriate MJCF model based on robot configuration
        project_root = Path(__file__).resolve().parents[4]
        asset_root = (
            project_root
            / "source/berkeley_humanoid_lite_assets/data/robots"
            / "berkeley_humanoid/berkeley_humanoid_lite"
        )
        mjcf_dir = asset_root / "mjcf"
        mesh_dir = asset_root / "meshes"

        if cfg.num_joints == 22:
            scene_name = "bhl_scene.xml"
            robot_name = "berkeley_humanoid_lite.xml"
        else:
            scene_name = "bhl_biped_scene.xml"
            robot_name = "berkeley_humanoid_lite_biped.xml"

        scene_path = mjcf_dir / scene_name
        robot_path = mjcf_dir / robot_name

        for required_path in (scene_path, robot_path, mesh_dir):
            if not required_path.exists():
                raise FileNotFoundError(
                    f"Required MuJoCo asset not found: {required_path}"
                )

        # The released MJCF refers to assets/merged, while the repository stores
        # the meshes in a sibling meshes directory. Build corrected temporary
        # XML files without modifying the assets submodule.
        robot_xml = robot_path.read_text()
        robot_xml = robot_xml.replace(
            'meshdir="assets"',
            f'meshdir="{mesh_dir.as_posix()}"',
        )
        robot_xml = robot_xml.replace('file="merged/', 'file="')

        with tempfile.TemporaryDirectory(prefix="bhl_mjcf_") as temp_dir:
            temp_path = Path(temp_dir)
            temporary_scene = temp_path / scene_name
            temporary_robot = temp_path / robot_name

            temporary_scene.write_text(scene_path.read_text())
            temporary_robot.write_text(robot_xml)

            self.mj_model = mujoco.MjModel.from_xml_path(
                str(temporary_scene)
            )

        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.cfg.physics_dt
        self.mj_viewer = mujoco.viewer.launch_passive(
            self.mj_model,
            self.mj_data,
            key_callback=self._on_key,
        )

    def _on_key(self, keycode: int) -> None:
        """Default keyboard callback for viewer-only environments."""
        pass


class MujocoVisualizer(MujocoEnv):
    """MuJoCo simulation environment for the Berkeley Humanoid Lite robot.

    This class handles the physics simulation, state observation, and control
    of the robot in the MuJoCo environment.

    Args:
        cfg (Cfg): Configuration object containing simulation parameters
    """
    def __init__(self, cfg: Cfg):
        super().__init__(cfg)

        self.num_dofs = self.mj_model.nu
        print(f"Number of DOFs: {self.num_dofs}")

    def reset(self) -> None:
        """Reset the simulation environment to initial state.

        Returns:
            torch.Tensor: Initial observations after reset
        """
        self.mj_data.qpos[0:3] = np.array([0.0, 0.0, 0.0])  # Reset base position to origin
        self.mj_data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # Default quaternion orientation
        self.mj_data.qpos[7:7 + self.num_dofs] = 0
        self.mj_data.qvel[:] = 0

    def step(self, robot_observations: np.array) -> None:
        """Execute one simulation step with the given actions.

        Args:
            actions (torch.Tensor): Joint position targets for controlled joints

        Returns:
            torch.Tensor: Updated observations after executing the action
        """
        robot_base_quat = robot_observations[0:4]
        robot_base_ang_vel = robot_observations[4:7]
        robot_joint_pos = robot_observations[7:7 + self.num_dofs]
        robot_joint_vel = robot_observations[7 + self.num_dofs:7 + self.num_dofs * 2]
        robot_mode = robot_observations[7 + self.num_dofs * 2]
        command_velocity = robot_observations[7 + self.num_dofs * 2 + 1:7 + self.num_dofs * 2 + 4]

        self.mj_data.qpos[0:3] = np.array([0.0, 0.0, 0.0])
        self.mj_data.qpos[3:7] = robot_base_quat
        self.mj_data.qvel[0:3] = np.array([0.0, 0.0, 0.0])
        self.mj_data.qvel[3:6] = robot_base_ang_vel
        self.mj_data.qpos[7:] = robot_joint_pos
        self.mj_data.qvel[6:] = robot_joint_vel

        mujoco.mj_step(self.mj_model, self.mj_data)
        self.mj_viewer.sync()


class MujocoSimulator(MujocoEnv):
    """MuJoCo simulation environment for the Berkeley Humanoid Lite robot.

    This class handles the physics simulation, state observation, and control
    of the robot in the MuJoCo environment.

    Args:
        cfg (Cfg): Configuration object containing simulation parameters
    """
    def __init__(self, cfg: Cfg):
        self.command_controller = KeyboardCommandController()
        super().__init__(cfg)
        self.physics_substeps = int(np.round(self.cfg.policy_dt / self.cfg.physics_dt))

        # Optional Sim2Real rigid-body dynamics mismatch test.
        # This uniformly scales robot body masses and inertias while
        # keeping geometry, friction, controller gains, and torque
        # limits unchanged.
        self._dynamics_mass_scale = float(
            os.environ.get(
                "BHL_DYNAMICS_MASS_SCALE",
                "1.0",
            )
        )

        if self._dynamics_mass_scale <= 0.0:
            raise ValueError(
                "BHL_DYNAMICS_MASS_SCALE must be positive"
            )

        self._nominal_total_mass = float(
            mujoco.mj_getTotalmass(
                self.mj_model
            )
        )

        self._scaled_total_mass = (
            self._nominal_total_mass
            * self._dynamics_mass_scale
        )

        if abs(
            self._dynamics_mass_scale - 1.0
        ) > 1.0e-12:
            mujoco.mj_setTotalmass(
                self.mj_model,
                self._scaled_total_mass,
            )

            # Recompute model constants derived from mass/inertia.
            mujoco.mj_setConst(
                self.mj_model,
                self.mj_data,
            )

        print(
            "Dynamics robustness: "
            f"mass/inertia scale="
            f"{self._dynamics_mass_scale:.2f}, "
            f"nominal mass="
            f"{self._nominal_total_mass:.3f} kg, "
            f"effective mass="
            f"{float(mujoco.mj_getTotalmass(self.mj_model)):.3f} kg"
        )

        # Initialize simulation parameters
        self.sensordata_dof_size = 3 * self.mj_model.nu
        self.gravity_vector = torch.tensor([0.0, 0.0, -1.0])

        # Initialize control parameters
        self.joint_kp = torch.zeros((self.cfg.num_joints,), dtype=torch.float32)
        self.joint_kd = torch.zeros((self.cfg.num_joints,), dtype=torch.float32)
        self.effort_limits = torch.zeros((self.cfg.num_joints,), dtype=torch.float32)

        self.joint_kp[:] = torch.tensor(self.cfg.joint_kp)
        self.joint_kd[:] = torch.tensor(self.cfg.joint_kd)
        self.effort_limits[:] = torch.tensor(self.cfg.effort_limits)

        self.n_steps = 0

        # Optional Sim2Real actuator / command-delay robustness test.
        # Delay is quantized to the 25 Hz policy period.
        self._action_delay_ms_requested = float(
            os.environ.get(
                "BHL_ACTION_DELAY_MS",
                "0",
            )
        )

        if self._action_delay_ms_requested < 0.0:
            raise ValueError(
                "BHL_ACTION_DELAY_MS must be non-negative"
            )

        self._action_delay_steps = int(
            round(
                (
                    self._action_delay_ms_requested
                    / 1000.0
                )
                / self.cfg.policy_dt
            )
        )

        self._action_delay_ms_effective = (
            self._action_delay_steps
            * self.cfg.policy_dt
            * 1000.0
        )

        self._action_delay_queue = deque()

        # Optional Sim2Real policy-observation sensor noise.
        # Noise is injected only into observations sent to the policy.
        # MuJoCo state and low-level PD feedback remain noise-free.
        self._sensor_noise_scale = float(
            os.environ.get(
                "BHL_SENSOR_NOISE_SCALE",
                "0.0",
            )
        )

        if self._sensor_noise_scale < 0.0:
            raise ValueError(
                "BHL_SENSOR_NOISE_SCALE must be non-negative"
            )

        self._sensor_noise_seed = int(
            os.environ.get(
                "BHL_SENSOR_NOISE_SEED",
                "0",
            )
        )

        self._sensor_noise_rng = np.random.default_rng(
            self._sensor_noise_seed
        )

        # 1.0x robustness-test noise amplitudes.
        self._sensor_orientation_sigma = np.deg2rad(
            0.5
        ) * self._sensor_noise_scale

        self._sensor_ang_vel_sigma = np.deg2rad(
            1.0
        ) * self._sensor_noise_scale

        self._sensor_joint_pos_sigma = np.deg2rad(
            0.2
        ) * self._sensor_noise_scale

        self._sensor_joint_vel_sigma = np.deg2rad(
            2.0
        ) * self._sensor_noise_scale

        print("Policy frequency: ", 1 / self.cfg.policy_dt)
        print("Physics frequency: ", 1 / self.cfg.physics_dt)
        print("Physics substeps: ", self.physics_substeps)
        print(
            "Sensor noise robustness: "
            f"scale={self._sensor_noise_scale:.2f}, "
            f"orientation="
            f"{np.degrees(self._sensor_orientation_sigma):.2f} deg, "
            f"angular velocity="
            f"{np.degrees(self._sensor_ang_vel_sigma):.2f} deg/s, "
            f"joint position="
            f"{np.degrees(self._sensor_joint_pos_sigma):.2f} deg, "
            f"joint velocity="
            f"{np.degrees(self._sensor_joint_vel_sigma):.2f} deg/s, "
            f"seed={self._sensor_noise_seed}"
        )

        print(
            "Action delay robustness: "
            f"requested="
            f"{self._action_delay_ms_requested:.1f} ms, "
            f"effective="
            f"{self._action_delay_ms_effective:.1f} ms, "
            f"policy steps="
            f"{self._action_delay_steps}"
        )

        # Initialize control mode and command variables
        self.is_killed = threading.Event()
        self.mode = 3.0  # Default to RL control mode
        self.command_velocity_x = 0.0
        self.command_velocity_y = 0.0
        self.command_velocity_yaw = 0.0

        # Use the leg policy's cleaner vx=0.50 gait for low-speed walking.
        # Ramp the command so starting and stopping remain smooth.
        self._natural_gait_enabled = self.cfg.num_actions == 12
        self._natural_gait_minimum_request = 0.3
        self._natural_gait_policy_velocity = 0.3
        self._command_ramp_duration = 0.5
        self._requested_velocity_x = 0.0
        self._smoothed_policy_velocity_x = 0.0

        # Short startup boost only:
        # leave the standing attractor at 0.40 m/s, then return to 0.30 m/s.
        self._walk_start_boost_enabled = self.cfg.num_actions == 12
        self._previous_requested_velocity_x = 0.0
        self._walk_start_boost_steps = 0
        self._walk_start_boost_velocity = 0.4
        self._walk_start_boost_duration = 1.0

        action_indices = {int(index) for index in self.cfg.action_indices}
        self._passive_joint_indices = torch.tensor(
            [
                index
                for index in range(self.cfg.num_joints)
                if index not in action_indices
            ],
            dtype=torch.long,
        )
        self._diagnostic_interval_steps = max(
            1, int(round(2.0 / self.cfg.policy_dt))
        )

        # Foot gait diagnostics.
        self._foot_body_ids = {
            "L": mujoco.mj_name2id(
                self.mj_model,
                mujoco.mjtObj.mjOBJ_BODY,
                "leg_left_ankle_roll",
            ),
            "R": mujoco.mj_name2id(
                self.mj_model,
                mujoco.mjtObj.mjOBJ_BODY,
                "leg_right_ankle_roll",
            ),
        }

        if any(body_id < 0 for body_id in self._foot_body_ids.values()):
            raise RuntimeError("Could not find ankle-roll bodies for gait diagnostics")

        # Find each foot's collision box.
        self._foot_geom_ids = {}

        for side, body_id in self._foot_body_ids.items():
            geom_start = int(self.mj_model.body_geomadr[body_id])
            geom_count = int(self.mj_model.body_geomnum[body_id])

            box_geoms = [
                geom_id
                for geom_id in range(
                    geom_start,
                    geom_start + geom_count,
                )
                if int(self.mj_model.geom_type[geom_id])
                == int(mujoco.mjtGeom.mjGEOM_BOX)
            ]

            if len(box_geoms) != 1:
                raise RuntimeError(
                    f"Expected one collision box for foot {side}, "
                    f"found {box_geoms}"
                )

            self._foot_geom_ids[side] = box_geoms[0]

        # Flat-ground height.
        plane_geoms = [
            geom_id
            for geom_id in range(self.mj_model.ngeom)
            if int(self.mj_model.geom_type[geom_id])
            == int(mujoco.mjtGeom.mjGEOM_PLANE)
        ]

        if not plane_geoms:
            raise RuntimeError("Could not find ground plane")

        self._ground_geom_id = int(
            plane_geoms[0]
        )

        self._ground_z = float(
            self.mj_model.geom_pos[
                self._ground_geom_id,
                2,
            ]
        )

        # Optional Sim2Real friction robustness test.
        # Scale the existing MuJoCo friction values for the ground
        # and both foot collision geoms without modifying the MJCF.
        self._friction_scale = float(
            os.environ.get(
                "BHL_FRICTION_SCALE",
                "1.0",
            )
        )

        if self._friction_scale <= 0.0:
            raise ValueError(
                "BHL_FRICTION_SCALE must be positive"
            )

        friction_geom_ids = [
            self._ground_geom_id,
            self._foot_geom_ids["L"],
            self._foot_geom_ids["R"],
        ]

        self._nominal_friction = {
            geom_id: self.mj_model.geom_friction[
                geom_id
            ].copy()
            for geom_id in friction_geom_ids
        }

        for geom_id in friction_geom_ids:
            self.mj_model.geom_friction[
                geom_id
            ] = (
                self._nominal_friction[geom_id]
                * self._friction_scale
            )

        print(
            "Friction robustness: "
            f"scale={self._friction_scale:.2f}, "
            f"ground="
            f"{self.mj_model.geom_friction[self._ground_geom_id]}, "
            f"L foot="
            f"{self.mj_model.geom_friction[self._foot_geom_ids['L']]}, "
            f"R foot="
            f"{self.mj_model.geom_friction[self._foot_geom_ids['R']]}"
        )

        print(
            "Foot collision geoms: "
            f"L={self._foot_geom_ids['L']}, "
            f"R={self._foot_geom_ids['R']}, "
            f"ground z={self._ground_z:.4f}"
        )

        self._reset_foot_metric_window()
        self._reset_benchmark_window()
        self._initialize_contact_gait_metrics()

        # Hold the direction captured when straight walking begins.
        self._heading_hold_enabled = self.cfg.num_actions == 12
        self._heading_target_yaw = None
        self._heading_hold_kp = 0.7
        self._heading_hold_kd = 0.05
        self._heading_correction_limit = 0.18
        self._requested_velocity_yaw = 0.0
        self._heading_error = 0.0

        # Counter the repeatable left turn during the first gait step.
        self._walk_start_yaw_bias = 0.0
        self._walk_start_yaw_bias_duration = 1.2
        self._walk_start_yaw_bias_total_steps = max(
            1,
            int(
                round(
                    self._walk_start_yaw_bias_duration
                    / self.cfg.policy_dt
                )
            ),
        )
        self._walk_start_yaw_bias_steps = 0
        self._walk_start_yaw_bias_applied = 0.0
        self._was_heading_hold_walking = False

        # Line-tracer-style path hold, updated at the 25 Hz policy rate.
        self._path_hold_enabled = self.cfg.num_actions == 12
        self._heading_reference_yaw = 0.0
        self._path_start_xy = None
        self._cross_track_error = 0.0
        self._path_lateral_command = 0.0
        self._path_heading_gain = 1.2
        self._path_heading_limit = np.deg2rad(12.0)
        self._path_heading_offset = 0.0

        # Small direct lateral correction in addition to heading control.
        # Positive cross-track error means the robot is left of the
        # reference line, so command negative vy to move back right.
        self._path_lateral_gain = 0.20
        self._path_lateral_limit = 0.03

    def _on_key(self, keycode: int) -> None:
        """Forward MuJoCo viewer key events to the keyboard controller."""
        self.command_controller.handle_key(keycode)

    def reset(self) -> torch.Tensor:
        """Reset the simulation environment to initial state.

        Returns:
            torch.Tensor: Initial observations after reset
        """
        self.mj_data.qpos[0:3] = self.cfg.default_base_position
        self.mj_data.qpos[3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Default quaternion orientation
        self.mj_data.qpos[7:] = self.cfg.default_joint_positions
        self.mj_data.qvel[:] = 0
        self._action_delay_queue.clear()

        # The reset quaternion is the commanded straight-ahead direction.
        self._heading_reference_yaw = self._get_heading_yaw()
        self._path_start_xy = None
        self._cross_track_error = 0.0
        self._path_lateral_command = 0.0

        if hasattr(
            self,
            "_gait_walk_initialized",
        ):
            self._reset_contact_gait_metrics()

        observations = self._get_observations()
        return observations

    def step(self, actions: torch.Tensor) -> torch.Tensor:
        """Execute one simulation step with the given actions.

        Args:
            actions (torch.Tensor): Joint position targets for controlled joints

        Returns:
            torch.Tensor: Updated observations after executing the action
        """
        step_start_time = time.perf_counter()

        actions_to_apply = self._get_delayed_actions(
            actions
        )

        for _ in range(self.physics_substeps):
            self._apply_actions(actions_to_apply)
            mujoco.mj_step(self.mj_model, self.mj_data)

        self.mj_viewer.sync()
        observations = self._get_observations()

        # Maintain real-time simulation
        time_until_next_step = self.cfg.policy_dt - (time.perf_counter() - step_start_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        self.n_steps += 1
        self._update_foot_metrics()
        self._update_benchmark_metrics()
        self._update_contact_gait_metrics()
        self._maybe_log_gait_diagnostics()
        return observations

    def _get_delayed_actions(
        self,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return policy actions after an optional discrete delay."""

        current_actions = actions.detach().clone()

        if self._action_delay_steps <= 0:
            return current_actions

        # Warm-start the delay line with the first command rather
        # than zero joint targets. This avoids introducing an
        # artificial initialization transient unrelated to delay.
        if not self._action_delay_queue:
            for _ in range(
                self._action_delay_steps
            ):
                self._action_delay_queue.append(
                    current_actions.clone()
                )

        self._action_delay_queue.append(
            current_actions
        )

        return self._action_delay_queue.popleft()

    def _apply_actions(self, actions: torch.Tensor):
        """Apply control actions to the robot.

        Implements PD control with torque limits and filtering.

        Args:
            actions (torch.Tensor): Target joint positions for controlled joints
        """
        target_positions = torch.zeros((self.cfg.num_joints,))
        target_positions[self.cfg.action_indices] = actions

        # PD control
        output_torques = self.joint_kp * (target_positions - self._get_joint_pos()) + \
            self.joint_kd * (-self._get_joint_vel())

        # Apply torque limits.
        output_torques_clipped = torch.clip(
            output_torques,
            -self.effort_limits,
            self.effort_limits,
        )

        self.mj_data.ctrl[:] = (
            output_torques_clipped.numpy()
        )

    def _get_base_pos(self) -> torch.Tensor:
        """Get base position of the robot.

        Returns:
            torch.Tensor: Base position [x, y, z]
        """
        return torch.tensor(self.mj_data.qpos[:3], dtype=torch.float32)

    def _get_base_quat(self) -> torch.Tensor:
        """Get base orientation quaternion from sensors.

        Returns:
            torch.Tensor: Base orientation quaternion [w, x, y, z]
        """
        return torch.tensor(self.mj_data.sensordata[self.sensordata_dof_size+0:self.sensordata_dof_size+4],
                          dtype=torch.float32)

    def _get_base_ang_vel(self) -> torch.Tensor:
        """Get base angular velocity from sensors.

        Returns:
            torch.Tensor: Base angular velocity [wx, wy, wz]
        """
        return torch.tensor(self.mj_data.sensordata[self.sensordata_dof_size+4:self.sensordata_dof_size+7],
                          dtype=torch.float32)

    def _get_projected_gravity(self) -> torch.Tensor:
        """Get gravity vector in the robot's base frame.

        Returns:
            torch.Tensor: Projected gravity vector
        """
        base_quat = self._get_base_quat()
        projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vector)
        return projected_gravity

    def _get_joint_pos(self) -> torch.Tensor:
        """Get joint positions from sensors.

        Returns:
            torch.Tensor: Joint positions
        """
        return torch.tensor(self.mj_data.sensordata[0:self.cfg.num_joints], dtype=torch.float32)

    def _get_joint_vel(self) -> torch.Tensor:
        """Get joint velocities from sensors.

        Returns:
            torch.Tensor: Joint velocities
        """
        return torch.tensor(self.mj_data.sensordata[self.cfg.num_joints:2*self.cfg.num_joints],
                          dtype=torch.float32)

    def _get_smoothed_policy_velocity_x(
        self, requested_velocity_x: float
    ) -> float:
        """Map low-speed requests to a cleaner gait and ramp commands."""
        target_velocity_x = requested_velocity_x

        if (
            self._natural_gait_enabled
            and abs(requested_velocity_x)
            >= self._natural_gait_minimum_request - 1.0e-6
        ):
            target_velocity_x = np.sign(requested_velocity_x) * max(
                abs(requested_velocity_x),
                self._natural_gait_policy_velocity,
            )

        maximum_change = (
            self._natural_gait_policy_velocity
            / self._command_ramp_duration
            * self.cfg.policy_dt
        )
        velocity_change = np.clip(
            target_velocity_x - self._smoothed_policy_velocity_x,
            -maximum_change,
            maximum_change,
        )
        self._smoothed_policy_velocity_x += float(velocity_change)

        if (
            abs(
                target_velocity_x
                - self._smoothed_policy_velocity_x
            )
            < 1.0e-6
        ):
            self._smoothed_policy_velocity_x = float(
                target_velocity_x
            )

        return self._smoothed_policy_velocity_x

    def _get_heading_yaw(self) -> float:
        """Return the robot heading in radians."""
        quat_w, quat_x, quat_y, quat_z = (
            float(value) for value in self.mj_data.qpos[3:7]
        )
        return float(
            np.arctan2(
                2.0 * (
                    quat_w * quat_z
                    + quat_x * quat_y
                ),
                1.0
                - 2.0 * (
                    quat_y ** 2
                    + quat_z ** 2
                ),
            )
        )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap an angle to [-pi, pi]."""
        return float(
            (angle + np.pi) % (2.0 * np.pi) - np.pi
        )

    def _apply_heading_hold(
        self,
        requested_velocity_x: float,
        requested_velocity_yaw: float,
    ) -> float:
        """Hold heading and compensate for the first-step left turn."""
        walking_requested = (
            self._heading_hold_enabled
            and abs(requested_velocity_x)
            >= self._natural_gait_minimum_request - 1.0e-6
        )

        if not walking_requested:
            self._heading_target_yaw = None
            self._heading_error = 0.0
            self._walk_start_yaw_bias_steps = 0
            self._walk_start_yaw_bias_applied = 0.0
            self._was_heading_hold_walking = False
            return requested_velocity_yaw

        current_yaw = self._get_heading_yaw()

        if not self._was_heading_hold_walking:
            self._heading_target_yaw = (
                self._heading_reference_yaw
            )
            self._walk_start_yaw_bias_steps = (
                self._walk_start_yaw_bias_total_steps
            )
            self._was_heading_hold_walking = True
            print(
                "Heading start assist: "
                f"bias={self._walk_start_yaw_bias:+.3f}, "
                f"duration={self._walk_start_yaw_bias_duration:.1f}s"
            )

        # Q/E 입력 중에는 자동 보정을 해제하고,
        # 회전 종료 지점을 새로운 직진 방향으로 저장합니다.
        if abs(requested_velocity_yaw) > 1.0e-6:
            self._heading_reference_yaw = current_yaw
            self._heading_target_yaw = current_yaw
            self._heading_error = 0.0
            self._walk_start_yaw_bias_steps = 0
            self._walk_start_yaw_bias_applied = 0.0
            return requested_velocity_yaw

        desired_heading_yaw = self._wrap_angle(
            self._heading_target_yaw
            + self._path_heading_offset
        )
        self._heading_error = self._wrap_angle(
            current_yaw - desired_heading_yaw
        )
        yaw_rate = float(self.mj_data.qvel[5])

        feedback_correction = (
            -self._heading_hold_kp * self._heading_error
            -self._heading_hold_kd * yaw_rate
        )

        self._walk_start_yaw_bias_applied = 0.0
        if (
            requested_velocity_x > 0.0
            and self._walk_start_yaw_bias_steps > 0
        ):
            remaining_ratio = (
                self._walk_start_yaw_bias_steps
                / self._walk_start_yaw_bias_total_steps
            )
            self._walk_start_yaw_bias_applied = (
                self._walk_start_yaw_bias
                * remaining_ratio
            )
            self._walk_start_yaw_bias_steps -= 1

        correction = (
            feedback_correction
            + self._walk_start_yaw_bias_applied
        )

        return float(
            np.clip(
                correction,
                -self._heading_correction_limit,
                self._heading_correction_limit,
            )
        )

    def _apply_lateral_path_hold(
        self,
        requested_velocity_x: float,
        requested_velocity_y: float,
        requested_velocity_yaw: float,
    ) -> float:
        """Steer toward the reference line using cross-track error."""
        walking_requested = (
            self._path_hold_enabled
            and abs(requested_velocity_x)
            >= self._natural_gait_minimum_request - 1.0e-6
        )

        if not walking_requested:
            self._path_start_xy = None
            self._cross_track_error = 0.0
            self._path_heading_offset = 0.0
            self._path_lateral_command = 0.0
            return requested_velocity_y

        current_xy = np.asarray(
            self.mj_data.qpos[0:2],
            dtype=float,
        ).copy()

        if self._path_start_xy is None:
            self._path_start_xy = current_xy.copy()

        # A/D 또는 Q/E 입력 중에는 사용자의 명령을 우선하고,
        # 입력 종료 위치에서 새로운 기준선을 시작합니다.
        if (
            abs(requested_velocity_y) > 1.0e-6
            or abs(requested_velocity_yaw) > 1.0e-6
        ):
            self._path_start_xy = current_xy.copy()
            self._cross_track_error = 0.0
            self._path_heading_offset = 0.0
            self._path_lateral_command = 0.0
            return requested_velocity_y

        path_yaw = self._heading_reference_yaw
        left_direction = np.array(
            [-np.sin(path_yaw), np.cos(path_yaw)],
            dtype=float,
        )

        position_delta = current_xy - self._path_start_xy
        self._cross_track_error = float(
            np.dot(position_delta, left_direction)
        )

        # 왼쪽의 양수 오차가 커질수록 오른쪽을 바라보도록
        # 목표 heading을 최대 12도까지 변경합니다.
        self._path_heading_offset = float(
            np.clip(
                -self._path_heading_gain
                * self._cross_track_error,
                -self._path_heading_limit,
                self._path_heading_limit,
            )
        )

        # Small lateral correction:
        #   +cross-track = robot is left of the reference line
        #   -vy          = move robot back toward the right
        self._path_lateral_command = float(
            np.clip(
                -self._path_lateral_gain
                * self._cross_track_error,
                -self._path_lateral_limit,
                self._path_lateral_limit,
            )
        )

        return self._path_lateral_command

    def _reset_foot_metric_window(self) -> None:
        self._foot_clearance_max = {
            "L": float("-inf"),
            "R": float("-inf"),
        }
        self._foot_clearance_min = {
            "L": float("inf"),
            "R": float("inf"),
        }

        self._foot_forward_min = {
            "L": float("inf"),
            "R": float("inf"),
        }
        self._foot_forward_max = {
            "L": float("-inf"),
            "R": float("-inf"),
        }

        self._foot_metric_samples = 0

    def _get_foot_clearance(self, side: str) -> float:
        """Return lowest point of foot collision box above ground."""

        geom_id = self._foot_geom_ids[side]

        center = np.asarray(
            self.mj_data.geom_xpos[geom_id],
            dtype=float,
        )

        rotation = np.asarray(
            self.mj_data.geom_xmat[geom_id],
            dtype=float,
        ).reshape(3, 3)

        half_size = np.asarray(
            self.mj_model.geom_size[geom_id, :3],
            dtype=float,
        )

        # Projection of the oriented box half-extents onto world Z.
        vertical_half_extent = float(
            np.sum(np.abs(rotation[2, :]) * half_size)
        )

        bottom_z = float(center[2] - vertical_half_extent)

        return bottom_z - self._ground_z

    def _update_foot_metrics(self) -> None:
        if abs(self._requested_velocity_x) < 0.29:
            self._reset_foot_metric_window()
            return

        yaw = self._get_heading_yaw()
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        base_x = float(self.mj_data.qpos[0])
        base_y = float(self.mj_data.qpos[1])

        for side, body_id in self._foot_body_ids.items():
            clearance = self._get_foot_clearance(side)

            self._foot_clearance_max[side] = max(
                self._foot_clearance_max[side],
                clearance,
            )
            self._foot_clearance_min[side] = min(
                self._foot_clearance_min[side],
                clearance,
            )

            foot_pos = self.mj_data.xpos[body_id]

            dx = float(foot_pos[0]) - base_x
            dy = float(foot_pos[1]) - base_y

            forward = cos_yaw * dx + sin_yaw * dy

            self._foot_forward_min[side] = min(
                self._foot_forward_min[side],
                forward,
            )
            self._foot_forward_max[side] = max(
                self._foot_forward_max[side],
                forward,
            )

        self._foot_metric_samples += 1

    def _reset_benchmark_window(self) -> None:
        """Reset interval statistics used for gait benchmarking."""
        self._benchmark_forward_velocity = []
        self._benchmark_path_lateral_velocity = []
        self._benchmark_cross_track = []
        self._benchmark_yaw_error = []
        self._benchmark_roll = []
        self._benchmark_pitch = []

        self._benchmark_leg_torque_sq_sum = 0.0
        self._benchmark_leg_torque_count = 0
        self._benchmark_leg_torque_peak = 0.0
        self._benchmark_leg_saturation_count = 0
        self._benchmark_leg_saturation_total = 0

        self._benchmark_samples = 0

    def _update_benchmark_metrics(self) -> None:
        """Accumulate quantitative walking metrics at the policy rate."""

        # Do not mix standing data into walking benchmarks.
        if abs(self._requested_velocity_x) < 0.29:
            self._reset_benchmark_window()
            return

        world_velocity = np.asarray(
            self.mj_data.qvel[0:3],
            dtype=float,
        )

        base_quat = torch.as_tensor(
            self.mj_data.qpos[3:7],
            dtype=torch.float32,
        )

        body_velocity = quat_rotate_inverse(
            base_quat,
            torch.as_tensor(
                world_velocity,
                dtype=torch.float32,
            ),
        )

        self._benchmark_forward_velocity.append(
            float(body_velocity[0])
        )

        # Lateral velocity relative to the original reference path,
        # not relative to the robot's instantaneous body heading.
        path_yaw = self._heading_reference_yaw
        left_direction = np.array(
            [-np.sin(path_yaw), np.cos(path_yaw)],
            dtype=float,
        )

        path_lateral_velocity = float(
            np.dot(
                world_velocity[0:2],
                left_direction,
            )
        )

        self._benchmark_path_lateral_velocity.append(
            path_lateral_velocity
        )

        self._benchmark_cross_track.append(
            float(self._cross_track_error)
        )

        self._benchmark_yaw_error.append(
            float(self._heading_error)
        )

        quat_w, quat_x, quat_y, quat_z = (
            float(value)
            for value in self.mj_data.qpos[3:7]
        )

        roll = float(
            np.arctan2(
                2.0
                * (
                    quat_w * quat_x
                    + quat_y * quat_z
                ),
                1.0
                - 2.0
                * (
                    quat_x ** 2
                    + quat_y ** 2
                ),
            )
        )

        sin_pitch = (
            2.0
            * (
                quat_w * quat_y
                - quat_z * quat_x
            )
        )

        pitch = float(
            np.arcsin(
                np.clip(
                    sin_pitch,
                    -1.0,
                    1.0,
                )
            )
        )

        self._benchmark_roll.append(roll)
        self._benchmark_pitch.append(pitch)

        # Leg torque statistics.
        action_indices = np.asarray(
            self.cfg.action_indices,
            dtype=int,
        )

        leg_torques = np.asarray(
            self.mj_data.ctrl,
            dtype=float,
        )[action_indices]

        leg_limits = (
            self.effort_limits[
                self.cfg.action_indices
            ]
            .detach()
            .cpu()
            .numpy()
        )

        self._benchmark_leg_torque_sq_sum += float(
            np.sum(leg_torques ** 2)
        )

        self._benchmark_leg_torque_count += int(
            leg_torques.size
        )

        self._benchmark_leg_torque_peak = max(
            self._benchmark_leg_torque_peak,
            float(
                np.max(
                    np.abs(leg_torques)
                )
            ),
        )

        valid_limits = leg_limits > 1.0e-6

        if np.any(valid_limits):
            saturated = (
                np.abs(leg_torques)
                >= (leg_limits - 1.0e-6)
            ) & valid_limits

            self._benchmark_leg_saturation_count += int(
                np.count_nonzero(saturated)
            )

            self._benchmark_leg_saturation_total += int(
                np.count_nonzero(valid_limits)
            )

        self._benchmark_samples += 1

    def _initialize_contact_gait_metrics(self) -> None:
        """Initialize lightweight foot-contact gait analysis."""

        self._gait_contact_state = {
            "L": False,
            "R": False,
        }

        self._gait_walk_initialized = False

        self._gait_last_touchdown_time = {
            "L": None,
            "R": None,
        }

        self._gait_last_touchdown_xy = {
            "L": None,
            "R": None,
        }

        self._gait_last_liftoff_time = {
            "L": None,
            "R": None,
        }

        self._gait_last_any_touchdown_time = None
        self._gait_last_any_touchdown_xy = None
        self._gait_last_any_touchdown_side = None

        self._gait_step_times = deque(maxlen=8)
        self._gait_step_lengths = deque(maxlen=8)

        self._gait_stride_times = {
            "L": deque(maxlen=6),
            "R": deque(maxlen=6),
        }

        self._gait_stride_lengths = {
            "L": deque(maxlen=6),
            "R": deque(maxlen=6),
        }

        self._gait_stance_times = {
            "L": deque(maxlen=6),
            "R": deque(maxlen=6),
        }

        self._gait_swing_times = {
            "L": deque(maxlen=6),
            "R": deque(maxlen=6),
        }

    def _reset_contact_gait_metrics(self) -> None:
        """Reset gait history when walking stops."""
        self._initialize_contact_gait_metrics()

    def _foot_ground_contact(self, side: str) -> bool:
        """Return whether a foot collision box contacts the ground."""

        foot_geom_id = self._foot_geom_ids[side]

        for contact_index in range(self.mj_data.ncon):
            contact = self.mj_data.contact[contact_index]

            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            if (
                (
                    geom1 == foot_geom_id
                    and geom2 == self._ground_geom_id
                )
                or (
                    geom2 == foot_geom_id
                    and geom1 == self._ground_geom_id
                )
            ):
                return True

        return False

    def _update_contact_gait_metrics(self) -> None:
        """Detect touchdown/liftoff events at the 25 Hz policy rate."""

        walking = abs(self._requested_velocity_x) >= 0.29

        if not walking:
            if self._gait_walk_initialized:
                self._reset_contact_gait_metrics()
            return

        contacts = {
            side: self._foot_ground_contact(side)
            for side in ("L", "R")
        }

        # 첫 walking sample은 초기 contact 상태만 저장합니다.
        if not self._gait_walk_initialized:
            self._gait_contact_state = contacts
            self._gait_walk_initialized = True
            return

        current_time = float(self.mj_data.time)

        path_yaw = self._heading_reference_yaw

        forward_direction = np.array(
            [
                np.cos(path_yaw),
                np.sin(path_yaw),
            ],
            dtype=float,
        )

        for side in ("L", "R"):
            previous_contact = self._gait_contact_state[side]
            current_contact = contacts[side]

            # Touchdown: no contact -> contact
            if not previous_contact and current_contact:
                foot_xy = np.asarray(
                    self.mj_data.xpos[
                        self._foot_body_ids[side],
                        0:2,
                    ],
                    dtype=float,
                ).copy()

                previous_touchdown_time = (
                    self._gait_last_touchdown_time[side]
                )

                previous_touchdown_xy = (
                    self._gait_last_touchdown_xy[side]
                )

                # Same-foot stride time
                if previous_touchdown_time is not None:
                    stride_time = (
                        current_time
                        - previous_touchdown_time
                    )

                    if stride_time > 0.0:
                        self._gait_stride_times[side].append(
                            stride_time
                        )

                # Same-foot stride length
                if previous_touchdown_xy is not None:
                    stride_delta = (
                        foot_xy
                        - previous_touchdown_xy
                    )

                    stride_length = float(
                        np.dot(
                            stride_delta,
                            forward_direction,
                        )
                    )

                    self._gait_stride_lengths[side].append(
                        stride_length
                    )

                # Swing time
                previous_liftoff_time = (
                    self._gait_last_liftoff_time[side]
                )

                if previous_liftoff_time is not None:
                    swing_time = (
                        current_time
                        - previous_liftoff_time
                    )

                    if swing_time > 0.0:
                        self._gait_swing_times[side].append(
                            swing_time
                        )

                # Alternating touchdown = one step
                if (
                    self._gait_last_any_touchdown_time
                    is not None
                    and self._gait_last_any_touchdown_side
                    != side
                ):
                    step_time = (
                        current_time
                        - self._gait_last_any_touchdown_time
                    )

                    if step_time > 0.0:
                        self._gait_step_times.append(
                            step_time
                        )

                    if (
                        self._gait_last_any_touchdown_xy
                        is not None
                    ):
                        step_delta = (
                            foot_xy
                            - self._gait_last_any_touchdown_xy
                        )

                        step_length = float(
                            np.dot(
                                step_delta,
                                forward_direction,
                            )
                        )

                        self._gait_step_lengths.append(
                            step_length
                        )

                self._gait_last_touchdown_time[side] = (
                    current_time
                )

                self._gait_last_touchdown_xy[side] = (
                    foot_xy
                )

                self._gait_last_any_touchdown_time = (
                    current_time
                )

                self._gait_last_any_touchdown_xy = (
                    foot_xy
                )

                self._gait_last_any_touchdown_side = side

            # Liftoff: contact -> no contact
            elif previous_contact and not current_contact:
                touchdown_time = (
                    self._gait_last_touchdown_time[side]
                )

                if touchdown_time is not None:
                    stance_time = (
                        current_time
                        - touchdown_time
                    )

                    if stance_time > 0.0:
                        self._gait_stance_times[side].append(
                            stance_time
                        )

                self._gait_last_liftoff_time[side] = (
                    current_time
                )

            self._gait_contact_state[side] = (
                current_contact
            )

    @staticmethod
    def _recent_mean(values):
        if not values:
            return None

        return float(
            np.mean(
                np.asarray(
                    values,
                    dtype=float,
                )
            )
        )

    @staticmethod
    def _gait_asymmetry_percent(
        left_value,
        right_value,
    ):
        if (
            left_value is None
            or right_value is None
        ):
            return None

        denominator = 0.5 * (
            abs(left_value)
            + abs(right_value)
        )

        if denominator < 1.0e-8:
            return None

        return (
            100.0
            * abs(
                left_value
                - right_value
            )
            / denominator
        )

    def _maybe_log_gait_diagnostics(self) -> None:
        """Report actual speed and passive-arm loading every two seconds."""
        if (
            not self._natural_gait_enabled
            or self.n_steps % self._diagnostic_interval_steps != 0
        ):
            return

        world_velocity = torch.as_tensor(
            self.mj_data.qvel[0:3],
            dtype=torch.float32,
        )
        base_quat = torch.as_tensor(
            self.mj_data.qpos[3:7],
            dtype=torch.float32,
        )
        body_velocity = quat_rotate_inverse(
            base_quat,
            world_velocity,
        )

        world_velocity_x = float(world_velocity[0])
        world_velocity_y = float(world_velocity[1])
        body_forward_velocity = float(body_velocity[0])

        quat_w, quat_x, quat_y, quat_z = (
            float(value) for value in base_quat
        )
        heading_yaw = float(
            np.arctan2(
                2.0 * (
                    quat_w * quat_z
                    + quat_x * quat_y
                ),
                1.0
                - 2.0 * (
                    quat_y ** 2
                    + quat_z ** 2
                ),
            )
        )
        yaw_rate = float(self.mj_data.qvel[5])
        base_angular_speed = float(
            np.linalg.norm(self.mj_data.qvel[3:6])
        )

        if self._passive_joint_indices.numel() > 0:
            applied_torques = torch.as_tensor(
                self.mj_data.ctrl,
                dtype=torch.float32,
            )
            arm_torques = applied_torques[
                self._passive_joint_indices
            ]
            arm_limits = self.effort_limits[
                self._passive_joint_indices
            ]
            valid_limits = arm_limits > 1.0e-6

            arm_torque_rms = float(
                torch.sqrt(torch.mean(arm_torques ** 2))
            )

            if bool(torch.any(valid_limits)):
                arm_saturation_percent = float(
                    (
                        torch.abs(arm_torques[valid_limits])
                        >= arm_limits[valid_limits] - 1.0e-6
                    ).float().mean()
                    * 100.0
                )
            else:
                arm_saturation_percent = 0.0
        else:
            arm_torque_rms = 0.0
            arm_saturation_percent = 0.0

        print(
            "Gait diagnostic: "
            f"requested vx={self._requested_velocity_x:+.2f}, "
            f"policy vx={self.command_velocity_x:+.2f}, "
            f"policy vy={self.command_velocity_y:+.3f}, "
            f"cross track={self._cross_track_error:+.3f} m, "
            f"path heading={np.degrees(self._path_heading_offset):+.1f} deg, "
            f"world vx={world_velocity_x:+.2f}, "
            f"world vy={world_velocity_y:+.2f}, "
            f"body forward={body_forward_velocity:+.2f}, "
            f"yaw={np.degrees(heading_yaw):+.1f} deg, "
            f"yaw error={np.degrees(self._heading_error):+.1f} deg, "
            f"yaw command={self.command_velocity_yaw:+.3f}, "
            f"start yaw bias={self._walk_start_yaw_bias_applied:+.3f}, "
            f"yaw rate={yaw_rate:+.2f}, "
            f"arm torque rms={arm_torque_rms:.2f}, "
            f"arm saturation={arm_saturation_percent:.0f}%, "
            f"base angular speed={base_angular_speed:.2f}"
        )


        if self._benchmark_samples > 0:
            forward_values = np.asarray(
                self._benchmark_forward_velocity,
                dtype=float,
            )

            lateral_values = np.asarray(
                self._benchmark_path_lateral_velocity,
                dtype=float,
            )

            cross_track_values = np.asarray(
                self._benchmark_cross_track,
                dtype=float,
            )

            yaw_error_values = np.asarray(
                self._benchmark_yaw_error,
                dtype=float,
            )

            roll_values = np.asarray(
                self._benchmark_roll,
                dtype=float,
            )

            pitch_values = np.asarray(
                self._benchmark_pitch,
                dtype=float,
            )

            forward_mean = float(
                np.mean(forward_values)
            )

            forward_std = float(
                np.std(forward_values)
            )

            forward_error = (
                forward_mean
                - self._requested_velocity_x
            )

            lateral_rms = float(
                np.sqrt(
                    np.mean(
                        lateral_values ** 2
                    )
                )
            )

            cross_track_rms = float(
                np.sqrt(
                    np.mean(
                        cross_track_values ** 2
                    )
                )
            )

            cross_track_max = float(
                np.max(
                    np.abs(
                        cross_track_values
                    )
                )
            )

            yaw_error_rms = float(
                np.sqrt(
                    np.mean(
                        yaw_error_values ** 2
                    )
                )
            )

            yaw_error_max = float(
                np.max(
                    np.abs(
                        yaw_error_values
                    )
                )
            )

            roll_rms = float(
                np.sqrt(
                    np.mean(
                        roll_values ** 2
                    )
                )
            )

            pitch_rms = float(
                np.sqrt(
                    np.mean(
                        pitch_values ** 2
                    )
                )
            )

            if self._benchmark_leg_torque_count > 0:
                leg_torque_rms = float(
                    np.sqrt(
                        self._benchmark_leg_torque_sq_sum
                        / self._benchmark_leg_torque_count
                    )
                )
            else:
                leg_torque_rms = 0.0

            if (
                self._benchmark_leg_saturation_total
                > 0
            ):
                leg_saturation_percent = (
                    100.0
                    * self._benchmark_leg_saturation_count
                    / self._benchmark_leg_saturation_total
                )
            else:
                leg_saturation_percent = 0.0

            benchmark_window = (
                self._benchmark_samples
                * self.cfg.policy_dt
            )

            print(
                "Benchmark diagnostic: "
                f"window={benchmark_window:.2f}s, "
                f"forward mean={forward_mean:+.3f} m/s, "
                f"std={forward_std:.3f}, "
                f"error={forward_error:+.3f}, "
                f"path lateral rms={lateral_rms:.3f} m/s, "
                f"cross-track rms={cross_track_rms:.3f} m, "
                f"max={cross_track_max:.3f} m, "
                f"yaw error rms="
                f"{np.degrees(yaw_error_rms):.2f} deg, "
                f"max="
                f"{np.degrees(yaw_error_max):.2f} deg, "
                f"roll rms="
                f"{np.degrees(roll_rms):.2f} deg, "
                f"pitch rms="
                f"{np.degrees(pitch_rms):.2f} deg, "
                f"leg torque rms={leg_torque_rms:.2f} Nm, "
                f"peak="
                f"{self._benchmark_leg_torque_peak:.2f} Nm, "
                f"saturation="
                f"{leg_saturation_percent:.1f}%"
            )

        if self._gait_walk_initialized:
            step_time = self._recent_mean(
                self._gait_step_times
            )

            step_length = self._recent_mean(
                self._gait_step_lengths
            )

            left_stride_time = self._recent_mean(
                self._gait_stride_times["L"]
            )

            right_stride_time = self._recent_mean(
                self._gait_stride_times["R"]
            )

            left_stride_length = self._recent_mean(
                self._gait_stride_lengths["L"]
            )

            right_stride_length = self._recent_mean(
                self._gait_stride_lengths["R"]
            )

            left_stance = self._recent_mean(
                self._gait_stance_times["L"]
            )

            right_stance = self._recent_mean(
                self._gait_stance_times["R"]
            )

            left_swing = self._recent_mean(
                self._gait_swing_times["L"]
            )

            right_swing = self._recent_mean(
                self._gait_swing_times["R"]
            )

            stride_length_asymmetry = (
                self._gait_asymmetry_percent(
                    left_stride_length,
                    right_stride_length,
                )
            )

            stride_time_asymmetry = (
                self._gait_asymmetry_percent(
                    left_stride_time,
                    right_stride_time,
                )
            )

            def format_value(
                value,
                scale=1.0,
                suffix="",
                digits=2,
            ):
                if value is None:
                    return "n/a"

                return (
                    f"{value * scale:.{digits}f}"
                    f"{suffix}"
                )

            print(
                "Contact gait diagnostic: "
                f"contact=L{int(self._gait_contact_state['L'])}"
                f"/R{int(self._gait_contact_state['R'])}, "
                f"step time="
                f"{format_value(step_time, suffix=' s')}, "
                f"step length="
                f"{format_value(step_length, 100.0, ' cm', 1)}, "
                f"L/R stride time="
                f"{format_value(left_stride_time, suffix=' s')}"
                f"/"
                f"{format_value(right_stride_time, suffix=' s')}, "
                f"L/R stride length="
                f"{format_value(left_stride_length, 100.0, ' cm', 1)}"
                f"/"
                f"{format_value(right_stride_length, 100.0, ' cm', 1)}, "
                f"L/R stance="
                f"{format_value(left_stance, suffix=' s')}"
                f"/"
                f"{format_value(right_stance, suffix=' s')}, "
                f"L/R swing="
                f"{format_value(left_swing, suffix=' s')}"
                f"/"
                f"{format_value(right_swing, suffix=' s')}, "
                f"stride length asym="
                f"{format_value(stride_length_asymmetry, suffix='%', digits=1)}, "
                f"stride time asym="
                f"{format_value(stride_time_asymmetry, suffix='%', digits=1)}"
            )

        if self._foot_metric_samples > 0:
            left_clearance = max(
                0.0,
                self._foot_clearance_max["L"],
            )
            right_clearance = max(
                0.0,
                self._foot_clearance_max["R"],
            )

            left_fore_aft = (
                self._foot_forward_max["L"]
                - self._foot_forward_min["L"]
            )
            right_fore_aft = (
                self._foot_forward_max["R"]
                - self._foot_forward_min["R"]
            )

            print(
                "Foot diagnostic: "
                f"L clearance={left_clearance * 100.0:.2f} cm, "
                f"R clearance={right_clearance * 100.0:.2f} cm, "
                f"L fore-aft={left_fore_aft * 100.0:.1f} cm, "
                f"R fore-aft={right_fore_aft * 100.0:.1f} cm"
            )

        self._reset_foot_metric_window()
        self._reset_benchmark_window()

    @staticmethod
    def _quat_multiply(
        q1: torch.Tensor,
        q2: torch.Tensor,
    ) -> torch.Tensor:
        """Hamilton product for [w, x, y, z] quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        return torch.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])

    def _apply_policy_sensor_noise(
        self,
        base_quat: torch.Tensor,
        base_ang_vel: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ):
        """Inject Gaussian noise only into policy observations."""

        if self._sensor_noise_scale <= 0.0:
            return (
                base_quat,
                base_ang_vel,
                joint_pos,
                joint_vel,
            )

        # Small random 3-D orientation error.
        rotation_vector = torch.tensor(
            self._sensor_noise_rng.normal(
                0.0,
                self._sensor_orientation_sigma,
                size=3,
            ),
            dtype=torch.float32,
        )

        angle = torch.linalg.vector_norm(
            rotation_vector
        )

        if float(angle) > 1.0e-12:
            axis = rotation_vector / angle

            delta_quat = torch.cat([
                torch.cos(angle / 2.0).reshape(1),
                axis * torch.sin(angle / 2.0),
            ])

            base_quat = self._quat_multiply(
                delta_quat,
                base_quat,
            )

            base_quat = (
                base_quat
                / torch.linalg.vector_norm(base_quat)
            )

        base_ang_vel = (
            base_ang_vel
            + torch.tensor(
                self._sensor_noise_rng.normal(
                    0.0,
                    self._sensor_ang_vel_sigma,
                    size=base_ang_vel.numel(),
                ),
                dtype=torch.float32,
            )
        )

        joint_pos = (
            joint_pos
            + torch.tensor(
                self._sensor_noise_rng.normal(
                    0.0,
                    self._sensor_joint_pos_sigma,
                    size=joint_pos.numel(),
                ),
                dtype=torch.float32,
            )
        )

        joint_vel = (
            joint_vel
            + torch.tensor(
                self._sensor_noise_rng.normal(
                    0.0,
                    self._sensor_joint_vel_sigma,
                    size=joint_vel.numel(),
                ),
                dtype=torch.float32,
            )
        )

        return (
            base_quat,
            base_ang_vel,
            joint_pos,
            joint_vel,
        )

    def _get_observations(self) -> torch.Tensor:
        """Get complete observation vector for the policy.

        Returns:
            torch.Tensor: Concatenated observation vector containing base orientation,
                         angular velocity, joint positions, velocities, and command state
        """
        command_velocity_x, command_velocity_y, command_velocity_yaw = (
            self.command_controller.get_velocity()
        )

        self._requested_velocity_x = command_velocity_x

        minimum_velocity = self._natural_gait_minimum_request
        boost_velocity = self._walk_start_boost_velocity

        crossed_start_threshold = (
            self._walk_start_boost_enabled
            and abs(self._requested_velocity_x)
            >= minimum_velocity - 1.0e-6
            and abs(self._previous_requested_velocity_x)
            < minimum_velocity - 1.0e-6
        )

        if crossed_start_threshold:
            self._walk_start_boost_steps = max(
                1,
                int(
                    round(
                        self._walk_start_boost_duration
                        / self.cfg.policy_dt
                    )
                ),
            )
            print(
                "Walk-start boost: "
                f"requested vx={self._requested_velocity_x:+.2f}, "
                f"policy vx={np.sign(self._requested_velocity_x) * boost_velocity:+.2f}, "
                f"duration={self._walk_start_boost_duration:.1f}s"
            )

        # Normal command is still the true requested velocity (0.30 m/s).
        command_velocity_x = self._get_smoothed_policy_velocity_x(
            command_velocity_x
        )

        # Override only during the startup window.
        if (
            self._walk_start_boost_steps > 0
            and abs(self._requested_velocity_x)
            >= minimum_velocity - 1.0e-6
        ):
            command_velocity_x = (
                np.sign(self._requested_velocity_x)
                * boost_velocity
            )
            self._walk_start_boost_steps -= 1
        elif abs(self._requested_velocity_x) < minimum_velocity - 1.0e-6:
            self._walk_start_boost_steps = 0

        self._previous_requested_velocity_x = (
            self._requested_velocity_x
        )

        self._requested_velocity_yaw = command_velocity_yaw
        command_velocity_y = self._apply_lateral_path_hold(
            self._requested_velocity_x,
            command_velocity_y,
            self._requested_velocity_yaw,
        )
        command_velocity_yaw = self._apply_heading_hold(
            self._requested_velocity_x,
            command_velocity_yaw,
        )

        self.command_velocity_x = command_velocity_x
        self.command_velocity_y = command_velocity_y
        self.command_velocity_yaw = command_velocity_yaw

        base_quat = self._get_base_quat()
        base_ang_vel = self._get_base_ang_vel()
        joint_pos = self._get_joint_pos()[
            self.cfg.action_indices
        ]
        joint_vel = self._get_joint_vel()[
            self.cfg.action_indices
        ]

        (
            base_quat,
            base_ang_vel,
            joint_pos,
            joint_vel,
        ) = self._apply_policy_sensor_noise(
            base_quat,
            base_ang_vel,
            joint_pos,
            joint_vel,
        )

        return torch.cat([
            base_quat,
            base_ang_vel,
            joint_pos,
            joint_vel,
            torch.tensor(
                [
                    self.mode,
                    self.command_velocity_x,
                    self.command_velocity_y,
                    self.command_velocity_yaw,
                ],
                dtype=torch.float32,
            ),
        ], dim=-1)
