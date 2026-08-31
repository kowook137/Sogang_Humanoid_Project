
import time
import threading
import tempfile
from pathlib import Path

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

        print("Policy frequency: ", 1 / self.cfg.policy_dt)
        print("Physics frequency: ", 1 / self.cfg.physics_dt)
        print("Physics substeps: ", self.physics_substeps)

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
        self._natural_gait_policy_velocity = 0.5
        self._command_ramp_duration = 0.5
        self._requested_velocity_x = 0.0
        self._smoothed_policy_velocity_x = 0.0

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

        # Hold the direction captured when straight walking begins.
        self._heading_hold_enabled = self.cfg.num_actions == 12
        self._heading_target_yaw = None
        self._heading_hold_kp = 0.4
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

        # Direct vy correction is disabled because the measured response
        # moved the robot farther away from the reference line.
        self._path_lateral_limit = 0.0

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

        # The reset quaternion is the commanded straight-ahead direction.
        self._heading_reference_yaw = self._get_heading_yaw()
        self._path_start_xy = None
        self._cross_track_error = 0.0
        self._path_lateral_command = 0.0

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

        for _ in range(self.physics_substeps):
            self._apply_actions(actions)
            mujoco.mj_step(self.mj_model, self.mj_data)

        self.mj_viewer.sync()
        observations = self._get_observations()

        # Maintain real-time simulation
        time_until_next_step = self.cfg.policy_dt - (time.perf_counter() - step_start_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        self.n_steps += 1
        self._maybe_log_gait_diagnostics()
        return observations

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

        # Apply EMA filtering and torque limits
        output_torques_clipped = torch.clip(output_torques, -self.effort_limits, self.effort_limits)

        self.mj_data.ctrl[:] = output_torques_clipped.numpy()

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

        # 현재 정책에서는 직접 vy 보정을 사용하지 않습니다.
        self._path_lateral_command = 0.0
        return 0.0

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
        command_velocity_x = self._get_smoothed_policy_velocity_x(
            command_velocity_x
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

        return torch.cat([
            self._get_base_quat(),
            self._get_base_ang_vel(),
            self._get_joint_pos()[self.cfg.action_indices],
            self._get_joint_vel()[self.cfg.action_indices],
            torch.tensor([self.mode, self.command_velocity_x, self.command_velocity_y, self.command_velocity_yaw],
                        dtype=torch.float32),
        ], dim=-1)
