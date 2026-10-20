import torch as th
from omnigibson.controllers.controller_base import (
    BaseController,
    ControlType,
    GripperController,
    IsGraspingState,
    LocomotionController,
    ManipulationController,
)
from omnigibson.utils.ui_utils import create_module_logger
import omnigibson as og  # For og.sim.device
from omnigibson.macros import gm
import numpy as np

# Create module logger
log = create_module_logger(module_name=__name__)


class IndividualJointPDController(LocomotionController, ManipulationController, GripperController):
    def __init__(
            self,
            control_freq,
            motor_type,  # This will be forced to 'effort' for hybrid control
            control_limits,
            dof_idx,
            command_input_limits="default",
            command_output_limits="default",
            kp=50,
            kd=1,
            use_impedances=False,
            use_gravity_compensation=False,
            use_cc_compensation=True,
            use_delta_commands=False,  # Delta commands are less common for torque control
            compute_delta_in_quat_space=None,  # Delta commands are less common for torque control
            max_effort=None,
            min_effort=None
    ):
        motor_type = "effort"

        self.kp = kp
        self.kd = kd
        self.max_effort = None if max_effort is None else th.tensor(max_effort).to(og.sim.device)
        self.min_effort = None if min_effort is None else th.tensor(min_effort).to(og.sim.device)

        self._motor_type = motor_type.lower()
        self._use_impedances = True

        self._use_gravity_compensation = use_gravity_compensation
        self._use_cc_compensation = use_cc_compensation

        super().__init__(
            control_freq=control_freq,
            control_limits=control_limits,
            dof_idx=dof_idx,
            command_input_limits=command_input_limits,
            command_output_limits=command_output_limits,
        )

        self.cached_torque = None

    def _update_goal(self, command, control_dict):
        target_joint_pos = command.to(og.sim.device)

        target_joint_pos = target_joint_pos.clip(
            self._control_limits[ControlType.get_type("position")][0][self.dof_idx],
            self._control_limits[ControlType.get_type("position")][1][self.dof_idx],
        )

        current_joint_pos = control_dict["joint_position"][self.dof_idx].to(og.sim.device)
        target_joint_vel = th.zeros_like(target_joint_pos)

        return dict(target_joint_pos=target_joint_pos, target_joint_vel=target_joint_vel)

    def compute_control(self, goal_dict, control_dict):
        current_joint_pos = control_dict["joint_position"][self.dof_idx].to(og.sim.device)
        current_joint_vel = control_dict["joint_velocity"][self.dof_idx].to(og.sim.device)

        joint_pos_desired = goal_dict["target_joint_pos"].to(og.sim.device)
        joint_vel_desired = goal_dict["target_joint_vel"].to(og.sim.device)

        u_feedback = self.kp * (joint_pos_desired - current_joint_pos) + self.kd * (joint_vel_desired - current_joint_vel)
        u_feedforward = th.zeros_like(u_feedback)
        u = u_feedback + self._to_tensor(u_feedforward[:7]).to(og.sim.device)

        if self.min_effort is not None and self.max_effort is not None:
            assert u.shape == self.max_effort.shape == self.min_effort.shape
            u = u.clip(
                self.min_effort,
                self.max_effort,
            )

        return u

    def clip_control(self, control):
        clipped_control = control.clip(
            self._control_limits[self.control_type][0][self.dof_idx],
            self._control_limits[self.control_type][1][self.dof_idx],
        )

        idx = [True] * self.control_dim

        control_copy = control.clone()
        control_copy[idx] = clipped_control[idx]
        return control_copy

    def compute_no_op_goal(self, control_dict):
        target_joint_pos = control_dict["joint_position"][self.dof_idx].to(og.sim.device)
        target_joint_vel = th.zeros_like(target_joint_pos)

        return dict(target_joint_pos=target_joint_pos, target_joint_vel=target_joint_vel)

    def _compute_no_op_action(self, control_dict):
        return th.zeros(self.command_dim, device=og.sim.device)

    def _get_goal_shapes(self):
        return dict(
            target_joint_pos=(self.control_dim,),
            target_joint_vel=(self.control_dim,)
        )

    def _to_tensor(self, input):
        if th.is_tensor(input):
            return input.to(th.Tensor())
        else:
            return th.tensor(input).to(th.Tensor())

    def _diagonalize_gain(self, gain: th.Tensor) -> th.Tensor:
        if gain.dim() == 1:
            return th.diag(gain)
        elif gain.dim() == 2:
            return gain
        else:
            raise ValueError(f"Gain tensor must be 1D or 2D, but got {gain.dim()}D.")

    def is_grasping(self):
        return IsGraspingState.UNKNOWN

    @property
    def motor_type(self):
        return self._motor_type

    @property
    def control_type(self):
        return ControlType.EFFORT

    @property
    def command_dim(self):
        return len(self.dof_idx)