import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math import wrap_to_pi
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.gs_utils import *
from .go2_deploy_config import GO2DeployCfg

class GO2Deploy(LeggedRobot):
    def __init__(self, cfg: GO2DeployCfg, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.height_samples = None
        self.debug_viz = self.cfg.env.debug_viz
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_device, headless)

        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
    
    def get_observations(self):
        return self.estimator_input_buf, self.critic_obs_buf
    
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        if self.cfg.sim.use_implicit_controller:
            target_dof_pos = self._compute_target_dof_pos(exec_actions)
            self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
            self.scene.step()
        else:
            for _ in range(self.cfg.control.decimation):
                self.torques = self._compute_torques(exec_actions)
                if self.num_build_envs == 0:
                    torques = self.torques.squeeze()
                    self.robot.control_dofs_force(torques, self.motor_dofs)
                else:
                    self.robot.control_dofs_force(self.torques, self.motor_dofs)
                self.scene.step()
                self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
                self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.estimator_input_buf = torch.clip(self.estimator_input_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.estimator_input_buf, self.estimator_true_value, self.critic_obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        estimator_input, estimator_true_value, critic_obs_buf, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return estimator_input, estimator_true_value, critic_obs_buf
    
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        base_quat_rel = gs_quat_mul(self.base_quat, gs_inv_quat(self.base_init_quat.reshape(1, -1).repeat(self.num_envs, 1)))
        self.base_euler = gs_quat2euler(base_quat_rel)
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat) # trasform to base frame
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.link_contact_forces[:] = torch.tensor(
            self.robot.get_links_net_contact_force(),
            device=self.device,
            dtype=gs.tc_float,
        )
        
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if self.num_build_envs > 0:
            self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.l3ast_dof_pos[:] = self.llast_dof_pos[:]
        self.llast_dof_pos[:] = self.last_dof_pos[:]
        self.last_dof_pos[:] = self.dof_pos[:]
        self.l3ast_dof_vel[:] = self.llast_dof_vel[:]
        self.llast_dof_vel[:] = self.last_dof_vel[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        if self.debug_viz:
            self._draw_debug_vis()
    
    def compute_observations(self):
        """ Computes observations
        """
        self.estimator_input_buf = torch.cat((  self.commands[:, :3] * self.commands_scale,                # cmd 3 
                                    self.projected_gravity,                                                # g 3
                                    self.base_ang_vel  * self.obs_scales.ang_vel,                          # omega 3
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,       # p_t 12
                                    self.dof_vel * self.obs_scales.dof_vel,                                # dp_t 12
                                    (self.last_dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # p_{t-dt} 12
                                    (self.llast_dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # p_{t-2dt} 12
                                    (self.l3ast_dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # p_{t-3dt} 12
                                    self.last_dof_vel * self.obs_scales.dof_vel,                           # dp_{t-dt} 12
                                    self.llast_dof_vel * self.obs_scales.dof_vel,                          # dp_{t-2dt} 12
                                    self.l3ast_dof_vel * self.obs_scales.dof_vel,                          # dp_{t-3dt} 12
                                    self.actions,                                                          # a_{t-dt} 12
                                    self.last_actions,                                                     # a_{t-2dt} 12
                                    ),dim=-1)
        
        # add perceptive inputs if not blind
        # if self.cfg.terrain.measure_heights:
        #     heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
        #     self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        
        self.estimator_true_value = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,       # 3
                                               (self.link_contact_forces[:, self.feet_indices, 2]/1.0).clip(min=0.,max=1.), # 4
                                              ), dim=-1)
        
        if self.num_privileged_obs is not None: # critic_obs, no noise
            self.privileged_obs_buf = torch.cat((self.estimator_input_buf, self.estimator_true_value), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.estimator_input_buf += (2 * torch.rand_like(self.estimator_input_buf) - 1) * self.noise_scale_vec
        # normal critic observation is with noise
        self.critic_obs_buf = torch.cat((self.estimator_input_buf, self.estimator_true_value), dim=-1)
    
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # # update curriculum
        # if self.cfg.terrain.curriculum:
        #     self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)
        
        # domain randomization
        if self.cfg.domain_rand.randomize_friction:
            self._randomize_friction(env_ids)
        if self.cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(env_ids)
        if self.cfg.domain_rand.randomize_com_displacement:
            self._randomize_com_displacement(env_ids)

        # reset buffers
        self.last_dof_pos[env_ids] = 0.
        self.llast_dof_pos[env_ids] = 0.
        self.l3ast_dof_pos[env_ids] = 0.
        self.llast_actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.llast_dof_vel[env_ids] = 0.
        self.l3ast_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    
    #------------- Callbacks --------------
    
    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.estimator_input_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = 0. # commands
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[9:9+1*self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos                    # p_t
        noise_vec[9+1*self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel # dp_t
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos # p_{t-dt}
        noise_vec[9+3*self.num_actions:9+4*self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos # p_{t-2dt}
        noise_vec[9+4*self.num_actions:9+5*self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos # p_{t-3dt}
        noise_vec[9+5*self.num_actions:9+6*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel # dp_{t-dt}
        noise_vec[9+6*self.num_actions:9+7*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel # dp_{t-2dt}
        noise_vec[9+7*self.num_actions:9+8*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel # dp_{t-3dt}
        noise_vec[9+8*self.num_actions:9+9*self.num_actions] = 0.  # a_{t-dt}
        noise_vec[9+9*self.num_actions:9+10*self.num_actions] = 0. # a_{t-2dt}
        
        # if self.cfg.terrain.measure_heights:
        #     noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec
    
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        self.common_step_counter = 0
        self.extras = {}
        
        self.forward_vec = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.forward_vec[:, 0] = 1.0
        self.base_init_pos = torch.tensor(
            self.cfg.init_state.pos, device=self.device
        )
        self.base_init_quat = torch.tensor(
            self.cfg.init_state.rot, device=self.device
        )
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.cfg.commands.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], 
            device=self.device,
            dtype=gs.tc_float, 
            requires_grad=False,) # TODO change this
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.llast_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.feet_air_time = torch.zeros((self.num_envs, len(self.feet_indices)), device=self.device, dtype=gs.tc_float)
        self.last_contacts = torch.zeros((self.num_envs, len(self.feet_indices)), device=self.device,dtype=gs.tc_int)
        self.link_contact_forces = torch.zeros((self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float)
        self.continuous_push = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.env_identities = torch.arange(self.num_envs, device=self.device, dtype=gs.tc_int)
        # For OnPolicyRunnerEE
        self.estimator_input_buf = torch.zeros(self.num_envs, self.num_estimator_input, device=self.device, dtype=gs.tc_float)
        self.critic_obs_buf = torch.zeros(self.num_envs, self.num_critic_obs, device=self.device, dtype=gs.tc_float)
        # History Observation Buffers
        self.last_dof_pos = torch.zeros_like(self.actions)
        self.llast_dof_pos = torch.zeros_like(self.actions)
        self.l3ast_dof_pos = torch.zeros_like(self.actions)
        self.llast_dof_vel = torch.zeros_like(self.actions)
        self.l3ast_dof_vel = torch.zeros_like(self.actions)
        
        self.noise_scale_vec = self._get_noise_scale_vec()
        
        # if self.cfg.terrain.measure_heights:
        #     self.height_points = self._init_height_points()
        # self.measured_heights = 0

        self.default_dof_pos = torch.tensor(
            [self.cfg.init_state.default_joint_angles[name] for name in self.cfg.asset.dof_names],
            device=self.device,
            dtype=gs.tc_float,
        )
        
        # PD control
        stiffness = self.cfg.control.stiffness
        damping = self.cfg.control.damping
        
        self.p_gains, self.d_gains = [], []
        for dof_name in self.cfg.asset.dof_names:
            for key in stiffness.keys():
                if key in dof_name:
                    self.p_gains.append(stiffness[key])
                    self.d_gains.append(damping[key])
        self.p_gains = torch.tensor(self.p_gains, device=self.device)
        self.d_gains = torch.tensor(self.d_gains, device=self.device)
        self.batched_p_gains = self.p_gains[None, :].repeat(self.num_envs, 1)
        self.batched_d_gains = self.d_gains[None, :].repeat(self.num_envs, 1)
        # PD control params
        self.robot.set_dofs_kp(self.p_gains, self.motor_dofs)
        self.robot.set_dofs_kv(self.d_gains, self.motor_dofs)
    
    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.dt
        if self.cfg.sim.use_implicit_controller: # use embedded PD controller
            self.sim_dt = self.dt
            self.sim_substeps = self.cfg.control.decimation
        else: # use explicit PD controller
            self.sim_dt = self.dt / self.cfg.control.decimation
            self.sim_substeps = 1
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.push_interval_s = self.cfg.domain_rand.push_interval_s

        self.dof_names = self.cfg.asset.dof_names
        self.simulate_action_latency = self.cfg.domain_rand.simulate_action_latency
        self.debug = self.cfg.env.debug
        # For OnPolicyRunnerEE
        self.num_estimator_input = self.cfg.env.num_estimator_input
        self.num_estimator_output = self.cfg.env.num_estimator_output
        self.num_critic_obs = self.cfg.env.num_critic_obs
        self.num_actor_obs = self.cfg.env.num_actor_obs
    