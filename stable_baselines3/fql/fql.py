from typing import Any, ClassVar, Optional, TypeVar, Union
from copy import deepcopy

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.fql.policies import LatentFQLPolicy

from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    RolloutReturn,
    Schedule,
    TrainFreq,
    TrainFrequencyUnit,
)
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv


SelfFQL = TypeVar("SelfFQL", bound="FQL")


class FQL(OffPolicyAlgorithm):
    """
    Latent-space Flow Q-Learning implementation.
    Built on Stable Baselines3 OffPolicyAlgorithm, adapted for DSRL/latent FQL setups.
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "LatentFQLPolicy": LatentFQLPolicy,
        "MlpLatentFQLPolicy": LatentFQLPolicy,
    }

    policy: LatentFQLPolicy

    def __init__(
		self,
		policy: Union[str, type[LatentFQLPolicy]],
		env: Union[GymEnv, str],
		learning_rate: Union[float, Schedule] = 3e-4,
		buffer_size: int = 1_000_000,  # 1e6
		learning_starts: int = 100,
		batch_size: int = 256,
		tau: float = 0.005,
		gamma: float = 0.99,
		train_freq: Union[int, tuple[int, str]] = 1,
		gradient_steps: int = 1,
		action_noise: Optional[ActionNoise] = None,
		replay_buffer_class: Optional[type[ReplayBuffer]] = None,
		replay_buffer_kwargs: Optional[dict[str, Any]] = None,
		optimize_memory_usage: bool = False,
		ent_coef: Union[str, float] = "auto",
		target_update_interval: int = 1,
		target_entropy: Union[str, float] = "auto",
		use_sde: bool = False,
		sde_sample_freq: int = -1,
		use_sde_at_warmup: bool = False,
		stats_window_size: int = 100,
		tensorboard_log: Optional[str] = None,
		policy_kwargs: Optional[dict[str, Any]] = None,
		verbose: int = 0,
		seed: Optional[int] = None,
		device: Union[th.device, str] = "auto",
		_init_setup_model: bool = True,
		diffusion_policy=None,
		diffusion_act_dim=None,
		noise_critic_grad_steps: int = 1,
		critic_backup_combine_type='min',
        sde_support: bool = False,
	):
        super().__init__(
			policy,
			env,
			learning_rate,
			buffer_size,
			learning_starts,
			batch_size,
			tau,
			gamma,
			train_freq,
			gradient_steps,
			action_noise,
			replay_buffer_class=replay_buffer_class,
			replay_buffer_kwargs=replay_buffer_kwargs,
			policy_kwargs=policy_kwargs,
			stats_window_size=stats_window_size,
			tensorboard_log=tensorboard_log,
			verbose=verbose,
			device=device,
			seed=seed,
			use_sde=use_sde,
			sde_sample_freq=sde_sample_freq,
			use_sde_at_warmup=use_sde_at_warmup,
			optimize_memory_usage=optimize_memory_usage,
			supported_action_spaces=(spaces.Box,),
			support_multi_env=True,
            sde_support=sde_support
		)

        self.q_target = None
        self.target_update_interval = target_update_interval

        self.diffusion_policy = diffusion_policy
        self.diffusion_act_chunk = diffusion_act_dim[0]
        self.diffusion_act_dim = diffusion_act_dim[1]
        self.train_freq = train_freq

        if _init_setup_model:
            self._setup_model()

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    def _setup_model(self) -> None:
        super()._setup_model()
        assert isinstance(self.policy, LatentFQLPolicy)

    def _create_aliases(self) -> None:
        pass

    # -------------------------------------------------------------------------
    # Training Loop (Fixed version)
    # -------------------------------------------------------------------------
    def train(self, gradient_steps: int, batch_size: int = 256) -> None:
        self.policy.set_training_mode(True)

        # 통계 수집용
        q_losses, flow_losses, q_values, target_q_values = [], [], [], []
        delta_norms, reg_values = [], []

        for step in range(gradient_steps):
            rb = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            s = th.as_tensor(rb.observations, device=self.device, dtype=th.float32)
            s2 = th.as_tensor(rb.next_observations, device=self.device, dtype=th.float32)
            r = th.as_tensor(rb.rewards, device=self.device, dtype=th.float32).squeeze(-1)
            d = th.as_tensor(rb.dones, device=self.device, dtype=th.float32).squeeze(-1)

            # ===== 1. Critic Update =====
            z = th.randn(batch_size, self.policy.z_dim, device=self.device)
            q_pred = self.policy.q_forward(s, z)
            
            # TD Target 계산
            with th.no_grad():
                z2 = th.randn(batch_size, self.policy.z_dim, device=self.device)
                z2p = self.policy.flow_forward(s2, z2)
                
                # Target Q-value clipping (CRITICAL)
                target_q = self.policy.q_forward(s2, z2p, use_target=True)
                target_q = th.clamp(target_q, -100, 100)
                y = r + (1 - d) * self.gamma * target_q
                y = th.clamp(y, -100, 100)
            
            # Huber Loss (outlier 저항성)
            loss_q = F.smooth_l1_loss(q_pred, y)
            
            self.policy.q_optimizer.zero_grad(set_to_none=True)
            loss_q.backward()
            th.nn.utils.clip_grad_norm_(self.policy.q_net.parameters(), max_norm=1.0)
            self.policy.q_optimizer.step()
            
            q_losses.append(loss_q.item())
            q_values.append(q_pred.mean().item())
            target_q_values.append(target_q.mean().item())
            
            for _ in range(self.train_freq if isinstance(self.train_freq, int) else 1):
                # ===== 2. Flow Update =====
                # Q network gradient 비활성화 (안정화)
                for p in self.policy.q_net.parameters():
                    p.requires_grad_(False)
                
                z_fresh = th.randn(batch_size, self.policy.z_dim, device=self.device)
                zp = self.policy.flow_forward(s, z_fresh)
                qv = self.policy.q_forward(s, zp)
                
                # Regularization
                reg = (zp - z_fresh).pow(2).mean()
                
                # Q-value normalization (CRITICAL for stability)
                denom = qv.abs().mean().detach() + 1e-6
                qv_normalized = qv / denom
                loss_flow = (-qv_normalized).mean() + self.policy.alpha * reg
                
                self.policy.flow_optimizer.zero_grad(set_to_none=True)
                loss_flow.backward()
                th.nn.utils.clip_grad_norm_(self.policy.flow_net.parameters(), max_norm=0.5)
                self.policy.flow_optimizer.step()
                
                # Q network gradient 재활성화
                for p in self.policy.q_net.parameters():
                    p.requires_grad_(True)
                
                flow_losses.append(loss_flow.item())
                delta_norms.append((zp - z_fresh).norm(dim=-1).mean().item())
                reg_values.append(reg.item())

            # ===== 3. Target Network Update =====
            if step % self.target_update_interval == 0:
                self.policy.update_target_network(self.tau)

            # ===== Debug Logging =====
            if step % 100 == 0 and self.verbose > 0:
                print(
                    f"[FQL] step={step:04d} | "
                    f"loss_q={loss_q.item():.4f} | loss_flow={loss_flow.item():.4f} | "
                    f"Δz={delta_norms[-1]:.3f} | Q={q_values[-1]:.2f} | "
                    f"target_Q={target_q_values[-1]:.2f} | R={r.mean().item():.3f}"
                )

        # Epoch 통계 (DSRL 스타일)
        self._n_updates += gradient_steps
        
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/q_loss", np.mean(q_losses))
        self.logger.record("train/flow_loss", np.mean(flow_losses))
        self.logger.record("train/q_value", np.mean(q_values))
        self.logger.record("train/target_q_value", np.mean(target_q_values))
        self.logger.record("train/delta_z_norm", np.mean(delta_norms))
        self.logger.record("train/regularization", np.mean(reg_values))

    # -------------------------------------------------------------------------
    # Action Sampling (DSRL-compatible)
    # -------------------------------------------------------------------------
    def _act_from_latent(self, obs_tensor: th.Tensor) -> tuple:
        """
        DSRL 스타일 action generation
        """
        B = obs_tensor.shape[0]
        
        # 1. Sample latent noise
        z = th.randn(B, self.policy.z_dim, device=self.device)
        
        with th.no_grad():
            # 2. Flow steering
            zp = self.policy.flow_forward(obs_tensor, z)
            # Latent clipping
            zp = th.clamp(zp, -5, 5)
        
        # 3. Reshape for diffusion policy (DSRL pattern)
        zp_input = zp.reshape(B, self.diffusion_act_chunk, self.diffusion_act_dim)
        
        # 4. Decode through frozen diffusion policy
        try:
            diffused_action = self.diffusion_policy(
                obs_tensor,
                zp_input,
                return_numpy=False,
            )
            diffused_action = diffused_action.reshape(B, self.diffusion_act_chunk * self.diffusion_act_dim)
        except Exception as e:
            print(f"[ERROR] Diffusion decode failed: {e}")
            print(f"  obs: {obs_tensor.shape}, zp_input: {zp_input.shape}")
            raise
        
        action = diffused_action.cpu().numpy()
        return action, zp

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        DSRL-style action sampling
        """
        assert self._last_obs is not None
        
        # Warmup: random action through diffusion
        if self.num_timesteps < learning_starts:
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
            
            action_tensor = th.as_tensor(unscaled_action, device=self.device, dtype=th.float32)
            obs_tensor = th.as_tensor(self._last_obs, device=self.device, dtype=th.float32)
            
            action_reshaped = action_tensor.reshape(n_envs, self.diffusion_act_chunk, self.diffusion_act_dim)
            with th.no_grad():
                diffused_action = self.diffusion_policy(obs_tensor, action_reshaped, return_numpy=False)
            diffused_action = diffused_action.reshape(n_envs, self.diffusion_act_chunk * self.diffusion_act_dim)
            action = diffused_action.cpu().numpy()
        else:
            # Training: use flow policy
            obs = th.as_tensor(self._last_obs, device=self.device, dtype=th.float32)
            action, _ = self._act_from_latent(obs)
        
        buffer_action = action
        return action, buffer_action

    def predict_diffused(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        DSRL-style prediction for evaluation
        """
        obs = th.as_tensor(observation, device=self.device, dtype=th.float32)
        action, _ = self._act_from_latent(obs)
        return action, state

    # -------------------------------------------------------------------------
    # Learning Wrapper
    # -------------------------------------------------------------------------
    def learn(
        self: SelfFQL,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "FQL",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfFQL:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    # -------------------------------------------------------------------------
    # Saving / Loading (DSRL-style)
    # -------------------------------------------------------------------------
    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params()

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "policy.flow_optimizer", "policy.q_optimizer"]
        saved_pytorch_variables: list[str] = []
        return state_dicts, saved_pytorch_variables
