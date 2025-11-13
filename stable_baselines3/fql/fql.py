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
		actor_gradient_steps: int = -1,
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

        if _init_setup_model:
            self._setup_model()
        
        self._last_latent = None


    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    def _setup_model(self) -> None:
        super()._setup_model()
        assert isinstance(self.policy, LatentFQLPolicy)
        self.q_target = deepcopy(self.policy.q_net).to(self.device)
        for p in self.q_target.parameters():
            p.requires_grad_(False)

    def _create_aliases(self) -> None:
        pass

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------

    def train(self, gradient_steps: int, batch_size: int = 256) -> None:
        self.policy.set_training_mode(True)

        for step in range(gradient_steps):
            rb = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            s = th.as_tensor(rb.observations, device=self.device, dtype=th.float32)
            s2 = th.as_tensor(rb.next_observations, device=self.device, dtype=th.float32)
            r = th.as_tensor(rb.rewards, device=self.device, dtype=th.float32).squeeze(-1)
            d = th.as_tensor(rb.dones, device=self.device, dtype=th.float32).squeeze(-1)

            # ===== Sample latent actions from info =====
            #TODO
            zb_np = rb.infos.get('latent', None)
            assert zb_np is not None, "[FQL] Latent actions not found in replay buffer infos."
            zb = th.as_tensor(zb_np, device=self.device, dtype=th.float32)

            # ===== Critic TD Target =====
            with th.no_grad():
                z2 = th.randn(batch_size, self.policy.z_dim, device=self.device)
                z2p = self.policy.flow_forward(s2, z2)
                y = r + (1 - d) * self.gamma * self.policy.q_forward(s2, z2p, use_target=True)

            # ===== Critic Loss =====
            q_pred = self.policy.q_forward(s, zb)
            loss_q = F.mse_loss(q_pred, y)
            self.policy.q_optimizer.zero_grad(set_to_none=True)
            loss_q.backward()
            self.policy.q_optimizer.step()

            # ===== Flow Update =====
            z = th.randn(batch_size, self.policy.z_dim, device=self.device)
            zp = self.policy.flow_forward(s, z)
            qv = self.policy.q_forward(s, zp)
            reg = (zp - z).pow(2).mean()
            loss_flow = (-qv).mean() + self.policy.alpha * reg
            self.policy.flow_optimizer.zero_grad(set_to_none=True)
            loss_flow.backward()
            self.policy.flow_optimizer.step()

            # ===== Target Update =====
            polyak_update(self.policy.q_net.parameters(), self.q_target.parameters(), self.tau)

            # ===== Debug Logging =====
            if step % 10 == 0:
                delta_norm = (zp - z).norm(dim=-1).mean().item()
                q_mean, q_std = q_pred.mean().item(), q_pred.std().item()
                print(
                    f"[FQL Debug] step={step:03d} | "
                    f"loss_q={loss_q.item():.4f} | loss_flow={loss_flow.item():.4f} | "
                    f"Δz_norm={delta_norm:.3f} | Qμ={q_mean:.3f}±{q_std:.3f}"
                )

        self._n_updates += gradient_steps


    # -------------------------------------------------------------------------
    # Action sampling
    # -------------------------------------------------------------------------
    def _act_from_latent(self, obs_tensor: th.Tensor) -> np.ndarray:
        B = obs_tensor.shape[0]
        z = th.randn(B, self.policy.z_dim, device=self.device) # TODO:latent critic Q와 동일한 z 사용하도록
        zp = self.policy.flow_forward(obs_tensor, z).detach()
        a = self.diffusion_policy(
            obs_tensor,
            zp.view(B, 1, -1),  # (B, Ta, Dz)
            return_numpy=False,
        )
        a  = a.view(B, self.diffusion_act_chunk, self.diffusion_act_dim)
        return a.cpu().numpy(), zp

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ):
        assert self._last_obs is not None
        obs = th.as_tensor(self._last_obs, device=self.device, dtype=th.float32)
        action, zprime = self._act_from_latent(obs)
        self._last_latent = zprime.detach().cpu().numpy()
        print("[FQL] Sampled latent:", self._last_latent)
        buffer_action = action
        return action, buffer_action

    def predict_diffused(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic=False,
    ):
        obs = th.as_tensor(observation, device=self.device, dtype=th.float32)
        action_chunked, _ = self._act_from_latent(obs)
        return action_chunked, state

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
    # Saving / Loading
    # -------------------------------------------------------------------------
    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params()

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "policy.flow_optimizer", "policy.q_optimizer"]
        saved_pytorch_variables: list[str] = []
        return state_dicts, saved_pytorch_variables
