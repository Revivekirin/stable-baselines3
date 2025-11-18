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

SelfFQL = TypeVar("SelfFQL", bound="FQL")


class FQL(OffPolicyAlgorithm):
    """
    Latent-space Flow Q-Learning with Offline/Online phases.
    - Offline: Train flow model with offline data
    - Online: Distill to student model (or fine-tune critic only)
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
        buffer_size: int = 1_000_000,
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
        target_update_interval: int = 1,
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
        # FQL-specific parameters
        offline_steps: int = 1000000,  # Extract from policy_kwargs if provided
        online_steps: int = 1000000,
        distillation_coef: float = 1.0,
        online_q_coef: float = 1.0,
        sde_support: bool = False,
    ):
        # Extract offline_steps from policy_kwargs if provided there
        if policy_kwargs is not None and 'offline_steps' in policy_kwargs:
            offline_steps = policy_kwargs.pop('offline_steps')
            if verbose > 0:
                print(f"[FQL] Using offline_steps={offline_steps} from policy_kwargs")

        if policy_kwargs is not None and 'online_steps' in policy_kwargs:
            online_steps = policy_kwargs.pop('online_steps')
            if verbose > 0:
                print(f"[FQL] Using online_steps={online_steps} from policy_kwargs")
        
        
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

        self.target_update_interval = target_update_interval
        self.diffusion_policy = diffusion_policy
        self.diffusion_act_chunk = diffusion_act_dim[0]
        self.diffusion_act_dim = diffusion_act_dim[1]

        # Offline/Online phase control
        self.offline_steps = offline_steps
        self.online_steps = online_steps
        self.distillation_coef = distillation_coef
        self.online_q_coef = online_q_coef
        
        # Anchor model for distillation (frozen after offline phase)
        self.anchor_flow_net = None
        
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        assert isinstance(self.policy, LatentFQLPolicy)

    def _create_aliases(self) -> None:
        pass

    # -------------------------------------------------------------------------
    # Phase Detection
    # -------------------------------------------------------------------------
    def _is_offline_phase(self) -> bool:
        """Check if we're in offline training phase"""
        return self.num_timesteps <= self.offline_steps

    def _is_online_phase(self) -> bool:
        """Check if we're in online fine-tuning phase"""
        return self.num_timesteps > self.offline_steps

    def _should_freeze_anchor(self) -> bool:
        """Check if we should freeze anchor model (at transition point)"""
        # Freeze when we FIRST enter online phase
        return (self.num_timesteps > self.offline_steps and 
                self.anchor_flow_net is None)

    # -------------------------------------------------------------------------
    # Training Loop (Offline/Online Split)
    # -------------------------------------------------------------------------
    def train(self, gradient_steps: int, batch_size: int = 256) -> None:
        self.policy.set_training_mode(True)

        # Freeze anchor model at offline→online transition
        if self._should_freeze_anchor():
            self._freeze_anchor_model()
            self._print_phase_transition()

        # Statistics collection
        q_losses, flow_losses, distill_losses = [], [], []
        q_values, target_q_values = [], []
        delta_norms, reg_values = [], []

        for step in range(gradient_steps):
            rb = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            s = th.as_tensor(rb.observations, device=self.device, dtype=th.float32)
            s2 = th.as_tensor(rb.next_observations, device=self.device, dtype=th.float32)
            r = th.as_tensor(rb.rewards, device=self.device, dtype=th.float32).squeeze(-1)
            d = th.as_tensor(rb.dones, device=self.device, dtype=th.float32).squeeze(-1)

            # ===== CRITIC UPDATE (Both phases) =====
            actions = th.as_tensor(rb.actions, device=self.device, dtype=th.float32)

            q_pred = self.policy.q_forward(s, actions)   # Q(s,a)

            with th.no_grad():
                z2 = th.randn(batch_size, self.policy.z_dim, device=self.device)
                w2 = self.policy.flow_forward(s2, z2)               
                w2 = th.clamp(w2, -5, 5)
                w2_input = w2.view(batch_size, self.diffusion_act_chunk, self.diffusion_act_dim)

                a2 = self.diffusion_policy(s2, w2_input, return_numpy=False)
                a2 = a2.view(batch_size, self.diffusion_act_chunk * self.diffusion_act_dim)

                target_q = self.policy.q_forward(s2, a2, use_target=True)
                target_q = th.clamp(target_q, -100, 100)
                y = r + (1 - d) * self.gamma * target_q
                y = th.clamp(y, -100, 100)

            loss_q = F.smooth_l1_loss(q_pred, y)
            
            self.policy.q_optimizer.zero_grad(set_to_none=True)
            loss_q.backward()
            th.nn.utils.clip_grad_norm_(self.policy.q_net.parameters(), max_norm=1.0)
            self.policy.q_optimizer.step()
            
            q_losses.append(loss_q.item())
            q_values.append(q_pred.mean().item())
            target_q_values.append(target_q.mean().item())

            # ===== FLOW/ACTOR UPDATE (Phase-dependent) =====
            if self._is_offline_phase():
                # OFFLINE: Train flow model with Q-maximization
                loss_flow = self._offline_flow_update(s, batch_size)
                flow_losses.append(loss_flow.item())
                
            else:
                # ONLINE: Distillation from frozen anchor
                loss_distill = self._online_distillation_update(s, batch_size)
                distill_losses.append(loss_distill.item())

            # Target network update
            if step % self.target_update_interval == 0:
                self.policy.update_target_network(self.tau)

            # Debug logging with progress bar
            if step % 100 == 0 and self.verbose > 0:
                self._print_training_status(
                    step, loss_q, 
                    flow_losses[-1] if flow_losses else None,
                    distill_losses[-1] if distill_losses else None,
                    q_values[-1]
                )

        # Logging
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/q_loss", np.mean(q_losses))
        self.logger.record("train/q_value", np.mean(q_values))
        self.logger.record("train/target_q_value", np.mean(target_q_values))
        
        if self._is_offline_phase():
            self.logger.record("train/flow_loss", np.mean(flow_losses))
            self.logger.record("train/phase", 0)  # 0 = offline
        else:
            self.logger.record("train/distill_loss", np.mean(distill_losses))
            self.logger.record("train/phase", 1)  # 1 = online

    # -------------------------------------------------------------------------
    # Offline Flow Update
    # -------------------------------------------------------------------------
    def _offline_flow_update(self, s: th.Tensor, batch_size: int) -> th.Tensor:
        # Q -> fix, flow-> update
        for p in self.policy.q_net.parameters():
            p.requires_grad_(False)

        z = th.randn(batch_size, self.policy.z_dim, device=self.device)
        w = self.policy.flow_forward(s, z)               # flow(s,z)
        w = th.clamp(w, -5, 5)
        w_input = w.view(batch_size, self.diffusion_act_chunk, self.diffusion_act_dim)

        a = self.diffusion_policy(s, w_input, return_numpy=False)
        a = a.view(batch_size, self.diffusion_act_chunk * self.diffusion_act_dim)

        qv = self.policy.q_forward(s, a)                 # Q(s,a)

        # regularization on latent movement
        reg = (w - z).pow(2).mean()

        denom = qv.abs().mean().detach() + 1e-6
        qv_normalized = qv / denom
        loss_flow = (-qv_normalized).mean() + self.policy.alpha * reg

        self.policy.flow_optimizer.zero_grad(set_to_none=True)
        loss_flow.backward()
        th.nn.utils.clip_grad_norm_(self.policy.flow_net.parameters(), max_norm=0.5)
        self.policy.flow_optimizer.step()

        for p in self.policy.q_net.parameters():
            p.requires_grad_(True)

        return loss_flow


    # -------------------------------------------------------------------------
    # Online Distillation Update
    # -------------------------------------------------------------------------
    def _online_distillation_update(self, s: th.Tensor, batch_size: int) -> th.Tensor:
        assert self.anchor_flow_net is not None, "Anchor model not frozen!"
        
        for p in self.policy.q_net.parameters():
            p.requires_grad_(False)

        z = th.randn(batch_size, self.policy.z_dim, device=self.device)

        # student flow
        w_student = self.policy.flow_forward(s, z)
        w_student = th.clamp(w_student, -5, 5)
        w_student_in = w_student.view(batch_size, self.diffusion_act_chunk, self.diffusion_act_dim)

        # teacher(flow anchor)
        with th.no_grad():
            w_anchor = self._forward_anchor(s, z)

        # decode student to action
        a_student = self.diffusion_policy(s, w_student_in, return_numpy=False)
        a_student = a_student.view(batch_size, self.diffusion_act_chunk * self.diffusion_act_dim)

        qv = self.policy.q_forward(s, a_student)
        denom = qv.abs().mean().detach() + 1e-6
        qv_normalized = qv / denom
        loss_q = (-qv_normalized).mean()

        # distillation in latent space
        loss_distill = F.mse_loss(w_student, w_anchor)

        loss_flow = self.online_q_coef * loss_q + self.distillation_coef * loss_distill

        self.policy.flow_optimizer.zero_grad(set_to_none=True)
        loss_flow.backward()
        th.nn.utils.clip_grad_norm_(self.policy.flow_net.parameters(), max_norm=0.5)
        self.policy.flow_optimizer.step()

        for p in self.policy.q_net.parameters():
            p.requires_grad_(True)

        return loss_flow


    # -------------------------------------------------------------------------
    # Anchor Model Management
    # -------------------------------------------------------------------------
    def _freeze_anchor_model(self) -> None:
        """Freeze anchor model at offline→online transition"""
        print("\n" + "="*80)
        print("[FQL] 🎯 PHASE TRANSITION: Freezing anchor model for distillation...")
        print("="*80 + "\n")
        self.anchor_flow_net = deepcopy(self.policy.flow_net)
        for param in self.anchor_flow_net.parameters():
            param.requires_grad_(False)
        self.anchor_flow_net.eval()

    def _forward_anchor(self, obs: th.Tensor, z: th.Tensor) -> th.Tensor:
        """Forward pass through frozen anchor"""
        return self.anchor_flow_net(obs, z)

    # -------------------------------------------------------------------------
    # Progress Monitoring
    # -------------------------------------------------------------------------
    def _print_phase_transition(self) -> None:
        """Print visual phase transition message"""
        print("\n" + "━"*80)
        print("🔄 TRANSITIONING: OFFLINE → ONLINE")
        print(f"   Steps completed: {self.num_timesteps:,} / {self.offline_steps:,}")
        print(f"   Anchor model frozen ✓")
        print(f"   Starting online distillation phase...")
        print("━"*80 + "\n")

    def _print_training_status(
        self, 
        step: int, 
        loss_q: th.Tensor, 
        loss_flow: Optional[float],
        loss_distill: Optional[float],
        q_value: float
    ) -> None:
        """Print detailed training status with progress bar"""
        phase = "OFFLINE" if self._is_offline_phase() else "ONLINE"
        
        # Calculate progress
        if self._is_offline_phase():
            current = self.num_timesteps
            total = self.offline_steps
            phase_name = "Offline Training"
            emoji = "📚"
        else:
            current = self.num_timesteps - self.offline_steps
            total = self.online_steps  # This will be updated by callback
            phase_name = "Online Distillation"
            emoji = "🚀"
        
        progress_pct = min(100.0, (current / max(total, 1)) * 100)
        
        # Progress bar
        bar_length = 30
        filled = int(bar_length * progress_pct / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        # Construct message
        print(f"\r{emoji} {phase_name} [{bar}] {progress_pct:.1f}% ", end="")
        print(f"| Steps: {current:,}/{total:,} ", end="")
        print(f"| Q_loss: {loss_q.item():.4f} ", end="")
        
        if loss_flow is not None:
            print(f"| Flow_loss: {loss_flow:.4f} ", end="")
        elif loss_distill is not None:
            print(f"| Distill_loss: {loss_distill:.4f} ", end="")
        
        print(f"| Q_val: {q_value:.2f}", end="", flush=True)

    def _get_phase_info(self) -> dict:
        """Get current phase information for logging"""
        if self._is_offline_phase():
            return {
                "phase": "offline",
                "phase_step": self.num_timesteps,
                "phase_progress": self.num_timesteps / self.offline_steps,
            }
        else:
            online_steps = self.num_timesteps - self.offline_steps
            return {
                "phase": "online",
                "phase_step": online_steps,
                "phase_progress": 1.0,  # Online phase progress depends on total_timesteps
            }

    # -------------------------------------------------------------------------
    # Action Sampling (Same as before)
    # -------------------------------------------------------------------------
    def _act_from_latent(self, obs_tensor: th.Tensor) -> tuple:
        """DSRL-style action generation"""
        B = obs_tensor.shape[0]
        z = th.randn(B, self.policy.z_dim, device=self.device)
        
        with th.no_grad():
            zp = self.policy.flow_forward(obs_tensor, z)
            zp = th.clamp(zp, -5, 5)
            zp_input = zp.reshape(B, self.diffusion_act_chunk, self.diffusion_act_dim)
            diffused_action = self.diffusion_policy(
            obs_tensor,
            zp_input,
            return_numpy=False,
        )
        
        
        diffused_action = diffused_action.reshape(B, self.diffusion_act_chunk * self.diffusion_act_dim)
        
        action = diffused_action.cpu().numpy()
        return action, zp

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """DSRL-style action sampling"""
        assert self._last_obs is not None
        
        if self.num_timesteps < learning_starts:
            # Warmup: random actions
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
        
        return action, action

    def predict_diffused(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """DSRL-style prediction for evaluation"""
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
        # Print initial training info
        self._print_training_info(total_timesteps)
        
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _print_training_info(self, total_timesteps: int) -> None:
        """Print training configuration at start"""
        print("\n" + "="*80)
        print("🎓 FQL Training Configuration")
        print("="*80)
        print(f"Total timesteps:     {total_timesteps:,}")
        print(f"Offline steps:       {self.offline_steps:,} ({self.offline_steps/total_timesteps*100:.1f}%)")
        print(f"Online steps:        {total_timesteps - self.offline_steps:,} ({(total_timesteps-self.offline_steps)/total_timesteps*100:.1f}%)")
        print(f"Batch size:          {self.batch_size}")
        print(f"Gradient steps/env:  {self.gradient_steps}")
        print(f"Learning rate:       {self.learning_rate}")
        print(f"Distillation coef:   {self.distillation_coef}")
        print(f"Latent dim:          {self.policy.z_dim}")
        print(f"Alpha (reg):         {self.policy.alpha}")
        print("="*80 + "\n")
        print("📍 Phase 1: OFFLINE TRAINING (Flow + Critic from offline data)")
        print("📍 Phase 2: ONLINE DISTILLATION (Student learns from frozen anchor)\n")

    def _excluded_save_params(self) -> list[str]:
        excluded = super()._excluded_save_params()
        excluded.append("anchor_flow_net")  # Don't save anchor separately
        return excluded

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "policy.flow_optimizer", "policy.q_optimizer"]
        saved_pytorch_variables: list[str] = []
        return state_dicts, saved_pytorch_variables