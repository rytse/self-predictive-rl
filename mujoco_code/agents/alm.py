from typing import Any, Dict, Optional, Tuple

from omegaconf.dictconfig import DictConfig
import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from torch.linalg import matrix_rank, cond
import numpy as np
import numpy.typing as npt

import utils
from utils import logger
from utils.torch_utils import make_normalizer
from models import *


class AlmAgent(object):
    def __init__(
        self,
        device: torch.device,
        action_low: float,
        action_high: float,
        reward_low: float,
        reward_high: float,
        num_states: int,
        num_actions: int,
        env_buffer_size: int,
        cfg: DictConfig,
    ):
        self.device = device
        self.action_low = action_low
        self.action_high = action_high
        self.reward_low = reward_low
        self.reward_high = reward_high

        # key hparams
        self.disable_svg = cfg.disable_svg
        self.disable_reward = cfg.disable_reward
        self.freeze_critic = cfg.freeze_critic
        self.online_encoder_actorcritic = cfg.online_encoder_actorcritic
        if cfg.wass_norm_reward:
            self.reward_normalizer = make_normalizer(
                input_range=(reward_low, reward_high),
                output_range=(0.0, 1.0),
                backend="np",
            )
        else:
            self.reward_normalizer = lambda x: x
        if cfg.wass_norm_action:
            self.action_normalizer = make_normalizer(
                input_range=(action_low, action_high),
                output_range=(-1.0, 1.0),
                backend="torch",
            )
        else:
            self.action_normalizer = lambda x: x

        # Wasserstein critic
        assert (
            cfg.bisim_gamma >= 0.0 and cfg.bisim_gamma <= 1.0
        ), f"bisim_gamma should be in [0, 1], got {cfg.bisim_gamma}"
        self.bisim_gamma = cfg.bisim_gamma
        self.was_critic_train_steps = cfg.wass_critic_train_steps
        self.wass_deterministic = cfg.wass_deterministic

        # aux
        self.aux = cfg.aux
        self.aux_optim = cfg.aux_optim
        self.aux_coef_cfg = cfg.aux_coef
        if self.aux is None:
            self.aux_optim = None
            self.aux_coef_cfg = "v-0.0"
        assert self.aux in [
            "fkl",
            "rkl",
            "l2",
            "op-l2",
            "op-kl",
            "bisim",
            "bisim_critic",
            "zp_critic",
            None,
        ]
        assert self.aux_optim in ["ema", "detach", "online", None]

        # learning
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.target_update_interval = cfg.target_update_interval
        self.max_grad_norm = cfg.max_grad_norm
        self.batch_size = cfg.batch_size
        self.seq_len = cfg.seq_len
        self.lambda_cost = cfg.lambda_cost

        # exploration
        self.expl_start = cfg.expl_start
        self.expl_end = cfg.expl_end
        self.expl_duration = cfg.expl_duration
        self.stddev_clip = cfg.stddev_clip

        # logging
        self.log_interval = cfg.log_interval

        self.env_buffer = utils.ReplayMemory(
            env_buffer_size, num_states, num_actions, np.float32
        )
        self._init_networks(
            num_states,
            num_actions,
            cfg.latent_dims,
            cfg.hidden_dims,
            cfg.model_hidden_dims,
        )
        self._init_optims(cfg.lr)

    def _init_networks(
        self,
        num_states: int,
        num_actions: int,
        latent_dims: int,
        hidden_dims: int,
        model_hidden_dims: int,
    ) -> None:

        if self.aux == "bisim":
            EncoderClass, ModelClass = StoEncoder, StoModel
        elif self.aux in ["bisim_critic", "zp_critic"]:
            if self.wass_deterministic:
                EncoderClass, ModelClass = DetEncoder, DetModel
            else:
                EncoderClass, ModelClass = StoEncoder, StoModel
        else:
            if self.aux in [None, "l2", "op-l2"]:
                EncoderClass, ModelClass = DetEncoder, DetModel
            elif self.aux in ["fkl", "rkl", "op-kl"]:
                EncoderClass, ModelClass = StoEncoder, StoModel
            else:
                raise ValueError(self.aux)

        self.encoder = EncoderClass(num_states, hidden_dims, latent_dims).to(
            self.device
        )
        self.encoder_target = EncoderClass(num_states, hidden_dims, latent_dims).to(
            self.device
        )
        utils.hard_update(self.encoder_target, self.encoder)

        self.model = torch.compile(
            ModelClass(
                latent_dims,
                num_actions,
                model_hidden_dims,
                obs_dims=(
                    num_states if self.aux is not None and "op" in self.aux else None
                ),  # learn ZP or OP
            ).to(self.device),
            mode="default",
        )

        self.critic = Critic(latent_dims, hidden_dims, num_actions).to(self.device)
        self.critic_target = Critic(latent_dims, hidden_dims, num_actions).to(
            self.device
        )
        utils.hard_update(self.critic_target, self.critic)

        if self.aux == "bisim_critic":
            self.wass_critic = torch.compile(
                BisimWassCritic(latent_dims, num_actions, hidden_dims).to(self.device),
                mode="default",
            )
        elif self.aux == "zp_critic":
            self.wass_critic = torch.compile(
                ZPWassersteinCritic(latent_dims, num_actions, hidden_dims).to(
                    self.device
                ),
                mode="default",
            )
        self.actor = torch.compile(
            Actor(
                latent_dims, hidden_dims, num_actions, self.action_low, self.action_high
            ).to(self.device),
            mode="default",
        )

        self.world_model_list = [self.model, self.encoder]
        self.actor_list = [self.actor]
        self.critic_list = [self.critic]

        if self.disable_reward:
            assert self.seq_len == 1
            assert self.disable_svg == True
        else:
            self.reward = torch.compile(
                RewardPrior(latent_dims, hidden_dims, num_actions).to(self.device),
                mode="default",
            )
            self.classifier = torch.compile(
                Discriminator(latent_dims, hidden_dims, num_actions).to(self.device),
                mode="default",
            )
            self.reward_list = [self.reward, self.classifier]

        cfg, value = self.aux_coef_cfg.split("-")
        value = float(value)
        assert cfg in ["v", "c"] and value >= 0.0
        if cfg == "c":
            self.aux_constraint = value
            self.aux_coef_log = torch.zeros(1, requires_grad=True, device=self.device)
        else:
            self.aux_constraint = None
            self.aux_coef = value

    def _init_optims(self, lr: Dict[str, Any]) -> None:
        self.model_opt = torch.optim.Adam(
            [
                {"params": self.encoder.parameters()},
                {"params": self.model.parameters(), "lr": lr["model"]},
            ],
            lr=lr["encoder"],
        )
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr["actor"])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr["critic"])

        if self.aux in ["bisim_critic", "zp_critic", "op_critic"]:
            self.wass_critic_opt = torch.optim.RMSprop(
                self.wass_critic.parameters(), lr=lr["bisim_critic"]
            )

        if not self.disable_reward:
            self.reward_opt = torch.optim.Adam(
                utils.get_parameters(self.reward_list), lr=lr["reward"]
            )
        if self.aux_constraint is not None:
            self.coef_opt = torch.optim.Adam([self.aux_coef_log], lr=lr["model"])

    def get_coef(self) -> float:
        if self.aux_constraint is None:
            return self.aux_coef
        return self.aux_coef_log.exp().item()

    @torch.compile
    def _get_action_torch(
        self, state: torch.Tensor, step: int, eval: bool
    ) -> torch.Tensor:
        with torch.no_grad():
            std = utils.linear_schedule(
                self.expl_start, self.expl_end, self.expl_duration, step
            )
            z = self.encoder(state).sample()
            action_dist = self.actor(z, std)
            action = action_dist.sample(clip=None)

            if eval:
                action = action_dist.mean

        return action

    def get_action(
        self, state: npt.NDArray, step: int, eval: bool = False
    ) -> npt.NDArray:
        state_torch = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_torch = self._get_action_torch(state_torch, step, eval)
        return action_torch.cpu().numpy()[0]

    def get_representation(self, state: npt.NDArray) -> npt.NDArray:
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            z = self.encoder(state).sample()

        return z.cpu().numpy()

    @torch.compile
    def _get_lower_bound_torch(
        self, state_batch: torch.Tensor, action_batch: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            z_batch = self.encoder_target(state_batch).sample()
            z_seq, action_seq = self._rollout_evaluation(z_batch, action_batch, std=0.1)

            reward = self.reward(z_seq[:-1], action_seq[:-1])
            kl_reward = self.classifier.get_reward(
                z_seq[:-1], action_seq[:-1], z_seq[1:]
            )
            discount = self.gamma * torch.ones_like(reward)
            q_values_1, q_values_2 = self.critic(z_seq[-1], action_seq[-1])
            q_values = torch.min(q_values_1, q_values_2)

            returns = torch.cat(
                [reward + self.lambda_cost * kl_reward, q_values.unsqueeze(0)]
            )
            discount = torch.cat([torch.ones_like(discount[:1]), discount])
            discount = torch.cumprod(discount, 0)

            lower_bound = torch.sum(discount * returns, dim=0)
        return lower_bound

    def get_lower_bound(
        self, state_batch: torch.Tensor, action_batch: torch.Tensor
    ) -> npt.NDArray:
        lower_bound = self._get_lower_bound_torch(state_batch, action_batch)
        return lower_bound.cpu().numpy()

    @torch.compile
    def _rollout_evaluation(
        self, z_batch: torch.Tensor, action_batch: torch.Tensor, std: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_seq = [z_batch]
        action_seq = [action_batch]
        with torch.no_grad():
            for _ in range(self.seq_len):
                z_batch = self.model(z_batch, action_batch).sample()

                action_dist = self.actor(z_batch.detach(), std)
                action_batch = action_dist.mean

                z_seq.append(z_batch)
                action_seq.append(action_batch)

        z_seq = torch.stack(z_seq, dim=0)
        action_seq = torch.stack(action_seq, dim=0)
        return z_seq, action_seq

    def update(self, step: int) -> None:
        metrics = dict()
        std = utils.linear_schedule(
            self.expl_start, self.expl_end, self.expl_duration, step
        )

        if step % self.log_interval == 0:
            log = True
        else:
            log = False

        self.update_representation(std, log, metrics)
        self.update_rest(std, log, metrics)

        if step % self.target_update_interval == 0:
            utils.soft_update(self.encoder_target, self.encoder, self.tau)
            utils.soft_update(self.critic_target, self.critic, self.tau)

        if log:
            logger.record_step("env_steps", step)
            for k, v in metrics.items():
                logger.record_tabular(k, v)
            logger.dump_tabular()

    def update_representation(
        self, std: float, log: bool, metrics: Dict[str, Any]
    ) -> None:
        (
            state_seq,
            action_seq,
            reward_seq,
            next_state_seq,
            done_seq,
        ) = self.env_buffer.sample_seq(self.seq_len, self.batch_size)

        state_seq = torch.FloatTensor(state_seq).to(self.device)  # (T, B, D)
        next_state_seq = torch.FloatTensor(next_state_seq).to(self.device)
        action_seq = torch.FloatTensor(action_seq).to(self.device)
        reward_seq = torch.FloatTensor(reward_seq).to(self.device)  # (T, B)
        done_seq = torch.FloatTensor(done_seq).to(self.device)  # (T, B)

        wass_critic_loss = None

        if self.aux == "bisim_critic":
            wass_critic_loss_fn = self.bisim_wass_critic_loss
        elif self.aux == "zp_critic":
            wass_critic_loss_fn = self.zp_wass_critic_loss
        else:
            wass_critic_loss_fn = None

        if wass_critic_loss_fn:
            for _ in range(self.was_critic_train_steps):
                self.wass_critic_opt.zero_grad()
                wass_critic_loss = wass_critic_loss_fn(
                    state_seq, action_seq, next_state_seq
                )
                wass_critic_loss.backward()
                self.wass_critic_opt.step()

        alm_loss, aux_loss = self.alm_loss(
            state_seq, action_seq, next_state_seq, reward_seq, std, metrics
        )

        self.model_opt.zero_grad()
        alm_loss.backward()
        model_grad_norm = torch.nn.utils.clip_grad_norm_(
            utils.get_parameters(self.world_model_list), max_norm=self.max_grad_norm
        )
        self.model_opt.step()

        if log:
            metrics["alm_loss"] = alm_loss.item()
            metrics["model_grad_norm"] = model_grad_norm.item()

            metrics["aux_loss"] = aux_loss.mean().item()
            if wass_critic_loss is not None:
                metrics["wass_critic_loss"] = wass_critic_loss.mean().item()

        if self.aux_constraint is not None:
            self.coef_opt.zero_grad()
            coef_loss = self.aux_coef_log.exp() * (
                self.aux_constraint - aux_loss.mean().item()
            )
            coef_loss.backward()
            self.coef_opt.step()

            if log:
                metrics["coef"] = self.get_coef()

    def alm_loss(
        self,
        state_seq: torch.Tensor,
        action_seq: torch.Tensor,
        next_state_seq: torch.Tensor,
        reward_seq: torch.Tensor,
        std: float,
        metrics: Dict[str, Any],
        check_collapse: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        z_dist = self.encoder(state_seq[0])
        z_batch = z_dist.rsample()  # z (B, Z)

        # This is an expensive operation
        if check_collapse:
            self._check_collapse(z_batch.detach(), metrics)

        log = True
        aux_loss = None

        if self.disable_reward:
            if self.aux is not None:
                aux_loss, _ = self._aux_loss(
                    z_batch,
                    action_seq[0],
                    next_state_seq[0],
                    reward_seq[0],
                    log,
                    metrics,
                )  # (B, 1)
                alm_loss = self.get_coef() * aux_loss
                alm_loss = alm_loss.mean()
            else:
                alm_loss = 0.0
                aux_loss = 0.0
        else:
            alm_loss = 0.0
            for t in range(self.seq_len):
                if t > 0 and log:
                    log = False

                aux_loss, z_next_prior_batch = self._aux_loss(
                    z_batch,  # WHY DOESN"T THIS DEPEND ON state_seq[t]?
                    action_seq[t],
                    next_state_seq[t],
                    reward_seq[t],
                    log,
                    metrics,
                )
                reward_loss = self._alm_reward_loss(
                    z_batch, action_seq[t], log, metrics
                )
                alm_loss += self.get_coef() * aux_loss - reward_loss

                z_batch = z_next_prior_batch  # z' ~ p(z' | z, a)

            alm_loss = alm_loss.mean()

        # max_{phi} Q(phi(s), pi(phi(s)))
        if self.freeze_critic:
            actor_loss = self._actor_loss(
                z_batch, std, detach_qz=True, detach_action=False
            )
        else:  # original ALM
            actor_loss = self._actor_loss(
                z_batch, std, detach_qz=False, detach_action=True
            )

        alm_loss += actor_loss
        return alm_loss, aux_loss

    def _check_collapse(self, z_batch: torch.Tensor, metrics: Dict[str, Any]) -> None:
        rank3 = matrix_rank(z_batch, atol=1e-3, rtol=1e-3)
        rank2 = matrix_rank(z_batch, atol=1e-2, rtol=1e-2)
        rank1 = matrix_rank(z_batch, atol=1e-1, rtol=1e-1)
        condition = cond(z_batch)

        metrics["rank-3"] = rank3.item()
        metrics["rank-2"] = rank2.item()
        metrics["rank-1"] = rank1.item()
        metrics["cond"] = condition.item()

    @torch.compile()
    def _get_z_next_dist(self, next_state_batch: torch.Tensor) -> td.Distribution:
        if self.aux_optim == "ema":
            with torch.no_grad():
                return self.encoder_target(next_state_batch)  # p(z' | s')
        elif self.aux_optim == "detach":
            with torch.no_grad():
                return self.encoder(next_state_batch)  # p(z' | s')
        elif self.aux_optim == "online":
            return self.encoder(next_state_batch)  # p(z' | s')
        else:
            raise ValueError(self.aux_optim)

    @torch.compile
    def bisim_wass_critic_loss(
        self,
        state_seq: torch.Tensor,
        action_seq: torch.Tensor,
        next_state_seq: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            z_dist = self.encoder(state_seq[0])
            z_batch = z_dist.rsample()  # z (B, Z)
            action_batch = action_seq[0]
            next_state_batch = next_state_seq[0]
            z_next_dist = self._get_z_next_dist(next_state_batch)

            norm_action_batch = self.action_normalizer(action_batch)

            idxs_i = torch.randperm(self.batch_size)
            idxs_j = torch.arange(0, self.batch_size)

        critique_i = self.wass_critic(
            z_next_dist.mean[idxs_i],
            z_batch[idxs_i],
            norm_action_batch[idxs_i],
            z_batch[idxs_j],
            norm_action_batch[idxs_j],
        )
        critique_j = self.wass_critic(
            z_next_dist.mean[idxs_j],
            z_batch[idxs_i],
            norm_action_batch[idxs_i],
            z_batch[idxs_j],
            norm_action_batch[idxs_j],
        )

        bisim_critic_loss = -torch.mean(critique_i - critique_j)  # signed!
        return bisim_critic_loss

    @torch.compile
    def bisim_wass_encoder_loss(
        self,
        z_batch: torch.Tensor,
        next_state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        reward_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the components of the bisim loss. This is separated from aux_loss so that it can
        be `torch.compile`d without worrying about logging.
        """
        z_next_dist = self._get_z_next_dist(next_state_batch)  # p(z' | s')

        idxs_i = torch.randperm(self.batch_size)
        idxs_j = torch.arange(0, self.batch_size)

        z_dist = torch.norm(z_batch[idxs_i] - z_batch[idxs_j], dim=-1).view(-1, 1) / 2.0
        r_dist = self.reward_normalizer(
            torch.abs(reward_batch[idxs_i] - reward_batch[idxs_j]).view(-1, 1)
        )

        norm_action_batch = self.action_normalizer(action_batch)

        if "critic" in self.aux:
            critique_i = self.wass_critic(
                z_next_dist.mean[idxs_i],
                z_batch[idxs_i],
                norm_action_batch[idxs_i],  # TODO normalize action inputs?
                z_batch[idxs_j],
                norm_action_batch[idxs_j],
            )
            critique_j = self.wass_critic(
                z_next_dist.mean[idxs_j],
                z_batch[idxs_i],
                norm_action_batch[idxs_i],
                z_batch[idxs_j],
                norm_action_batch[idxs_j],
            )
            transition_dist = torch.abs(critique_i - critique_j).view(-1, 1) / 2.0
        else:
            transition_dist = (
                torch.sqrt(
                    (z_next_dist.mean[idxs_i] - z_next_dist.mean[idxs_j]).pow(2)
                    + (z_next_dist.stddev[idxs_i] - z_next_dist.stddev[idxs_j]).pow(2)
                )
                / 2.0
            )

        bisimilarity = (
            1.0 - self.bisim_gamma
        ) * r_dist + self.bisim_gamma * transition_dist
        bisim_loss = torch.square(z_dist - bisimilarity)

        return bisim_loss, z_dist, r_dist, transition_dist

    @torch.compile
    def zp_wass_critic_loss(
        self,
        state_seq: torch.Tensor,
        action_seq: torch.Tensor,
        next_state_seq: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            z_dist = self.encoder(state_seq[0])
            z_batch = z_dist.rsample()  # z (B, Z)
            action_batch = action_seq[0]
            next_state_batch = next_state_seq[0]
            z_next_dist = self._get_z_next_dist(next_state_batch)
            norm_action_batch = self.action_normalizer(action_batch)
        z_next_prior_dist = self.model(z_batch, action_batch)  # p_z(z' | z, a)

        critique_pred = self.wass_critic(
            z_next_prior_dist.mean,
            z_batch,
            norm_action_batch,
        )
        critique_true = self.wass_critic(
            z_next_dist.mean,
            z_batch,
            norm_action_batch,
        )

        return -torch.mean(critique_pred - critique_true)

    @torch.compile
    def zp_wass_encoder_loss(
        self,
        z_batch: torch.Tensor,
        next_state_batch: torch.Tensor,
        action_batch: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            z_next_dist = self._get_z_next_dist(next_state_batch)
            norm_action_batch = self.action_normalizer(action_batch)
        z_next_prior_dist = self.model(z_batch, action_batch)  # p_z(z' | z, a)

        critique_pred = self.wass_critic(
            z_next_prior_dist.mean,
            z_batch,
            norm_action_batch,
        )
        critique_true = self.wass_critic(
            z_next_dist.mean,
            z_batch,
            norm_action_batch,
        )

        return F.mse_loss(critique_pred, critique_true), z_next_prior_dist.sample()

    def _aux_loss(
        self,
        z_batch: torch.Tensor,
        action_batch: torch.Tensor,
        next_state_batch: torch.Tensor,
        reward_batch: torch.Tensor,
        log: bool,
        metrics: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if "op" in self.aux:
            next_state_pred = self.model(z_batch, action_batch)  # p_o(s' | z, a)

            if self.aux == "op-l2":
                distance = ((next_state_pred.rsample() - next_state_batch) ** 2).sum(
                    -1, keepdim=True
                )  # (B, 1)
            else:  # op-kl
                # fkl: negative log_prob
                distance = -next_state_pred.log_prob(next_state_batch).unsqueeze(
                    -1
                )  # (B, 1)
            if log:
                metrics[self.aux] = distance.mean().item()

            return distance, None

        if self.aux in ["bisim", "bisim_critic"]:
            z_next_prior_sample = None
            (
                distance,
                z_dist,
                r_dist,
                transition_dist,
            ) = self.bisim_wass_encoder_loss(
                z_batch,
                next_state_batch,
                action_batch,
                reward_batch,
            )
            if log:
                metrics["z_dist"] = z_dist.mean().item()
                metrics["r_dist"] = r_dist.mean().item()
                metrics["transition_dist"] = transition_dist.mean().item()
        elif self.aux == "zp_critic":
            distance, z_next_prior_sample = self.zp_wass_encoder_loss(
                z_batch, next_state_batch, action_batch
            )
        else:  # TODO simplify this. for now we do this to torch.compile(bisim_encoder_loss) nicely
            z_next_prior_dist = self.model(z_batch, action_batch)  # p_z(z' | z, a)
            z_next_dist = self._get_z_next_dist(next_state_batch)  # p(z' | s')

            if self.aux == "l2":
                distance = (
                    (z_next_dist.rsample() - z_next_prior_dist.rsample()) ** 2
                ).sum(
                    -1, keepdim=True
                )  # (B, 1)
                if log:
                    metrics["l2"] = distance.mean().item()

            else:  # fkl, rkl
                if self.aux == "fkl":
                    distance = td.kl_divergence(
                        z_next_dist, z_next_prior_dist
                    ).unsqueeze(
                        -1
                    )  # (B, 1)
                else:
                    distance = td.kl_divergence(
                        z_next_prior_dist, z_next_dist
                    ).unsqueeze(
                        -1
                    )  # (B, 1)

            if log:
                metrics[self.aux] = distance.mean().item()
                # metrics["prior_entropy"] = z_next_prior_dist.entropy().mean().item()
                # metrics["posterior_entropy"] = z_next_dist.entropy().mean().item()

            z_next_prior_sample = z_next_prior_dist.rsample()

        return distance, z_next_prior_sample

    def _alm_reward_loss(
        self,
        z_batch: torch.Tensor,
        action_batch: torch.Tensor,
        log: bool,
        metrics: Dict[str, Any],
    ) -> torch.Tensor:
        with utils.FreezeParameters(self.reward_list):
            reward = self.reward(z_batch, action_batch)  # r_z(z, a)

        if log:
            metrics["alm_reward_batch"] = reward.mean().item()

        return reward

    def update_rest(self, std, log: bool, metrics: Dict[str, Any]) -> None:
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = self.env_buffer.sample(self.batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        discount_batch = self.gamma * (1 - done_batch)

        if self.online_encoder_actorcritic:
            z_dist = self.encoder(state_batch)

        with torch.no_grad():
            if not self.online_encoder_actorcritic:
                z_dist = self.encoder_target(state_batch)
            z_next_prior_dist = self.model(z_dist.sample(), action_batch)
            z_next_dist = self.encoder_target(next_state_batch)

        if not self.disable_reward:
            # update reward and classifier
            self.update_reward(
                z_dist.sample(),
                action_batch,
                reward_batch,
                z_next_dist.sample(),
                z_next_prior_dist.sample(),
                log,
                metrics,
            )

        # update critic
        self.update_critic(
            z_dist.rsample(),  # encoder may update from this
            action_batch,
            reward_batch,
            z_next_dist.sample(),
            discount_batch,
            std,
            log,
            metrics,
        )

        # update actor
        self.update_actor(z_dist.sample(), std, log, metrics)

    def update_reward(
        self,
        z_batch: torch.Tensor,
        action_batch: torch.Tensor,
        reward_batch: torch.Tensor,
        z_next_batch: torch.Tensor,
        z_next_prior_batch: torch.Tensor,
        log: bool,
        metrics: Dict[str, Any],
    ) -> None:
        reward_loss = self._extrinsic_reward_loss(
            z_batch, action_batch, reward_batch.unsqueeze(-1), log, metrics
        )
        classifier_loss = self._intrinsic_reward_loss(
            z_batch, action_batch, z_next_batch, z_next_prior_batch, log, metrics
        )
        self.reward_opt.zero_grad()
        (reward_loss + classifier_loss).backward()
        reward_grad_norm = torch.nn.utils.clip_grad_norm_(
            utils.get_parameters(self.reward_list), max_norm=self.max_grad_norm
        )
        self.reward_opt.step()

        if log:
            metrics["reward_grad_norm"] = reward_grad_norm.mean().item()

    def _extrinsic_reward_loss(
        self,
        z_batch: torch.Tensor,
        action_batch: torch.Tensor,
        reward_batch: torch.Tensor,
        log: bool,
        metrics: Dict[str, Any],
    ) -> torch.Tensor:
        reward_pred = self.reward(z_batch, action_batch)
        reward_loss = F.mse_loss(reward_pred, reward_batch)

        if log:
            metrics["reward_loss"] = reward_loss.item()
            metrics["min_true_reward"] = torch.min(reward_batch).item()
            metrics["max_true_reward"] = torch.max(reward_batch).item()
            metrics["mean_true_reward"] = torch.mean(reward_batch).item()

        return reward_loss

    def _intrinsic_reward_loss(
        self,
        z: torch.Tensor,
        action_batch: torch.Tensor,
        z_next: torch.Tensor,
        z_next_prior: torch.Tensor,
        log: bool,
        metrics: Dict[str, Any],
    ) -> torch.Tensor:
        ip_batch_shape = z.shape[0]
        false_batch_idx = np.random.choice(
            ip_batch_shape, ip_batch_shape // 2, replace=False
        )
        z_next_target = z_next
        z_next_target[false_batch_idx] = z_next_prior[false_batch_idx]

        labels = torch.ones(ip_batch_shape, dtype=torch.long, device=self.device)
        labels[false_batch_idx] = 0.0

        logits = self.classifier(z, action_batch, z_next_target)
        classifier_loss = nn.CrossEntropyLoss()(logits, labels)

        if log:
            metrics["classifier_loss"] = classifier_loss.item()

        return classifier_loss

    def update_critic(
        self,
        z_batch: torch.Tensor,
        action_batch: torch.Tensor,
        reward_batch: torch.Tensor,
        z_next_batch: torch.Tensor,
        discount_batch: torch.Tensor,
        std: float,
        log: bool,
        metrics: Dict[str, Any],
    ) -> None:
        critic_loss = self._critic_loss(
            z_batch, action_batch, reward_batch, z_next_batch, discount_batch, std
        )

        if self.online_encoder_actorcritic:
            self.model_opt.zero_grad()
        self.critic_opt.zero_grad()

        critic_loss.backward()

        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            utils.get_parameters(self.critic_list), max_norm=self.max_grad_norm
        )
        self.critic_opt.step()

        if self.online_encoder_actorcritic:
            model_grad_norm = torch.nn.utils.clip_grad_norm_(
                utils.get_parameters(self.world_model_list), max_norm=self.max_grad_norm
            )
            self.model_opt.step()

        if log:
            metrics["critic_loss"] = critic_loss.item()
            metrics["critic_grad_norm"] = critic_grad_norm.mean().item()

    def _critic_loss(
        self,
        z_batch: torch.Tensor,
        action_batch: torch.Tensor,
        reward_batch: torch.Tensor,
        z_next_batch: torch.Tensor,
        discount_batch: torch.Tensor,
        std: float,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_action_dist = self.actor(z_next_batch, std)
            next_action_batch = next_action_dist.sample(clip=self.stddev_clip)

            target_Q1, target_Q2 = self.critic_target(z_next_batch, next_action_batch)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward_batch.unsqueeze(-1) + discount_batch.unsqueeze(-1) * (
                target_V
            )

        Q1, Q2 = self.critic(z_batch, action_batch)
        critic_loss = (F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)) / 2

        return critic_loss

    def update_actor(
        self,
        z_batch: torch.Tensor,
        std: torch.Tensor,
        log: bool,
        metrics: Dict[str, Any],
    ) -> None:
        if self.disable_svg:
            actor_loss = self._actor_loss(
                z_batch, std, detach_qz=True, detach_action=False
            )
        else:
            actor_loss = self._lambda_svg_loss(z_batch, std, log, metrics)

        self.actor_opt.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            utils.get_parameters(self.actor_list), max_norm=self.max_grad_norm
        )
        self.actor_opt.step()

        if log:
            metrics["actor_grad_norm"] = actor_grad_norm.mean().item()

    @torch.compile
    def _actor_loss(
        self, z_batch: torch.Tensor, std: float, detach_qz: bool, detach_action: bool
    ) -> torch.Tensor:
        with utils.FreezeParameters(self.critic_list):
            action_dist = self.actor(z_batch, std)
            action_batch = action_dist.sample(clip=self.stddev_clip)
            Q1, Q2 = self.critic(
                z_batch.detach() if detach_qz else z_batch,
                action_batch.detach() if detach_action else action_batch,
            )
            Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()
        return actor_loss

    @torch.compile
    def _lambda_svg_loss(
        self, z_batch: torch.Tensor, std: float, log: bool, metrics: Dict[str, Any]
    ) -> torch.Tensor:
        actor_loss = 0
        z_seq, action_seq = self._rollout_imagination(z_batch, std)

        with utils.FreezeParameters(
            [self.model, self.reward, self.classifier, self.critic]
        ):
            reward = self.reward(z_seq[:-1], action_seq[:-1])
            kl_reward = self.classifier.get_reward(
                z_seq[:-1], action_seq[:-1], z_seq[1:].detach()
            )
            discount = self.gamma * torch.ones_like(reward)
            q_values_1, q_values_2 = self.critic(z_seq, action_seq.detach())
            q_values = torch.min(q_values_1, q_values_2)

            returns = lambda_returns(
                reward + self.lambda_cost * kl_reward,
                discount,
                q_values[:-1],
                q_values[-1],
                self.seq_len,
            )
            discount = torch.cat([torch.ones_like(discount[:1]), discount])
            discount = torch.cumprod(discount[:-1], 0)
            actor_loss = -torch.mean(discount * returns)

        if log:
            metrics["min_imag_reward"] = torch.min(reward).item()
            metrics["max_imag_reward"] = torch.max(reward).item()
            metrics["mean_imag_reward"] = torch.mean(reward).item()
            metrics["min_imag_kl_reward"] = torch.min(kl_reward).item()
            metrics["max_imag_kl_reward"] = torch.max(kl_reward).item()
            metrics["mean_imag_kl_reward"] = torch.mean(kl_reward).item()
            metrics["actor_loss"] = actor_loss.item()
            metrics["lambda_cost"] = self.lambda_cost
            metrics["min_imag_value"] = torch.min(q_values).item()
            metrics["max_imag_value"] = torch.max(q_values).item()
            metrics["mean_imag_value"] = torch.mean(q_values).item()
            metrics["action_std"] = std

        return actor_loss

    @torch.compile
    def _rollout_imagination(
        self, z_batch: torch.Tensor, std: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_seq = [z_batch]
        action_seq = []
        with utils.FreezeParameters([self.model]):
            for t in range(self.seq_len):
                action_dist = self.actor(z_batch.detach(), std)
                action_batch = action_dist.sample(self.stddev_clip)
                z_batch = self.model(z_batch, action_batch).rsample()
                action_seq.append(action_batch)
                z_seq.append(z_batch)

            action_dist = self.actor(z_batch.detach(), std)
            action_batch = action_dist.sample(self.stddev_clip)
            action_seq.append(action_batch)

        z_seq = torch.stack(z_seq, dim=0)
        action_seq = torch.stack(action_seq, dim=0)
        return z_seq, action_seq

    def get_save_dict(self) -> Dict[str, Any]:
        return {
            "encoder": self.encoder.state_dict(),
            "encoder_target": self.encoder_target.state_dict(),
            "model": self.model.state_dict(),
            "reward": self.reward.state_dict(),
            "classifier": self.classifier.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor": self.actor.state_dict(),
        }

    def load_save_dict(self, saved_dict: Dict[str, Any]) -> None:
        self.encoder.load_state_dict(saved_dict["encoder"])
        self.encoder_target.load_state_dict(saved_dict["encoder_target"])
        self.model.load_state_dict(saved_dict["model"])
        self.reward.load_state_dict(saved_dict["reward"])
        self.classifier.load_state_dict(saved_dict["classifier"])
        self.critic.load_state_dict(saved_dict["critic"])
        self.critic_target.load_state_dict(saved_dict["critic_target"])
        self.actor.load_state_dict(saved_dict["actor"])


@torch.compile
def lambda_returns(
    reward: torch.Tensor,
    discount: torch.Tensor,
    q_values: torch.Tensor,
    bootstrap: torch.Tensor,
    horizon: int,
    lambda_: float = 0.95,
) -> torch.Tensor:
    next_values = torch.cat([q_values[1:], bootstrap[None]], 0)
    inputs = reward + discount * next_values * (1 - lambda_)
    last = bootstrap
    returns = []
    for t in reversed(range(horizon)):
        inp, disc = inputs[t], discount[t]
        last = inp + disc * lambda_ * last
        returns.append(last)

    returns = torch.stack(list(reversed(returns)), dim=0)
    return returns
    returns = torch.stack(list(reversed(returns)), dim=0)
    return returns
