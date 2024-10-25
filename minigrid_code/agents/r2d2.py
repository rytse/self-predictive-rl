import random
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Type
import gym
import numpy as np
import numpy.typing as npt
import torch
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
import torch.nn as nn
import logger

from r2d2replaybuffer import r2d2_ReplayMemory
from models import (
    AISModel,
    LatentModel,
    pack_padded_sequence,
    soft_update,
    hard_update,
    SeqEncoder,
    QNetwork_discrete,
    convert_int_to_onehot,
)


class R2D2(ABC):
    """
    Recurrent Replay Distributed DQN agent
    """

    obs_dim: int
    act_dim: int
    AIS_state_size: int
    hidden_size: int
    model_dim: int

    optim: Optimizer
    model_type: Optional[Type[nn.Module]]
    model: Optional[nn.Module]

    def __init__(self, env: gym.Env, args: Dict[str, Any]):
        self.env = env
        self.args = args

        self.device = torch.device(args["device"])

        self.obs_dim = np.prod(env.observation_space["image"].shape)  # flatten
        self.act_dim = env.action_space.n  # n discrete actions. will be 1-hot encoded
        self.AIS_state_size = args["AIS_state_size"]  # TODO rename z_dim
        self.hidden_size = args["hidden_size"]

        self.model_dim = 0

        self.gamma = args["gamma"]
        self.tau = args["tau"]
        self.target_update_interval = args["target_update_interval"]

        self.setup_networks()
        self.setup_optimizers()

        logger.log(self.encoder, self.model, self.critic)

        self.aux_optim = args["aux_optim"]
        self.aux_coef = args["aux_coef"]
        assert self.aux_optim in ["None", "ema", "detach", "online"]
        assert self.aux_coef >= 0.0

        self.update_to_q = 0
        self.eps_greedy_parameters = {
            "EPS_START": args["EPS_start"],
            "EPS_END": args["EPS_end"],
            "EPS_DECAY": args["EPS_decay"],
        }
        self.env_steps = 0

        self.get_initial_hidden = lambda: self.encoder.get_initial_hidden(
            1, self.device
        )

    def setup_networks(self) -> None:
        self.encoder = SeqEncoder(self.obs_dim, self.act_dim, self.AIS_state_size).to(
            self.device
        )
        self.encoder_target = SeqEncoder(
            self.obs_dim, self.act_dim, self.AIS_state_size
        ).to(self.device)
        hard_update(self.encoder_target, self.encoder)

        self.critic = torch.compile(
            QNetwork_discrete(self.AIS_state_size, self.act_dim, self.hidden_size).to(
                device=self.device
            ),
            mode="default",
        )
        self.critic_target = torch.compile(
            QNetwork_discrete(self.AIS_state_size, self.act_dim, self.hidden_size).to(
                self.device
            )
        )
        hard_update(self.critic_target, self.critic)

    @abstractmethod
    def setup_optimizers(self) -> None:
        pass

    # @torch.compile
    @torch.no_grad()
    def select_action(
        self,
        state_np: npt.NDArray,
        action_int: int,
        reward_int: int,
        hidden_p: Tuple[torch.Tensor, torch.Tensor],
        EPS_up: bool,
        evaluate: bool,
    ) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        action = convert_int_to_onehot(action_int, self.act_dim)
        reward = torch.Tensor([reward_int])
        state = torch.Tensor(state_np)
        rho_input = torch.cat((state, action, reward)).reshape(1, 1, -1).to(self.device)

        ais_z, hidden_p = self.encoder(
            rho_input,
            batch_size=1,
            hidden=hidden_p,
            device=self.device,
            batch_lengths=[],
            pack_sequence=False,
        )

        if evaluate is False and EPS_up:
            self.env_steps += 1

        if self.args["EPS_decay_type"] == "exponential":
            eps_threshold = self.eps_greedy_parameters["EPS_END"] + (
                self.eps_greedy_parameters["EPS_START"]
                - self.eps_greedy_parameters["EPS_END"]
            ) * math.exp(
                -1.0 * self.env_steps / self.eps_greedy_parameters["EPS_DECAY"]
            )
        elif self.args["EPS_decay_type"] == "linear":
            eps_threshold = self.eps_greedy_parameters["EPS_START"] + (
                self.eps_greedy_parameters["EPS_END"]
                - self.eps_greedy_parameters["EPS_START"]
            ) * (self.env_steps / self.eps_greedy_parameters["EPS_DECAY"])
            eps_threshold = max(eps_threshold, self.eps_greedy_parameters["EPS_END"])
        else:
            raise ValueError("Invalid EPS_decay_type")

        sample = random.random()
        if (sample < eps_threshold and evaluate is False) or (
            sample < self.args["test_epsilon"] and evaluate is True
        ):
            return random.randrange(self.act_dim), hidden_p

        qf = self.critic(ais_z)[0, 0]
        greedy_action = torch.argmax(qf).item()

        return greedy_action, hidden_p

    def update_parameters(
        self, memory, batch_size: int, updates: int
    ) -> Dict[str, Any]:
        metrics = {}
        for _ in range(updates):
            self.update_to_q += 1
            metrics = self.single_update(memory, batch_size)

            if self.update_to_q % self.target_update_interval == 0:
                # We change the hard update to soft update
                soft_update(self.encoder_target, self.encoder, self.tau)
                soft_update(self.critic_target, self.critic, self.tau)

        return metrics

    def single_update(
        self, memory: r2d2_ReplayMemory, batch_size: int
    ) -> Dict[str, Any]:
        metrics = {}
        losses = 0.0

        # 1. Sample a batch of data in numpy arrays
        (
            batch_burn_in_hist,  # (B, H, O+A+1)
            batch_learn_hist,  # (B, L+N, O+A+1)
            batch_rewards,  # (B, L)
            batch_learn_len,  # (B,) in [1, L]
            batch_forward_idx,  # (B, L) in [N, L+N-1] and {0, <N}
            batch_final_flag,  # (B, L) in {0,1}
            batch_current_act,  # (B, L)
            batch_hidden,  # (B, Z), (B, Z)
            batch_burn_in_len,  # (B,) in [0, H]
            batch_learn_forward_len,  # (B,) in [1, L+N]
            batch_next_obs,  # (B, L, O)
            batch_model_target_reward,  # (B, L)
            batch_model_final_flag,  # (B, L) in {0,1}
            batch_gammas,  # (B, L) in [0, 1]
        ) = memory.sample(batch_size)

        ## 2. Burn-in to get the initial hidden states for learning
        batch_hidden = (
            torch.from_numpy(batch_hidden[0]).view(1, batch_size, -1).to(self.device),
            torch.from_numpy(batch_hidden[1]).view(1, batch_size, -1).to(self.device),
        )
        batch_burn_in_hist = torch.from_numpy(batch_burn_in_hist).to(self.device)
        zero_idx = np.where(batch_burn_in_len == 0)[0]
        batch_burn_in_len[zero_idx] = 1  # temporarily change 0 to 1

        with torch.no_grad():
            _, hidden_burn_in = self.encoder(
                batch_burn_in_hist,
                batch_size,
                batch_hidden,
                self.device,
                list(batch_burn_in_len),
            )  # (1, B, Z), (1, B, Z)
            hidden_burn_in[0][:, zero_idx, :] = batch_hidden[0][:, zero_idx, :]
            hidden_burn_in[1][:, zero_idx, :] = batch_hidden[1][:, zero_idx, :]

            _, target_hidden_burn_in = self.encoder_target(
                batch_burn_in_hist,
                batch_size,
                batch_hidden,
                self.device,
                list(batch_burn_in_len),
            )
            target_hidden_burn_in[0][:, zero_idx, :] = batch_hidden[0][:, zero_idx, :]
            target_hidden_burn_in[1][:, zero_idx, :] = batch_hidden[1][:, zero_idx, :]

        # 3. Forward RNN for a length
        batch_learn_hist = torch.from_numpy(batch_learn_hist).to(self.device)
        ais_z, _ = self.encoder(
            batch_learn_hist,
            batch_size,
            hidden_burn_in,
            self.device,
            list(batch_learn_forward_len),
        )
        unpacked_ais_z, _ = pad_packed_sequence(ais_z, batch_first=True)  # (B,L+N,Z)
        with torch.no_grad():
            target_ais_z, _ = self.encoder_target(
                batch_learn_hist,
                batch_size,
                target_hidden_burn_in,
                self.device,
                list(batch_learn_forward_len),
            )
            target_unpacked_ais_z, _ = pad_packed_sequence(
                target_ais_z, batch_first=True
            )  # (B, L+N, Z)

        q_z = pack_padded_sequence(
            unpacked_ais_z,
            torch.Tensor(batch_learn_len),  # we only use first L hidden states
            batch_first=True,
            enforce_sorted=False,
        )

        qf = self.eval_critic(q_z)

        # if self.debug:
        #    self.report_rank(q_z.data.detach(), metrics)

        batch_current_act = torch.from_numpy(batch_current_act).to(self.device)
        packed_current_act = pack_padded_sequence(
            batch_current_act,
            torch.Tensor(batch_learn_len),
            batch_first=True,
            enforce_sorted=False,
        )
        # get Q(h[t], a[t])
        qf = qf.gather(1, packed_current_act.data.view(-1, 1).long())  # (*, 1)

        # 4. (optional) compute auxiliary loss
        batch_final_flag = torch.from_numpy(batch_final_flag).to(self.device)
        packed_final = pack_padded_sequence(
            batch_final_flag,
            torch.Tensor(batch_learn_len),
            batch_first=True,
            enforce_sorted=False,
        )

        # TODO Only necessary for some, maybe we take this out? Does the compiler know?
        q_next_z = pack_padded_sequence(
            unpacked_ais_z[:, 1:],  # shift one step
            torch.Tensor(batch_learn_len),  # we only use first L hidden states
            batch_first=True,
            enforce_sorted=False,
        )
        q_next_z_target = pack_padded_sequence(
            target_unpacked_ais_z[:, 1:],  # shift one step
            torch.Tensor(batch_learn_len),  # we only use first L hidden states
            batch_first=True,
            enforce_sorted=False,
        )
        next_obs = torch.from_numpy(batch_next_obs).to(self.device)
        next_obs_packed = pack_padded_sequence(
            next_obs,
            torch.Tensor(batch_learn_len),
            batch_first=True,
            enforce_sorted=False,
        )  # o'
        next_rew = (
            torch.from_numpy(batch_model_target_reward).to(self.device).unsqueeze(-1)
        )  # (B, L, 1)
        next_rew_packed = pack_padded_sequence(
            next_rew,
            torch.Tensor(batch_learn_len),
            batch_first=True,
            enforce_sorted=False,
        )  # r
        batch_model_final_flag = torch.from_numpy(batch_model_final_flag).to(
            self.device
        )
        packed_model_final = pack_padded_sequence(
            batch_model_final_flag,
            torch.Tensor(batch_learn_len),
            batch_first=True,
            enforce_sorted=False,
        )

        aux_loss = self.eval_aux_loss(
            metrics=metrics,
            batch_z=q_z.data,
            batch_act=packed_current_act.data,
            batch_next_z=q_next_z.data,
            batch_next_z_target=q_next_z_target.data,
            batch_next_obs=next_obs_packed.data,
            batch_rew=next_rew_packed.data,
            batch_final_flag=packed_final.data,
            batch_model_final_flag=packed_model_final.data,
        )
        losses += self.aux_coef * aux_loss

        # 5. Compute Target for Double Q-learning
        with torch.no_grad():
            batch_forward_idx = torch.from_numpy(batch_forward_idx).to(self.device)

            ais_z_for_target_action = unpacked_ais_z.gather(
                1,
                batch_forward_idx.view(batch_size, -1, 1)
                .expand(-1, -1, self.AIS_state_size)
                .long(),
            )  # z
            packed_ais_z_for_target_action = pack_padded_sequence(
                ais_z_for_target_action,
                torch.Tensor(batch_learn_len),
                batch_first=True,
                enforce_sorted=False,
            )
            max_idx = self.critic(packed_ais_z_for_target_action.data).max(1)[
                1
            ]  # argmax_a Q(z, a)

            qf_target = self.eval_qf_target(
                packed_ais_z_for_target_action,
                target_unpacked_ais_z,
                batch_forward_idx,
                batch_learn_len,
                batch_size,
            )

            qf_target = qf_target.gather(1, max_idx.view(-1, 1).long())  # (*, 1)

            batch_rewards = torch.from_numpy(batch_rewards).to(self.device)
            batch_gammas = torch.from_numpy(batch_gammas).to(self.device)

            packed_reward = pack_padded_sequence(
                batch_rewards,
                torch.Tensor(batch_learn_len),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_gammas = pack_padded_sequence(
                batch_gammas,
                torch.Tensor(batch_learn_len),
                batch_first=True,
                enforce_sorted=False,
            )

            next_q_value = (
                packed_reward.data
                + packed_gammas.data * packed_final.data * qf_target.view(-1)
            )

        # 6. Sum Q-learning loss and Auxiliary loss
        if self.args["TD_loss"] == "mse":
            qf_loss = F.mse_loss(qf.view(-1), next_q_value, reduce=False)
        elif self.args["TD_loss"] == "smooth_l1":
            qf_loss = F.smooth_l1_loss(qf.view(-1), next_q_value, reduce=False)
        else:
            raise ValueError("Invalid value for TD_loss")

        qf_loss = qf_loss.mean()
        # metrics["critic_loss"] = qf_loss.item()
        losses += qf_loss

        self.step_optimizers(losses)

        return metrics

    @abstractmethod
    def eval_critic(self, q_z: PackedSequence) -> torch.Tensor:
        """
        Evaluate the critic network for the total R2D2 loss. This might detach q_z or not in
        different implementations.
        """
        pass

    @abstractmethod
    def eval_aux_loss(
        self,
        batch_z: torch.Tensor,
        batch_act: torch.Tensor,
        batch_next_z: torch.Tensor,
        batch_next_z_target: torch.Tensor,
        batch_next_obs: torch.Tensor,
        batch_rew: torch.Tensor,
        batch_final_flag: torch.Tensor,
        batch_model_final_flag: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate the auxiliary loss for the R2D2 agent. This is the self-predictive loss for the
        AIS model.
        """
        pass

    @abstractmethod
    def eval_qf_target(
        self,
        packed_ais_z_for_target_action: PackedSequence,
        target_unpacked_ais_z: torch.Tensor,
        batch_forward_idx: torch.Tensor,
        batch_learn_len: npt.NDArray,
        batch_size: int,
    ) -> torch.Tensor:
        pass

    def step_optimizers(self, losses: torch.Tensor) -> None:
        self.optim.zero_grad()
        losses.backward()
        self.optim.step()


class End2End(R2D2, ABC):
    """
    R2D2-based agent that learns the encoder end-to-end with the DQN critic.
    """

    def __init__(self, env, args):
        self.model_type = LatentModel
        super().__init__(env, args)

    def setup_optimizers(self):
        assert self.model is not None
        self.optim = Adam(
            list(self.encoder.parameters()) + list(self.critic.parameters()),
            lr=self.args["rl_lr"],
        )
        self.AIS_optim = Adam(
            self.model.parameters(),
            lr=self.args["aux_lr"],
        )

    def eval_critic(self, q_z: PackedSequence) -> torch.Tensor:
        return self.critic(q_z.data)

    def eval_qf_target(
        self,
        packed_ais_z_for_target_action: PackedSequence,
        target_unpacked_ais_z: torch.Tensor,
        batch_forward_idx: torch.Tensor,
        batch_learn_len: npt.NDArray,
        batch_size: int,
    ) -> torch.Tensor:
        ais_z_target = target_unpacked_ais_z.gather(
            1,
            batch_forward_idx.view(batch_size, -1, 1)
            .expand(-1, -1, self.AIS_state_size)
            .long(),  # (B, L, Z)
        )  # (B, L, Z) z^tar
        packed_target_ais = pack_padded_sequence(
            ais_z_target,
            torch.Tensor(batch_learn_len),
            batch_first=True,
            enforce_sorted=False,
        )
        return self.critic_target(packed_target_ais.data)  # Q^tar(z^tar)

    def step_optimizers(self, losses: torch.Tensor) -> None:
        self.optim.zero_grad()
        self.AIS_optim.zero_grad()
        losses.backward()
        self.optim.step()
        self.AIS_optim.step()


class Phased(R2D2, ABC):
    """
    R2D2-based agent that alternates between updating the encoder and the DQN critic.
    """

    model: Optional[nn.Module]

    def __init__(self, env, args):
        self.model_type = AISModel
        super().__init__(env, args)

    def setup_optimizers(self):
        self.optim = Adam(list(self.critic.parameters()), lr=self.args["rl_lr"])
        if self.model is not None:
            self.AIS_optim = Adam(
                list(self.encoder.parameters()) + list(self.model.parameters()),
                lr=self.args["aux_lr"],
            )

    def eval_critic(self, q_z: PackedSequence) -> torch.Tensor:
        return self.critic(q_z.data.detach())

    def eval_qf_target(
        self,
        packed_ais_z_for_target_action: PackedSequence,
        target_unpacked_ais_z: torch.Tensor,
        batch_forward_idx: torch.Tensor,
        batch_learn_len: npt.NDArray,
        batch_size: int,
    ) -> torch.Tensor:
        return self.critic_target(packed_ais_z_for_target_action.data)  # Q^tar(z)


class ModelFree(Phased):
    @torch.compile
    def eval_aux_loss(
        self,
        batch_z: torch.Tensor,
        batch_act: torch.Tensor,
        batch_next_z: torch.Tensor,
        batch_next_z_target: torch.Tensor,
        batch_next_obs: torch.Tensor,
        batch_rew: torch.Tensor,
        batch_final_flag: torch.Tensor,
        batch_model_final_flag: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros((1,))


def tensor_rank_multires(z: torch.Tensor) -> Tuple[float, float, float]:
    rank1 = torch.linalg.matrix_rank(z, atol=1e-1, rtol=1e-1)
    rank2 = torch.linalg.matrix_rank(z, atol=1e-2, rtol=1e-2)
    rank3 = torch.linalg.matrix_rank(z, atol=1e-3, rtol=1e-3)
    return rank1, rank2, rank3
