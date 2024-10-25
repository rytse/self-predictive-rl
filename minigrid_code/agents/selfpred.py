from abc import ABC

import torch
import torch.nn.functional as F

from agents.r2d2 import R2D2, End2End, Phased
from models import AISModel


class SelfPred(R2D2, ABC):
    """
    R2D2-based agent that learns its encoder using a self-predictive loss
    """

    def setup_networks(self):
        super().setup_networks()
        self.model = torch.compile(
            self.model_type(
                self.model_dim,
                self.act_dim,
                self.AIS_state_size,
            ).to(self.device),
            mode="default",
        )


class ZP(SelfPred, ABC):
    def setup_networks(self):
        self.model_dim = self.AIS_state_size
        super().setup_networks()


class OP(SelfPred, ABC):
    def setup_networks(self):
        self.model_dim = self.obs_dim
        super().setup_networks()


class ZPPhased(Phased, ZP):
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
        model_input = torch.cat(
            (batch_z, F.one_hot(batch_act.long(), self.act_dim).float()), -1
        ).to(
            self.device
        )  # (z, a)

        if self.aux_optim == "online":
            true_next_z = batch_next_z
        elif self.aux_optim == "detach":
            true_next_z = batch_next_z.detach()
        elif self.aux_optim == "ema":
            true_next_z = batch_next_z_target
        else:
            raise ValueError(self.aux_optim)

        predicted_next_z, predicted_rew = self.model(model_input)  # z', r

        z_squared_error = (predicted_next_z - true_next_z) ** 2
        rew_squared_error = (predicted_rew - batch_rew) ** 2
        z_loss = (
            z_squared_error.sum(-1) * batch_final_flag
        ).mean() / self.AIS_state_size  # normalized
        rew_loss = (rew_squared_error.sum(-1) * batch_model_final_flag).mean()

        # metrics["zp_loss"] = z_loss.item()
        # metrics["rp_loss"] = rew_loss.item()

        AIS_loss = self.aux_coef * z_loss + rew_loss

        self.AIS_optim.zero_grad()
        AIS_loss.backward()
        self.AIS_optim.step()

        return torch.zeros_like(AIS_loss)


class ZPEnd2End(End2End, ZP):
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
        assert self.model is not None

        model_input = torch.cat(
            (batch_z, F.one_hot(batch_act.long(), self.act_dim).float()), -1
        ).to(
            self.device
        )  # (z, a)
        predicted_next_z = self.model(model_input)  # z'

        if self.aux_optim == "online":
            true_next_z = batch_next_z
        elif self.aux_optim == "detach":
            true_next_z = batch_next_z.detach()
        elif self.aux_optim == "ema":
            true_next_z = batch_next_z_target
        else:
            raise ValueError(self.aux_optim)

        squared_error = (predicted_next_z - true_next_z) ** 2
        model_loss = squared_error.sum(-1) * batch_final_flag

        model_loss = model_loss.mean() / self.AIS_state_size  # normalized

        # metrics["zp_loss"] = model_loss.item()

        return model_loss


class OPPhased(Phased, OP):
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
        model_input = torch.cat(
            (batch_z, F.one_hot(batch_act.long(), self.act_dim).float()), -1
        ).to(
            self.device
        )  # (z, a)

        predicted_obs, predicted_rew = self.model(model_input)
        obs_squared_error = (predicted_obs - batch_next_obs) ** 2
        rew_squared_error = (predicted_rew - batch_rew) ** 2
        obs_loss = (
            obs_squared_error.sum(-1) * batch_final_flag
        ).mean() / self.obs_dim  # normalized
        rew_loss = (rew_squared_error.sum(-1) * batch_model_final_flag).mean()

        # metrics["op_loss"] = obs_loss.item()
        # metrics["rp_loss"] = rew_loss.item()

        AIS_loss = self.aux_coef * obs_loss + rew_loss

        self.AIS_optim.zero_grad()
        AIS_loss.backward()
        self.AIS_optim.step()

        return torch.zeros_like(AIS_loss)


class OPEnd2End(End2End, OP):
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
        model_input = torch.cat(
            (batch_z, F.one_hot(batch_act.long(), self.act_dim).float()), -1
        ).to(
            self.device
        )  # (z, a)

        predicted_obs = self.model(model_input)
        squared_error = (predicted_obs - batch_next_obs) ** 2
        model_loss = squared_error.sum(-1) * batch_final_flag

        model_loss = model_loss.mean() / self.obs_dim  # normalized

        # metrics["op_loss"] = model_loss.item()

        return model_loss
