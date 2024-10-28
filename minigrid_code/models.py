import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from torch.nn.utils.parametrizations import spectral_norm


def normalize_packed_sequence(packed_seq: PackedSequence) -> PackedSequence:
    """
    Applies f(x) = x / (1 + ||x||) to a PackedSequence,
    where the norm is computed over the last dimension.

    Args:
        packed_seq (PackedSequence): Input packed sequence with data shape
            (sum of sequence lengths, hidden_dim)

    Returns:
        PackedSequence: Normalized packed sequence with same structure as input
            and normalized values in data tensor
    """
    # Get the packed data
    data: torch.Tensor = packed_seq.data

    # Calculate the norm over the last dimension
    norm: torch.Tensor = torch.norm(data, p=2, dim=-1, keepdim=True)

    # Apply the normalization formula
    normalized_data: torch.Tensor = data / (1 + norm)

    # Create new PackedSequence with normalized data
    return PackedSequence(
        data=normalized_data,
        batch_sizes=packed_seq.batch_sizes,
        sorted_indices=packed_seq.sorted_indices,
        unsorted_indices=packed_seq.unsorted_indices,
    )


class SeqEncoder(nn.Module):
    """
    rho in AIS, phi in RL literature.
    Deterministic model z = phi(h)
    """

    def __init__(self, num_obs, num_actions, AIS_state_size):
        super(SeqEncoder, self).__init__()
        input_ndims = (
            num_obs + num_actions + 1
        )  # including reward, but it is uninformative
        self.AIS_state_size = AIS_state_size
        self.fc1 = nn.Linear(input_ndims, AIS_state_size)
        self.fc2 = nn.Linear(AIS_state_size, AIS_state_size)
        self.lstm = nn.LSTM(AIS_state_size, AIS_state_size, batch_first=True)

        self.apply(weights_init_)

    def get_initial_hidden(self, batch_size, device):  # TODO:
        return (
            torch.zeros(1, batch_size, self.AIS_state_size).to(device),
            torch.zeros(1, batch_size, self.AIS_state_size).to(device),
        )

    def forward(
        self,
        x,
        batch_size,
        hidden,
        device,
        batch_lengths,
        pack_sequence=True,
    ):
        if hidden == None:
            hidden = self.get_initial_hidden(batch_size, device)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        if pack_sequence is True:
            x = pack_padded_sequence(
                x, batch_lengths, batch_first=True, enforce_sorted=False
            )
            # print('packed',x.data.shape)

        x, hidden = self.lstm(x, hidden)

        if isinstance(x, torch.Tensor):
            x_normed = x / (1 + torch.norm(x, p=2, dim=-1, keepdim=True))
        else:
            x_normed = normalize_packed_sequence(x)
        return x_normed, hidden


class LatentModel(nn.Module):
    """
    psi in AIS, P_theta in RL.
    Deterministic latent transition models.
    E[o' | z, a] or E[z' | z, a], depends on num_obs
    """

    def __init__(self, num_obs, num_actions, AIS_state_size):
        super(LatentModel, self).__init__()
        input_ndims = AIS_state_size + num_actions
        self.fc1_d = nn.Linear(input_ndims, AIS_state_size // 2)
        self.fc2_d = nn.Linear(AIS_state_size // 2, num_obs)

        self.apply(weights_init_)

    def forward(self, x):
        x_d = F.elu(self.fc1_d(x))
        obs = self.fc2_d(x_d)
        return obs


class AISModel(nn.Module):
    """
    psi in AIS, P_theta in RL.
    Deterministic transition and reward models.
    E[o' | z, a] or E[z' | z, a] AND E[r | z, a]
    """

    def __init__(self, num_obs, num_actions, AIS_state_size):
        super(AISModel, self).__init__()
        input_ndims = AIS_state_size + num_actions
        self.fc1_d = nn.Linear(input_ndims, AIS_state_size // 2)
        self.fc2_d = nn.Linear(AIS_state_size // 2, num_obs)
        self.fc1_r = nn.Linear(input_ndims, AIS_state_size // 2)
        self.fc2_r = nn.Linear(AIS_state_size // 2, 1)

        self.apply(weights_init_)

    def forward(self, x):
        x_d = F.elu(self.fc1_d(x))
        obs = self.fc2_d(x_d)
        x_r = F.elu(self.fc1_r(x))
        rew = self.fc2_r(x_r)
        return obs, rew


class QNetwork_discrete(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork_discrete, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        x1 = F.elu(self.linear1(state))
        x1 = F.elu(self.linear2(x1))
        x1 = self.linear3(x1)

        return x1


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super(MLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.apply(weights_init_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WassersteinModel(nn.Module):
    def __init__(
        self,
        z_dim: int,
        a_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.arg_net = nn.Sequential(
            spectral_norm(nn.Linear(z_dim, hidden_dim)),
            nn.ReLU(),
        )
        self.cond_net = nn.Sequential(
            nn.Linear(a_dim, hidden_dim),
            nn.ReLU(),
        )
        self.combine_net = nn.Sequential(
            spectral_norm(nn.Linear(2 * hidden_dim, hidden_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, z_dim)),
        )

    def forward(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        arg = self.arg_net(z)
        cond = self.cond_net(a)
        return self.combine_net(torch.cat([arg, cond], -1))


# TODO don't repeate code
class WassersteinCritic(nn.Module):
    def __init__(
        self,
        z_dim: int,
        a_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.arg_net = nn.Sequential(
            spectral_norm(nn.Linear(z_dim, hidden_dim)),
            nn.ReLU(),
        )
        self.cond_net = nn.Sequential(
            nn.Linear(z_dim + a_dim, hidden_dim),
            nn.ReLU(),
        )
        self.combine_net = nn.Sequential(
            spectral_norm(nn.Linear(2 * hidden_dim, hidden_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, 1)),
        )

    def forward(
        self,
        zk: torch.Tensor,
        zi: torch.Tensor,
        ai: torch.Tensor,
    ) -> torch.Tensor:
        arg = self.arg_net(zk)
        cond = self.cond_net(torch.cat([zi, ai], -1))
        return self.combine_net(torch.cat([arg, cond], -1))


class BisimWassersteinCritic(nn.Module):
    def __init__(
        self,
        z_dim: int,
        a_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.arg_net = nn.Sequential(
            spectral_norm(nn.Linear(z_dim, hidden_dim)),
            nn.ReLU(),
        )
        self.cond_net = nn.Sequential(
            nn.Linear(2 * (z_dim + a_dim), hidden_dim),
            nn.ReLU(),
        )
        self.combine_net = nn.Sequential(
            spectral_norm(nn.Linear(2 * hidden_dim, hidden_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, 1)),
        )

    def forward(
        self,
        zk: torch.Tensor,
        zi: torch.Tensor,
        ai: torch.Tensor,
        zj: torch.Tensor,
        aj: torch.Tensor,
    ) -> torch.Tensor:
        arg = self.arg_net(zk)
        cond = self.cond_net(torch.cat([zi, ai, zj, aj], -1))
        return self.combine_net(torch.cat([arg, cond], -1))


def convert_int_to_onehot(value, num_values):
    onehot = torch.zeros(num_values)
    if value >= 0:  # ignore negative index
        onehot[int(value)] = 1.0
    return onehot


def weights_init_(m, gain=1):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        torch.nn.init.constant_(m.bias, 0)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
