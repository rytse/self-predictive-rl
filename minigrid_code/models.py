import torch
import torch.nn as nn
import torch.distributions as td
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


class DetLatentModel(nn.Module):
    """
    psi in AIS, P_theta in RL.
    Deterministic latent transition models.
    E[o' | z, a] or E[z' | z, a], depends on num_obs
    """

    def __init__(self, num_obs: int, num_actions: int, AIS_state_size: int):
        super(DetLatentModel, self).__init__()
        input_ndims = AIS_state_size + num_actions
        self.fc1_d = nn.Linear(input_ndims, AIS_state_size // 2)
        self.fc2_d = nn.Linear(AIS_state_size // 2, num_obs)

        self.apply(weights_init_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_d = F.elu(self.fc1_d(x))
        obs = self.fc2_d(x_d)
        return obs


class StoLatentModel(nn.Module):
    def __init__(self, latent_dim: int, num_actions: int, hidden_dim: int):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim + num_actions, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

        self.std_min = 0.1
        self.std_max = 10.0
        self.apply(weights_init_)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> td.Independent:
        catted = torch.cat([z, a], dim=-1)
        zp = self.encoder(catted)
        mean, std = zp.chunk(2, dim=-1)
        mean = 30.0 * torch.tanh(mean / 30.0)
        std = self.std_max - F.softplus(self.std_max - std)
        std = self.std_min + F.softplus(std - self.std_min)
        return td.independent.Independent(td.Normal(mean, std), 1)


class GenLatentModel(nn.Module):
    """
    Generative latent transition model.
    Maps (z, a, noise) -> z' samples
    Output shape: [batch_size x latent_dim x n_samples]
    """

    def __init__(
        self, latent_dim: int, num_actions: int, hidden_dim: int, noise_dim: int
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.noise_dim = noise_dim

        self.generator = nn.Sequential(
            nn.Linear(latent_dim + num_actions + noise_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.apply(weights_init_)

    def forward(
        self, z: torch.Tensor, a: torch.Tensor, n_samples: int = 1
    ) -> torch.Tensor:
        """
        Args:
            z: Current latent state [batch_size x latent_dim]
            a: Action [batch_size x action_dim]
            n_samples: Number of samples to generate per (z,a) pair
        Returns:
            Samples of next latent state [batch_size x latent_dim x n_samples]
        """
        batch_size = z.shape[0]

        # Expand inputs to include samples dimension
        z = z.unsqueeze(-1).expand(-1, -1, n_samples)  # [B x D x S]
        a = a.unsqueeze(-1).expand(-1, -1, n_samples)  # [B x A x S]

        # Sample noise
        noise = torch.randn(
            batch_size, self.noise_dim, n_samples, device=z.device
        )  # [B x N x S]

        # Concatenate inputs
        x = torch.cat([z, a, noise], dim=1)  # [B x (D+A+N) x S]

        # Reshape to [B*S x (D+A+N)] for Linear layers
        x = x.permute(0, 2, 1)  # [B x S x (D+A+N)]
        x = x.reshape(-1, x.shape[-1])  # [B*S x (D+A+N)]

        # Generate samples
        z_next = self.generator(x)  # [B*S x D]

        # Reshape back to [B x D x S]
        z_next = z_next.reshape(batch_size, n_samples, -1)  # [B x S x D]
        z_next = z_next.permute(0, 2, 1)  # [B x D x S]

        return z_next

    def sample(
        self, z: torch.Tensor, a: torch.Tensor, n_samples: int = 1
    ) -> torch.Tensor:
        """Convenience method for generating samples"""
        with torch.no_grad():
            return self.forward(z, a, n_samples)


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
