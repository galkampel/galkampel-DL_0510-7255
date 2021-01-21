import ase
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.nn import radius_graph, MessagePassing
from model.schnet import ShiftedSoftplus

try:
    import schnetpack as spk
except ImportError:
    spk = None

qm9_target_dict = {
    0: 'dipole_moment',
    1: 'isotropic_polarizability',
    2: 'homo',
    3: 'lumo',
    4: 'gap',
    5: 'electronic_spatial_extent',
    6: 'zpve',
    7: 'energy_U0',
    8: 'energy_U',
    9: 'enthalpy_H',
    10: 'free_energy',
    11: 'heat_capacity',
}


class NMPEdge(torch.nn.Module):
    r"""
    `"NMPEdge: Neural Message Passing with Edge Updates for Predicting Properties
     of Molecules and Materials" <https://arxiv.org/abs/1806.03146>`

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_filters (int, optional): The number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): The number of interaction blocks.
            (default: :obj:`3`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`150`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`15.0`)
        readout (string, optional): Whether to apply :obj:`"add"` or
            :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
        hypernet_update (bool, optionl): If true use hyperNetwork for the
            state-transition function (https://arxiv.org/pdf/2002.00240.pdf)
        dipole (bool, optional): If set to :obj:`True`, will use the magnitude
            of the dipole moment to make the final prediction, *e.g.*, for
            target 0 of :class:`torch_geometric.datasets.QM9`.
            (default: :obj:`False`)
        mean (float, optional): The mean of the property to predict.
            (default: :obj:`None`)
        std (float, optional): The standard deviation of the property to
            predict. (default: :obj:`None`)
        atomref (torch.Tensor, optional): The reference of single-atom
            properties.
            Expects a vector of shape :obj:`(max_atomic_number, )`.
    """

    def __init__(self, num_embeddings=100, hidden_channels=256, num_filters=256, num_interactions=4, num_gaussians=150,
                 cutoff=15.0, readout='add', hypernet_update=False, g_hidden_channels=128, f_hidden_channels=64,
                 device='cpu', dipole=False, mean=None, std=None, atomref=None):
        super(NMPEdge, self).__init__()

        assert readout in ['add', 'sum', 'mean']

        self.hidden_channels = hidden_channels
        self.num_embeddings = num_embeddings
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.readout = readout
        self.dipole = dipole
        self.readout = 'add' if self.dipole else self.readout
        self.mean = mean
        self.std = std
        self.scale = None
        self.hypernet_update = hypernet_update
        self.f_hidden_channels = f_hidden_channels
        self.g_hidden_channels = g_hidden_channels
        self.device = device
        atomic_mass = torch.from_numpy(ase.data.atomic_masses)
        self.register_buffer('atomic_mass', atomic_mass)

        self.embedding = nn.Embedding(num_embeddings, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.edge_updates = self.init_edge_list()
        self.msg_passes = self.init_msg_list()
        self.state_transitions = self.init_state_list()
        ### readout function ###
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.fc2 = nn.Linear(hidden_channels // 2, 1)
        ### readout function ###
        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = nn.Embedding(num_embeddings, 1)
            self.atomref.weight.data.copy_(atomref)

        self.reset_parameters()

    def init_edge_list(self):
        edge_updates = nn.ModuleList()
        edge_net0 = EdgeUpdate(self.num_gaussians + 2 * self.hidden_channels, self.hidden_channels)
        edge_updates.append(edge_net0)
        for _ in range(self.num_interactions - 1):
            edge_net = EdgeUpdate(3 * self.hidden_channels, self.hidden_channels)
            edge_updates.append(edge_net)
        return edge_updates

    def init_msg_list(self):
        msg_passes = nn.ModuleList()
        for _ in range(self.num_interactions):
            msg_pass = MessageFunction(self.hidden_channels, self.num_filters)
            msg_passes.append(msg_pass)
        return msg_passes

    def init_state_list(self):
        state_transitions = nn.ModuleList()
        if self.hypernet_update:
            state_transition = StateHyper(self.hidden_channels, self.f_hidden_channels, self.g_hidden_channels,
                                          self.device)
            for _ in range(self.num_interactions):
                state_transitions.append(state_transition)
        else:
            for _ in range(self.num_interactions):
                state_transition = StateMLP(self.hidden_channels)
                state_transitions.append(state_transition)

        return state_transitions

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for i in range(self.num_interactions):
            self.msg_passes[i].reset_parameters()
            self.edge_updates[i].reset_parameters()
            self.state_transitions[i].reset_parameters()
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0)
        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)

    def forward(self, z, pos, batch=None):
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch
        h = self.embedding(z)
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        e = self.distance_expansion(edge_weight)
        h0 = h.clone()
        s_t = None
        for i in range(self.num_interactions):
            e = self.edge_updates[i](h, edge_index, e)
            msg = self.msg_passes[i](h, edge_index, e)
            if self.hypernet_update:
                s_t = self.state_transitions[i](h0, h, msg)
            else:
                s_t = self.state_transitions[i](msg)
            h = h + s_t

        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)

        if self.dipole:
            # Get center of mass.
            mass = self.atomic_mass[z].view(-1, 1)
            c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
            h = h * (pos - c[batch])

        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)

        out = scatter(h, batch, dim=0, reduce=self.readout)

        if self.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)

        if self.scale is not None:
            out = self.scale * out

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'cutoff={self.cutoff})')


class MessageFunction(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters):
        super(MessageFunction, self).__init__()
        self.filter_mlp = nn.Sequential(
            nn.Linear(hidden_channels, num_filters),
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_filters),
            ShiftedSoftplus()
        )
        self.cf_conv = CFConv(hidden_channels, num_filters, self.filter_mlp)
        self.reset_parameters()

    def reset_parameters(self):
        self.cf_conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.filter_mlp[0].weight)
        self.filter_mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.filter_mlp[2].weight)
        self.filter_mlp[2].bias.data.fill_(0)

    def forward(self, h, edge_index, edge_attr):
        x = self.cf_conv(h, edge_index, edge_attr)
        return x


class EdgeUpdate(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(EdgeUpdate, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels * 2)
        self.fc2 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.act = ShiftedSoftplus()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        x_j = x[edge_index[0, :], :]
        x_i = x[edge_index[1, :], :]
        edge_attr = torch.cat([x_i, x_j, edge_attr], dim=1)
        edge_attr = self.fc1(edge_attr)
        edge_attr = self.act(edge_attr)
        edge_attr = self.fc2(edge_attr)
        # edge_attr = self.act(edge_attr)
        return edge_attr


class StateHyper(nn.Module):
    def __init__(self, hidden_channels, f_hidden_channels=64, g_hidden_channels=128, device="cpu"):
        super(StateHyper, self).__init__()
        self.hidden_channels = hidden_channels
        self.f_hidden_channels = f_hidden_channels
        self.g_hidden_channels = g_hidden_channels
        self.c = nn.Parameter(torch.rand(1), requires_grad=True)
        self.device = device

        self.fc1 = nn.Linear(self.hidden_channels, self.f_hidden_channels, bias=False)
        self.fc2 = nn.Linear(self.f_hidden_channels, self.f_hidden_channels, bias=False)
        self.fc3 = nn.Linear(self.f_hidden_channels, self.f_hidden_channels, bias=False)
        self.fc4 = nn.Linear(self.f_hidden_channels, 2 * (self.hidden_channels * self.g_hidden_channels), bias=False)
        self.act = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        torch.nn.init.uniform_(self.c, 0, 1)

    def forward(self, h_0, h_t, msg):
        if self.c < 0.0:
            self.c.data = torch.tensor([0.0]).float().to(self.device).data
        elif self.c > 1.0:
            self.c.data = torch.tensor([1.0]).float().to(self.device).data
        h = self.c * h_0 + (1 - self.c) * h_t
        ######## f ########
        f1_out = self.fc1(h)
        f2_in = self.act(f1_out)
        f2_out = self.fc2(f2_in)
        f3_in = self.act(f2_out)
        f3_out = self.fc3(f3_in)
        f4_in = self.act(f3_out)
        f4_out = self.fc4(f4_in)
        ######## f ########
        g1_weights = f4_out[:, :self.hidden_channels * self.g_hidden_channels].view(-1, self.g_hidden_channels,
                                                                                    self.hidden_channels)
        g2_weights = f4_out[:, self.hidden_channels * self.g_hidden_channels:].view(-1, self.hidden_channels,
                                                                                    self.g_hidden_channels)
        ####### g #######
        g1_out = torch.bmm(msg.unsqueeze(1), g1_weights.transpose(1, 2))
        g2_in = self.act(g1_out)
        g2_out = torch.bmm(g2_in, g2_weights.transpose(1, 2))
        g2_out = g2_out.squeeze()
        out = self.act(g2_out)
        ####### g #######
        return out


class StateMLP(nn.Module):
    def __init__(self, hidden_channels):
        super(StateMLP, self).__init__()
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.act = ShiftedSoftplus()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class CFConv(MessagePassing):
    def __init__(self, in_channels, num_filters, filter_nn):
        super(CFConv, self).__init__(aggr='add')
        self.fc1 = nn.Linear(in_channels, num_filters, bias=False)
        self.filter_nn = filter_nn
        # self.cutoff = cutoff
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x, edge_index, edge_attr):
        W = self.filter_nn(edge_attr)
        x_msg = self.propagate(edge_index, x=x, W=W)
        return x_msg

    def message(self, x_j, W):
        x_w = self.fc1(x_j)
        return x_w * W


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        stop = stop - (stop - start) / num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = - 0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift

