import torch
from torch.nn import Parameter, Linear
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


class HyperGATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        use_hypernetworks (boll, optional): if True, weights are taken from hypernetworks
        hidden_channels (int, optional): the hidden size in the hypernetwork
        fixes_c (bool, optional): if set to :obj:`Fasle`, the hypernetwork's input is a convex combination of the
            current hidden state and the projected initial (hidden) state.
            Otherwise, the hypernetwork's input is solely based on the current hidden state
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0, bias=True,
                 use_hypernetworks=False, **kwargs):
        super(HyperGATConv, self).__init__(aggr='add', **kwargs)
        self.in_channels = in_channels  # ct_in
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.use_hypernetworks = use_hypernetworks
        self.__alpha__ = None

        self.lin = Linear(in_channels, heads * out_channels, bias=False) if not self.use_hypernetworks else None
        self.W_hyper = None
        self.att_i = Parameter(torch.Tensor(1, heads, out_channels)) if not self.use_hypernetworks else None
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels)) if not self.use_hypernetworks else None

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if not self.use_hypernetworks:
            nn.init.xavier_uniform_(self.lin.weight)
            nn.init.xavier_uniform_(self.att_i)
            nn.init.xavier_uniform_(self.att_j)

    def get_dims(self):
        return self.in_channels, self.out_channels, self.heads

    def set_hypernetworks_weights(self, W_hyper=None, att_i=None, att_j=None):
        if self.use_hypernetworks and torch.is_tensor(W_hyper):
            self.W_hyper = W_hyper
        if self.use_hypernetworks and torch.is_tensor(att_i):
            self.att_i = att_i
        if self.use_hypernetworks and torch.is_tensor(att_j):
            self.att_j = att_j

    def forward(self, x, edge_index, h0=None, mask=None, return_attention_weights=False):
        """"""
        if torch.is_tensor(x):
            if torch.is_tensor(self.W_hyper) and self.use_hypernetworks:  # hypernetworks that predict W's/(W and att_i, att_j)'s weights
                x = torch.bmm(x.unsqueeze(1), self.W_hyper.transpose(1, 2)).squeeze(1)
            else:
                x = self.lin(x)

            x = (x, x)
        else:
            x = (self.lin(x[0]), self.lin(x[1]))
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x[1].size(self.node_dim))

        out = self.propagate(edge_index, x=x,
                             return_attention_weights=return_attention_weights)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_i, x_j, edge_index_i, edge_index_j, size_i,
                return_attention_weights):
        # Compute attention coefficients.
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        alpha = (x_i * self.att_i).sum(-1) + (x_j * self.att_j).sum(-1) if not self.use_hypernetworks \
            else (x_i * self.att_i[edge_index_i]).sum(-1) + (x_j * self.att_j[edge_index_j]).sum(-1)

        # either broadcasting over the examples or per example (hypernetwork) )
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        if return_attention_weights:
            self.__alpha__ = alpha

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
