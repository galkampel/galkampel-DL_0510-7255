
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATGCN(nn.Module):
    r"""
    Args:
        model1_layers (nn.ModuleList): list of layers for model 1
        model2_layers (nn.ModuleList): list of layers for model 2
        models_name (list): names of model1 and model2 respectively
        c_dict (dictionary): key: out or hyper, value: tuple: (requires_grad, init_val)
        p (float, optional): probability to zero out features (mask)
        p_hyper (float, optional): probability to zero out hypernetworks' features (mask)
        use_hypernetowrks (bool, optional): use hypernetworks the learon the weights of the two models simultaneously
        f_n_hidden (int, optinal): number of hidden layers of hypernetworks
        f_hidden_size (int, optional): number of neurons for a hypernetworks' hidden layer
    """
    def __init__(self, model1_layers, model2_layers, models_name, c_dict, p=0.0, p_hyper=0.0, use_hypernetworks=False,
                 f_n_hidden=1, f_hidden_size=128):
        super(GATGCN, self).__init__()
        self.model1_layers = model1_layers
        self.model2_layers = model2_layers
        self.models_name = models_name
        self.p = p
        self.use_hypernetworks = use_hypernetworks
        self.c_dict = c_dict
        self.c_out = None
        self.c_hyper = None
        self.hyper_params = None
        self.p_hyper = p_hyper
        if self.use_hypernetworks:
            self.lin_f_in = None
            self.lins_f_hidden = None
            self.lin_f_out = None
            self.f_n_hidden = f_n_hidden
            self.init_hypernetwork(f_hidden_size)
        self.reset_parameters()

    def get_total_params(self, model_number, layer):
        if self.models_name[model_number] == 'gat':
            in_channels, out_channels, heads = layer.get_dims()
            return (in_channels + 2) * heads * out_channels
        elif self.models_name[model_number] == 'gcn':
            in_channels, out_channels = layer.get_dims()
            return in_channels * out_channels

    def init_hypernetwork(self, hidden_size):
        total_params1 = self.get_total_params(0, self.model1_layers[1])
        total_params2 = self.get_total_params(1, self.model2_layers[1])
        layer_dims = self.model1_layers[1].get_dims()
        self.lin_f_in = nn.Linear(layer_dims[0], hidden_size, bias=False)  # default bias=False
        self.lins_f_hidden = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=False) for i in range(1, self.f_n_hidden)])  # default bias=False
        self.lin_f_out = nn.Linear(hidden_size, total_params1 + total_params2, bias=False)
        self.hyper_params = nn.ModuleList([ self.lin_f_in, self.lin_f_out]) if self.f_n_hidden == 1 else \
                            nn.ModuleList([self.lin_f_in, self.lins_f_hidden, self.lin_f_out])

    def reset_parameters(self):
        c_init = self.c_dict["out"].setdefault("init_val", 0.5)
        self.c_out = nn.Parameter(torch.Tensor([c_init]), requires_grad=False)
        if self.use_hypernetworks:
            c_init = self.c_dict["hyper"].setdefault("init_val", 0.5)
            self.c_hyper = nn.Parameter(torch.Tensor([c_init]), requires_grad=False)
            nn.init.xavier_uniform_(self.lin_f_in.weight)
            for i in range(1, self.f_n_hidden):
                nn.init.xavier_uniform_(self.lins_f_hidden[i - 1].weight)
            nn.init.xavier_uniform_(self.lin_f_out.weight)

    @staticmethod
    def create_mask(x, p, training):
        device = x.device
        mask = torch.ones_like(x, dtype=torch.float, device=device)
        if training:
            mask = (torch.rand(*x.shape, dtype=torch.float, device=device) > p) / (1 - p)
        return mask

    def create_models_params(self, x, edge_index, edge_weight, i):
        if self.models_name[i] == 'gcn':
            return F.relu, [x, edge_index, edge_weight]
        else:
            return F.elu, [x, edge_index]

    def set_layer_weights(self, f_out, layer, model_number, start=0):
        if self.models_name[model_number] == 'gat':
            in_channels, out_channels, heads = layer.get_dims()
            W_size = in_channels * heads * out_channels
            att_size = heads * out_channels
            W_hyper = f_out[:, start:start + W_size].view(-1, heads * out_channels, in_channels)
            att_i = f_out[:, start + W_size: start + W_size + att_size].view(-1, heads, out_channels)
            att_j = f_out[:, start + W_size + att_size: start + W_size + 2 * att_size].view(-1, heads, out_channels)
            layer.set_hypernetworks_weights(W_hyper, att_i, att_j)
            return start + W_size + 2 * att_size
        elif self.models_name[model_number] == 'gcn':
            in_channels, out_channels = layer.get_dims()
            W_size = in_channels * out_channels
            W_hyper = f_out[:, start:start + W_size].view(-1, out_channels, in_channels)
            layer.set_hypernetworks_weights(W_hyper)
            return start + W_size

    def clip_cs(self,  device):
        if self.c_out is not None:
            if self.c_out < 0.0:
                self.c_out.data = torch.tensor([0.0]).float().to(device).data
            elif self.c_out > 1.0:
                self.c_out.data = torch.tensor([1.0]).float().to(device).data
        if self.c_hyper is not None:
            if self.c_hyper < 0.0:
                self.c_hyper.data = torch.tensor([0.0]).float().to(device).data
            elif self.c_hyper > 1.0:
                self.c_hyper.data = torch.tensor([1.0]).float().to(device).data

    def set_cs_grad(self, epoch, requires_grad=True):
        if self.c_out is not None:
            if self.c_dict['out'].setdefault("requires_grad", False) and requires_grad and \
                        epoch % self.c_dict['out'].setdefault("update_every", 1) == 0:
                self.c_out.requires_grad = requires_grad
            elif self.c_dict['out'].setdefault("requires_grad", False) and not requires_grad and \
                        epoch % self.c_dict['out'].setdefault("update_every", 1) == 0:
                self.c_out.requires_grad = requires_grad

        if self.c_hyper is not None:
            if self.c_dict['hyper'].setdefault("requires_grad", False) and requires_grad and \
                        epoch % self.c_dict['hyper'].setdefault("update_every", 1) == 0:
                self.c_hyper.requires_grad = requires_grad
            elif self.c_dict['hyper'].setdefault("requires_grad", False) and not requires_grad and \
                        epoch % self.c_dict['hyper'].setdefault("update_every", 1) == 0:
                self.c_hyper.requires_grad = requires_grad

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        mask = self.create_mask(x, p=self.p, training=self.training)
        x = x * mask
        model1_act, model1_params = self.create_models_params(x, edge_index, edge_weight, 0)
        model2_act, model2_params = self.create_models_params(x, edge_index, edge_weight, 1)
        h_model1 = model1_act(self.model1_layers[0](* model1_params))
        h_model2 = model2_act(self.model2_layers[0](* model2_params))
        mask = self.create_mask(h_model1, self.p, training=self.training)
        if self.use_hypernetworks:
            x_hyper = mask * (self.c_hyper * h_model1 + (1 - self.c_hyper) * h_model2)
            f_out = self.lin_f_in(x_hyper)
            f_out = torch.tanh(f_out)
            f_out = F.dropout(f_out, self.p_hyper)
            for i in range(1, self.f_n_hidden):
                f_out = self.lins_f_hidden[i - 1](f_out)
                f_out = torch.tanh(f_out)
                f_out = F.dropout(f_out, self.p_hyper)
            f_out = self.lin_f_out(f_out)
            start = self.set_layer_weights(f_out, self.model1_layers[1], model_number=0)
            self.set_layer_weights(f_out, self.model2_layers[1], model_number=1, start=start)
        model1_params[0] = h_model1 * mask
        model2_params[0] = h_model2 * mask
        out_model1 = self.model1_layers[1](* model1_params)
        out_model2 = self.model2_layers[1](* model2_params)
        out = self.c_out * out_model1 + (1 - self.c_out) * out_model2
        return F.log_softmax(out, dim=1)
