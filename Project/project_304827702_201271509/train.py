
import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from model.gatgcn import GATGCN
from model.gat import HyperGATConv
from model.gcn import HyperGCNConv


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(description="Model parameters")
    ###### dataset parameters ######
    parser.add_argument("--folder_params", type=str, default="input", choices=["input"])
    parser.add_argument("--file_params", type=str, default="pubmed_gatgcn_hyper_params", choices=[
        "pubmed_gatgat_params", "pubmed_gatgat_hyper_params", "pubmed_gatgcn_params",
        "pubmed_gatgcn_hyper_params", "pubmed_gcngcn_params", "pubmed_gcngcn_hyper_params",
        "cora_gatgat_params", "cora_gatgat_hyper_params", "cora_gatgcn_params",
        "cora_gatgcn_hyper_params", "cora_gcngcn_params", "cora_gcngcn_hyper_params",
    ])
    return parser.parse_args(arg_list)


class Trainer:
    def __init__(self, model, optimizer, device, params):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.eval_every = params["eval_every"]
        self.decay_factor = params["decay_factor"]
        self.decay_every = params["decay_every"]
        self.save_model = params["save_model"]
        if self.save_model:
            self.checkpoint_folder = os.path.join(os.getcwd(), 'checkpoint')
            os.makedirs(self.checkpoint_folder, exist_ok=True)
        self.model_name = self.set_model_name(params)

    def set_model_name(self, params):
        dataset, seed = params["dataset"], params["seed"]
        models_name = self.model.models_name
        models_str = ''
        for i, model_name in enumerate(models_name, 1):
            models_str += f'model{i}={model_name}_'
        name = f'{dataset}_{models_str}hypernetworks={self.model.use_hypernetworks}_seed={seed}'
        return name

    def update_lr(self):
        for i in range(len(self.optimizer.param_groups)):
            self.optimizer.param_groups[i]["lr"] *= self.decay_factor

    def predict(self, data, masks=['train_mask']):
        accs = []
        with torch.no_grad():
            self.model.eval()
            logits = self.model(data)
            for _, mask in data(*masks):
                pred = logits[mask].max(1)[1]  # [0] is the values and [1] is the relevant class
                acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
                accs.append(acc)
        self.model.train()
        return accs

    def fit(self, data, epochs):
        best_val_acc, best_test_acc = 0, 0
        best_c_hyper, best_c_out = 0.5, 0.5
        self.model.train()
        for epoch in range(1, epochs + 1):
            self.model.set_cs_grad(epoch, requires_grad=True)
            self.optimizer.zero_grad()
            pred = self.model(data)[data.train_mask]
            target = data.y[data.train_mask]
            loss = F.nll_loss(pred, target)
            loss.backward()
            self.model.set_cs_grad(epoch, requires_grad=False)
            self.optimizer.step()
            self.model.clip_cs(self.device)
            if epoch % self.eval_every == 0:
                train_acc, val_acc, test_acc = self.predict(data, ['train_mask', 'val_mask', 'test_mask'])
                best_c_out = self.model.c_out.item()
                best_c_hyper = self.model.c_hyper.item() if self.model.c_hyper is not None else 0.5
                print(f'Epoch: {epoch:03d}\tTrain: {train_acc:.5f}\tVal: {val_acc:.4f}\tTest: {test_acc:.4f}')
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    if self.save_model:
                        self.save_checkpoint(epoch)

            if epoch % self.decay_every == 0:
                self.update_lr()

        return best_val_acc, best_test_acc, best_c_hyper, best_c_out

    def save_checkpoint(self, epoch):
        full_path = os.path.join(self.checkpoint_folder, f'{self.model_name}.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'device': self.device,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch
        }, full_path)

    def get_pretrained_model_path(self):
        path_model = os.path.join(self.checkpoint_folder, f'{self.model_name}.pth')
        if os.path.exists(path_model):
            return path_model
        else:
            return None

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.device = checkpoint['device']
        self.model = self.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def get_dataset_properties(params):
    dataset_name = params["dataset"]
    split = params["split"]
    path = os.path.join(os.getcwd(), "dataset", dataset_name)
    os.makedirs(path, exist_ok=True)
    torch.manual_seed(params["seed"])
    dataset = Planetoid(path, dataset_name, split=split, transform=T.NormalizeFeatures())
    data = dataset[0]
    n_features = dataset.num_features
    n_classes = dataset.num_classes
    return data, n_features, n_classes


def set_model_layers(models_dict, model_number, n_features, n_classes, use_hypernetworks):
    model_dict = models_dict[f"model{model_number}"]
    model_name = model_dict["name"]
    model_layers = None
    if model_name == "gat":
        layer1_params = model_dict["params1"]
        out_channels1 = layer1_params["out_channels"]
        heads1 = layer1_params["heads"]
        p_att1 = layer1_params["p_att"]
        bias1 = layer1_params["bias"]
        in_channels2 = out_channels1 * heads1
        layer2_params = model_dict["params2"]
        heads2 = layer2_params["heads"]
        p_att2 = layer2_params["p_att"]
        bias2 = layer2_params["bias"]

        model_layers = nn.ModuleList([
            HyperGATConv(n_features, out_channels1, heads1, dropout=p_att1, bias=bias1),
            HyperGATConv(in_channels2, n_classes, heads2, concat=False, dropout=p_att2, bias=bias2,
                         use_hypernetworks=use_hypernetworks)
        ])
    elif model_name == "gcn":
        layer1_params = model_dict["params1"]
        out_channels1 = layer1_params["out_channels"]
        cached1 = layer1_params["cached"]
        bias1 = layer1_params["bias"]
        normalize1 = layer1_params["normalize"]
        layer2_params = model_dict["params2"]
        cached2 = layer2_params["cached"]
        bias2 = layer2_params["bias"]
        normalize2 = layer2_params["normalize"]

        model_layers = nn.ModuleList([
            HyperGCNConv(n_features, out_channels1, cached=cached1, bias=bias1, normalize=normalize1),
            HyperGCNConv(out_channels1, n_classes, cached=cached2, bias=bias2, normalize=normalize2,
                         use_hypernetworks=use_hypernetworks)
        ])

    return model_name, model_layers


def create_model(params, n_features, n_classes):
    use_hypernetworks = params["use_hypernetworks"]
    model1_name, model1_layers = set_model_layers(params["models_dict"], 1, n_features, n_classes, use_hypernetworks)
    model2_name, model2_layers = set_model_layers(params["models_dict"], 2, n_features, n_classes, use_hypernetworks)
    models_name = [model1_name, model2_name]
    c_dict = params["c_dict"]
    p_input = params["p_input"]  # default 0.6
    p_hyper = params["p_hyper"]
    f_n_hidden = params["f_n_hidden"]  # best 4?
    f_hidden_size = params["f_hidden_size"]  # best 128
    model = GATGCN(model1_layers, model2_layers, models_name=models_name, c_dict=c_dict, p=p_input, p_hyper=p_hyper,
                   use_hypernetworks=use_hypernetworks, f_n_hidden=f_n_hidden, f_hidden_size=f_hidden_size)
    return model


def get_lr_wd(params, model_number=None, layer_number=None):
    lr, wd = 0.0, 0.0
    if model_number is not None:
        lr = params[f"model{model_number}"][f"hyperparams{layer_number}"]["lr"]
        wd = params[f"model{model_number}"][f"hyperparams{layer_number}"]["weight_decay"]
    else:
        lr = params["lr"]
        wd = params["weight_decay"]
    return lr, wd


def set_optimizer(model, params):
    lr, weight_decay = get_lr_wd(params)
    models_dict = params["models_dict"]
    optimizer = None
    lr11, wd11 = get_lr_wd(models_dict, 1, 1)
    lr12, wd12 = get_lr_wd(models_dict, 1, 2)
    lr21, wd21 = get_lr_wd(models_dict, 2, 1)
    lr22, wd22 = get_lr_wd(models_dict, 2, 2)
    if params["use_hypernetworks"]:
        lr_hyper, wd_hyper = get_lr_wd(params["hypernetworks_dict"])
        optimizer = torch.optim.Adam([
            dict(params=model.model1_layers[0].parameters(), lr=lr11, weight_decay=wd11),
            dict(params=model.model1_layers[1].parameters(), lr=lr12, weight_decay=wd12),
            dict(params=model.model2_layers[0].parameters(), lr=lr21, weight_decay=wd21),
            dict(params=model.model2_layers[1].parameters(), lr=lr22, weight_decay=wd22),
            dict(params=model.hyper_params.parameters(), lr=lr_hyper, weight_decay=wd_hyper),
            dict(params=model.c_hyper),
            dict(params=model.c_out)
        ], lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam([
            dict(params=model.model1_layers[0].parameters(), lr=lr11, weight_decay=wd11),
            dict(params=model.model1_layers[1].parameters(), lr=lr12, weight_decay=wd12),
            dict(params=model.model2_layers[0].parameters(), lr=lr21, weight_decay=wd21),
            dict(params=model.model2_layers[1].parameters(), lr=lr22, weight_decay=wd22),
            dict(params=model.c_out)   # no unique lr, wd?
        ], lr=lr, weight_decay=weight_decay)
    return optimizer


def main(args):
    full_path = os.path.join(os.getcwd(), args.folder_params, f'{args.file_params}.json')
    with open(full_path) as json_file:
        run_params = json.load(json_file)
    data, n_features, n_classes = get_dataset_properties(run_params)
    model = create_model(run_params, n_features, n_classes)
    device = torch.device(f'cuda:{run_params["cuda"]}' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    model = model.to(device)
    optimizer = set_optimizer(model, run_params)
    trainer = Trainer(model, optimizer, device, run_params)
    if run_params["load_pretrained_model"]:
        pretrained_model_path = trainer.get_pretrained_model_path()
        if pretrained_model_path is not None:
            trainer.load_model(pretrained_model_path)
            train_acc, val_acc, test_acc = trainer.predict(data, ['train_mask', 'val_mask', 'test_mask'])
            print(f'Pretrained model {trainer.model_name} Accuracies:')
            print(f'Train: {train_acc:.4f}\tVal: {val_acc:.4f}\tTest: {test_acc:.4f}')
        else:
            print(f"pretrained model path does not exist")
    else:
        epochs = run_params["epochs"]
        val_acc, test_acc, c_hyper, c_out = trainer.fit(data, epochs)
        acc_str = f'validation accuracy={val_acc:.4f}\ttest accuracy={test_acc:.4f}'
        if run_params["print_Cs"]:
            c_str = f'c_out={c_out:.3f}\tc_hyper={c_hyper:.3f}'
            print(f'Model {trainer.model_name}:\n{acc_str}\t{c_str}')
        else:
            print(f'Model {trainer.model_name}:\n{acc_str}')


if __name__ == "__main__":
    arguments = get_arguments()
    main(arguments)
