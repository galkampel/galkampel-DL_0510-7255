
import torch
from torch import optim
from model.nmp_edge import NMPEdge
from dataloader import QM9Loader
import os
import numpy as np
import argparse

SEED = 0
optimizers = {'Adam': optim.Adam, 'SGD': optim.SGD}


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(description="Model parameters")
    ###### dataset parameters ######
    parser.add_argument("--dataset", type=str, default="QM9", choices=["QM9"])
    parser.add_argument("--save_test", type=bool, default=True, choices=[True, False])
    # parser.add_argument("--root_path", type=str, default="/home/galkampel/tmp/QM9")  # path to save/load dataset for training
    parser.add_argument("--dataset_folder", type=str, default="dataset")  # path to save/load dataset for training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_size", type=int, default=120000)
    parser.add_argument("--val_size", type=int, default=10000)
    ###### run configuration ######
    parser.add_argument("--model_filename", type=str, default=None)  # load (partially trained) model
    parser.add_argument("--max_iters", type=int, default=int(1e7))
    parser.add_argument("--target", type=int, default=7, choices=range(12))  # checked on {7, 10}
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--decay_every", type=int, default=100000)
    parser.add_argument("--eval_every", type=int, default=50000)
    parser.add_argument("--no_improvement_thres", type=int, default=1000000)  # if there is no improvement within no_improvement_thres update steps - stop
    parser.add_argument("--lr_decay_factor", type=float, default=0.96)
    ###### model parameters ######
    parser.add_argument("--model_name", type=str, default="NMPEdge")
    parser.add_argument("--cutoff", type=float, default=15.0)
    parser.add_argument("--num_passes", type=int, default=4)
    parser.add_argument("--num_gaussians", type=int, default=150)
    parser.add_argument("--embed_size", type=int, default=256)
    # parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--num_filters", type=int, default=256)
    parser.add_argument("--gpu_device", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument("--readout", type=str, default="add", choices=["add", "mean"])
    parser.add_argument("--hypernet_update", type=bool, default=True, choices=[True, False])
    parser.add_argument("--f_hidden_channels", type=int, default=64)
    parser.add_argument("--g_hidden_channels", type=int, default=128)
    return parser.parse_args(arg_list)


class Trainer:
    def __init__(self, model_run, args):
        self.model_name = model_run.get_model_name()
        self.device = model_run.get_device()
        self.model = model_run.get_model().to(self.device)
        self.optimizer = optimizers[args.optimizer](self.model.parameters(), lr=args.learning_rate)
        self.decay_factor = args.lr_decay_factor
        self.decay_every = args.decay_every
        self.eval_every = args.eval_every
        self.no_improvement_thres = args.no_improvement_thres
        self.target = args.target
        self.max_iters = args.max_iters
        self.checkpoint_folder = os.path.join(os.getcwd(), 'checkpoint')
        os.makedirs(self.checkpoint_folder, exist_ok=True)
        self.start_iter = 1
        self.model_filename = args.model_name
        self.is_best_model = False
        if self.model_filename and os.path.exists(os.path.join(self.checkpoint_folder, f'{self.model_filename}.pth')):
            self.load_model()

    def update_lr(self):
        self.optimizer.param_groups[0]["lr"] *= self.decay_factor

    def predict(self, data_loader):
        maes = []
        with torch.no_grad():
            self.model.eval()
            for data_batch in data_loader:
                data_batch = data_batch.to(self.device)
                pred = self.model(data_batch.z, data_batch.pos, data_batch.batch)
                maes.append((pred.view(-1) - data_batch.y[:, self.target]).abs().cpu().numpy())
        self.model.train()
        mae = np.concatenate(maes).mean()
        return mae

    def fit(self, train_loader, val_loader):
        n_iter = self.start_iter
        best_mae = np.inf
        best_iter = n_iter
        has_best_model = self.is_best_model
        while n_iter < self.max_iters and not has_best_model:
            self.model.train()
            for train_batch in train_loader:
                train_batch = train_batch.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(train_batch.z, train_batch.pos, train_batch.batch)
                loss = (pred.view(-1) - train_batch.y[:, self.target]).abs().mean()
                loss.backward()
                # mae = loss.item()
                self.optimizer.step()
                if n_iter % self.eval_every == 0:
                    print(f'iteration: {n_iter}')
                    train_mae = self.predict(train_loader)
                    val_mae = self.predict(val_loader)
                    print(f'train MAE = {train_mae}\nvalidation MAE = {val_mae}')
                    if val_mae < best_mae:
                        self.save_model(n_iter)
                        best_mae = val_mae
                        best_iter = n_iter

                if n_iter % self.decay_every == 0:
                    self.update_lr()

                if n_iter - best_iter >= self.no_improvement_thres:
                    has_best_model = True
                    self.is_best_model = has_best_model
                    self.save_model(best_iter)

                n_iter += 1

    def save_model(self, iteration):
        model_saved_name = f'{self.model_name}_target={self.target}'
        if self.is_best_model:
            model_saved_name = f'{self.model_name}_pretrained_target={self.target}'
        full_path = os.path.join(self.checkpoint_folder, f'{model_saved_name}.pth')
        torch.save({'model_state_dict': self.model.state_dict(),
                    'device': self.device,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'iteration': iteration,
                    'is_best_model': self.is_best_model}, full_path)

    def load_model(self):
        path_model = os.path.join(self.checkpoint_folder, f'{self.model_filename}.pth')
        checkpoint = torch.load(path_model)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.device = checkpoint['device']
        self.model = self.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_iter = checkpoint['iteration']
        self.is_best_model = checkpoint['is_best_model']


class ModelRun:
    def __init__(self, args):
        self.hypernet_update = args.hypernet_update
        gpu_device = args.gpu_device
        self.device = torch.device(f'cuda:{gpu_device}' if torch.cuda.is_available() else 'cpu')
        self.model_name = args.model_name
        self.model = self.set_model(args)

    def get_model(self):
        return self.model

    def get_device(self):
        return self.device

    def get_model_name(self):
        name = self.model_name
        if self.hypernet_update:
            name = f'{name}_hypernet'
        return name

    def set_model(self, args):
        if self.model_name == "NMPEdge":
            return NMPEdge(num_gaussians=args.num_gaussians, cutoff=args.cutoff, num_interactions=args.num_passes,
                           hidden_channels=args.embed_size, num_filters=args.num_filters, readout=args.readout,
                           hypernet_update=self.hypernet_update, g_hidden_channels=args.g_hidden_channels,
                           f_hidden_channels=args.f_hidden_channels, device=self.device)  # no num_embeddings


def main(args):
    data_loader = None
    if args.dataset == "QM9":
        # if not os.path.exists(os.path.join(*(['/'] + args.root_path.split('/')[1:-1]))):
        QM9_path = os.path.join(os.getcwd(), args.dataset_folder, args.dataset)
        # if not os.path.exists(QM9_path):
        #     print(f'{QM9_path} is not a legit directory to save the data')
        #     exit()
        data_loader = QM9Loader(QM9_path, args.target)

    print(f'model name={args.model_name}, use_hypernetworks={args.hypernet_update}, target={args.target}')
    print(f'Readout aggregation={args.readout}, gpu={args.gpu_device}')
    train_set, val_set, test_set = data_loader.train_test_split(args.train_size, args.val_size)
    train_loader = data_loader.set_data_loader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = data_loader.set_data_loader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = data_loader.set_data_loader(test_set, batch_size=args.batch_size, shuffle=False)
    if args.save_test:
        folder_path = os.path.join(os.getcwd(), 'dataset')
        os.makedirs(folder_path, exist_ok=True)
        if not os.path.exists(os.path.join(folder_path, f'{args.model_name}_test.pth')):
            data_loader.save_data_loader(test_loader, folder_path, 'test')
    model_run = ModelRun(args)
    trainer = Trainer(model_run, args)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    arguments = get_arguments()
    main(arguments)

