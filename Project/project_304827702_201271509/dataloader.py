
import torch
from torch_geometric.datasets import QM9, Planetoid
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import os

SEED = 0


class DatasetLoader:
    def __init__(self):
        self.root = None
        self.dataset = None
        self.seed = SEED
        self.target = None

    @staticmethod
    def set_data_loader(dataset, batch_size=32, shuffle=False):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def save_data_loader(data_loader, folder_path, file_name):
        full_path = os.path.join(folder_path, f'{file_name}.pth')
        torch.save(data_loader, full_path)

    @staticmethod
    def load_data_loader(folder_path, file_name):
        full_path = os.path.join(folder_path, f'{file_name}.pth')
        data_loader = torch.load(full_path)
        return data_loader

    def get_dataset(self): return self.dataset

    def get_target(self): return self.target


class QM9Loader(DatasetLoader):
    def __init__(self, root, target):
        super().__init__()
        self.root = root
        self.dataset = QM9(root=root)
        self.target = target

    def train_test_split(self, train_size=120000, val_size=10000):
        torch.manual_seed(self.seed)
        N = len(self.dataset)
        train_val_set, test_set = torch.utils.data.random_split(self.dataset, [train_size, N - train_size])
        train_set, val_set = torch.utils.data.random_split(train_val_set, [train_size - val_size, val_size])
        return train_set, val_set, test_set


class PlanetoidLoader:
    def __init__(self, root='/home/galkampel/tmp', dataset_name="PubMed", split_type='public'):
        names = {"Cora", "CiteSeer", "PubMed"}
        self.dataset_name = dataset_name
        self.split_type = split_type
        path = os.path.join(root, dataset_name)
        self.dataset = Planetoid(path, dataset_name, split=split_type, transform=T.NormalizeFeatures())

    def get_data(self):
        return self.dataset[0]

    def get_dataset_name(self):
        return self.dataset_name

    def get_split_type(self):
        return self.split_type


