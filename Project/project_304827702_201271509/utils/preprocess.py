
import numpy as np
import torch
from torch_geometric.data import Data


class TransductivePreprocessing:
    def __init__(self, nodes_per_class=20, n_validation=500, n_test=1000, seed=0):
        # self.seed = seed
        self.nodes_per_class = nodes_per_class
        self.n_validation = n_validation
        self.n_test = n_test
        np.random.seed(seed)

    @staticmethod
    def create_label_idx(label_idx, n_nodes):
        rel_label_idx = np.random.choice(label_idx, n_nodes, replace=False)
        return rel_label_idx

    @staticmethod
    def set_size_per_label(N, labels):
        node_per_class = N // len(labels)
        resid = N - node_per_class * len(labels)
        labels_split = {label: node_per_class for label in labels}
        labels_split[np.random.choice(labels)] += resid
        return labels_split

    @staticmethod
    def set_mask(N, idx):
        mask = np.zeros(N, dtype=np.bool)
        mask[idx] = True  # np.True_
        return mask

    def train_val_test_idx_split(self, y):
        labels = y.unique().numpy()
        N = len(y)
        idx_arr = np.arange(N)
        train_idx, val_idx, test_idx = [], [], []
        val_label_split = self.set_size_per_label(self.n_validation, labels)
        test_label_split = self.set_size_per_label(self.n_test, labels)
        for label in labels:
            label_idx = idx_arr[y == label]
            train_label_idx = self.create_label_idx(label_idx, self.nodes_per_class)
            train_idx += train_label_idx.tolist()
            rel_idx = np.setxor1d(label_idx, train_label_idx)
            val_label_idx = self.create_label_idx(rel_idx, val_label_split[label])
            val_idx += val_label_idx.tolist()
            rel_idx = np.setxor1d(rel_idx, val_label_idx)
            test_label_idx = self.create_label_idx(rel_idx, test_label_split[label])
            test_idx += test_label_idx.tolist()

        return train_idx, val_idx, test_idx
        # train_mask = self.set_mask(N, train_idx)
        # val_mask = self.set_mask(N, val_idx)
        # test_mask = self.set_mask(N, test_idx)
        # return train_mask, val_mask, test_mask

    def create_dataset(self, data, idx, data_size):
        data_list = []
        cur_idx = idx.copy()
        N = len(cur_idx)
        while N >= data_size:
            rel_idx = np.random.choice(cur_idx, data_size, replace=False)
            data_idx = torch.from_numpy(rel_idx).long()
            cur_data = Data(x=data.x[data_idx], y=data.y[data_idx], edge_index=data.edge_index)
            data_list.append(cur_data)
            cur_idx = np.setxor1d(cur_idx, rel_idx)
            N = len(cur_idx)

        if N > 0:
            data_idx = torch.from_numpy(cur_idx).long()
            cur_data = Data(x=data.x[data_idx], y=data.y[data_idx], edge_index=data.edge_index)
            data_list.append(cur_data)
        return data_list





