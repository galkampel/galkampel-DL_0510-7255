
import torch
from model.nmp_edge import NMPEdge
import os
import numpy as np
import argparse


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(description="Test parameters")
    parser.add_argument("--data_folder", type=str, default="dataset", choices=["dataset"])
    parser.add_argument("--data_filename", type=str, default="NMPEdge_test", choices=["NMPEdge_test"])
    parser.add_argument("--target", type=int, default=7, choices=range(12))  # 7/10
    parser.add_argument("--gpu_device", type=int, default=1, choices=[0, 1, 2, 3])
    parser.add_argument("--model_folder", type=str, default="checkpoint")
    parser.add_argument("--model_filename", type=str, default="NMPEdge_hypernet_pretrained_target=7")  # NMPEdge_pretrained_target=7/10, NMPEdge_hypernet_pretrained_target=7/10
    parser.add_argument("--readout", type=str, default="add", choices=["add", "mean"])
    parser.add_argument("--hypernet_update", type=bool, default=True, choices=[True, False])
    parser.add_argument("--model_name", type=str, default="NMPEdge")
    return parser.parse_args(arg_list)


class TestPretrained:
    def __init__(self, model_name, target, gpu_device):
        self.model_name = model_name
        self.target = target
        self.model = None
        self.device = torch.device(f'cuda:{gpu_device}' if torch.cuda.is_available() else 'cpu')

    def init_model(self, args):
        model = None
        if self.model_name == "NMPEdge":
            model = NMPEdge(readout=args.readout, hypernet_update=args.hypernet_update, device=self.device)
        return model

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

    def load_testloader(self, data_folder, data_filename):
        path_testloader = os.path.join(data_folder, f'{data_filename}.pth')
        if not os.path.exists(path_testloader):
            print(f'file {path_testloader} does not exist')
            exit()
        test_loader = torch.load(path_testloader)
        return test_loader

    def load_model(self, folder_name, filename, args):
        path_pretrained_model = os.path.join(folder_name, f'{filename}.pth')
        pretrained_params = torch.load(path_pretrained_model)
        self.model = self.init_model(args)
        self.model.load_state_dict(pretrained_params['model_state_dict'])
        self.model = self.model.to(self.device)


def test_pretrained_model(args):
    test_pretrained = TestPretrained(args.model_name, args.target, args.gpu_device)
    model_file_path = os.path.join(args.model_folder, f'{args.model_name}.pth')
    if not os.path.exists(model_file_path):
        print(f'path to file {model_file_path} does not exist')
        exit()
    test_pretrained.load_model(args.model_folder, args.model_name, args)
    test_loader = test_pretrained.load_testloader(args.data_folder, args.data_filename)
    mae = test_pretrained.predict(test_loader)
    hyper_str = 'with hypernetworks' if 'hypernet' in args.model_name else 'without hypernetwroks'
    print(f"{args.model_name} {hyper_str}\ttarget={args.target}:")
    print(f'Test set MAE = {mae}')


if __name__ == "__main__":
    arguments = get_arguments()
    test_pretrained_model(arguments)
