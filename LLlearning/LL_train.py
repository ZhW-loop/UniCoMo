import sys, os, pickle
sys.path.append("tensorize_notation")
from utils import EWC, ewc_train, normal_train
from tensorize_notation.TN_train import LambdaRankLoss, SegmentDataLoader
from catDataLoaderLL import CatSegmentDataLoaderforTrain

from tqdm import tqdm
import random
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import argparse
from common import TensorizeSet_dir

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
epochs = 50
batch_size = 1024

TNortlp = "TN"


device = "cuda:0"

class AttentionModule(nn.Module):  
    def __init__(self, fea_size, step_size):
        super().__init__()
        self.fea_size = fea_size
        self.step_size = step_size
        self.res_block_cnt = 2
        self.attention_matrix = None
        in_dim = self.fea_size
        hidden_dim = [64, 128, 256, 256]
        out_dim = [256, 128, 64, 1]
        hidden_dim_1 = hidden_dim[-1]

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.ReLU(),
            nn.Linear(hidden_dim[2], hidden_dim[3]),
            nn.ReLU(),
        )
        
        # self.position_encoding = self._get_position_encoding(self.step_size, hidden_dim_1)
        attention_head = 8
        self.attention = nn.MultiheadAttention(
            hidden_dim_1, attention_head)

        self.l0 = nn.Sequential(
            nn.Linear(hidden_dim_1, hidden_dim_1),
            nn.ReLU(),
        )
        self.l_list = []
        for i in range(self.res_block_cnt):
            self.l_list.append(nn.Sequential(
                nn.Linear(hidden_dim_1, hidden_dim_1), 
                nn.ReLU()
            ))
        self.l_list = nn.Sequential(*self.l_list)

        # self.layer_norm = nn.LayerNorm(hidden_dim_1)  # Add LayerNorm layer
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim_1, out_dim[0]),
            nn.ReLU(),
            nn.Linear(out_dim[0], out_dim[1]),
            nn.ReLU(),
            nn.Linear(out_dim[1], out_dim[2]),
            nn.ReLU(),
            nn.Linear(out_dim[2], out_dim[3]),
        )
    
    # def _get_position_encoding(self, seq_len, dim):
    #     position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    #     div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
    #     position_encoding = torch.zeros((seq_len, dim))
    #     position_encoding[:, 0::2] = torch.sin(position * div_term)
    #     position_encoding[:, 1::2] = torch.cos(position * div_term)
    #     return position_encoding

    def forward(self, batch_datas_steps):

        # batch_datas_steps = batch_datas_steps[:, :self.step_size, :self.fea_size]
        
        encoder_output = self.encoder(batch_datas_steps)

        # # Add position encoding
        # encoder_output = encoder_output + self.position_encoding.to(encoder_output.device)
        
        encoder_output = encoder_output.transpose(0, 1)
        
        output, self.attention_matrix = \
            self.attention(encoder_output, encoder_output, encoder_output)
        
        # output = self.layer_norm(output)  # Apply LayerNorm
        
        output = output + encoder_output
        
        # output = encoder_output
        
        for l in self.l_list:
            output = l(output) + output

        output = self.decoder(output).sum(0)

        return output.squeeze()


def standard_process(fea_size, step_size):
    if last_task_id < 0:
        model = AttentionModule(fea_size, step_size).to(device)
        print("new model")
    else:
        load_name = os.path.join(f"{TensorizeSet_dir}/LL_records/normal", 
                                TNortlp, 
                                f"task{last_task_id}_models", 
                                f"{TNortlp}_model_{last_task_model_id}.pkl")
        with open(load_name, 'rb') as f:
            model = pickle.load(f).to(device)
        print("recover model")
    
    train_loader_path = os.path.join(f'{TensorizeSet_dir}/LL_records/tasks_dataset', f"{TNortlp}", f"task{new_task_id}_train.pkl.DataLoader")
    with open(train_loader_path, 'rb') as f:
        train_loader = pickle.load(f)
    
    loss_func = LambdaRankLoss(device)
    optimizer = optim.Adam(model.parameters(), lr=7e-4, weight_decay=1e-6)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=epochs // 3, gamma=1)

    for epoch in tqdm(range(epochs)):
        normal_train(TNortlp, new_task_id, epoch, model, optimizer, train_loader, loss_func, lr_scheduler)

def cut_train_loader(train_loader: SegmentDataLoader):
    length = len(train_loader.datas_steps)
    sample_size = 500
    indices = torch.randint(0, length, (sample_size,))
    
    train_loader.datas_steps = train_loader.datas_steps[indices]
    train_loader.labels = train_loader.labels[indices]
    train_loader.min_latency = train_loader.labels[indices]
    train_loader.number = len(train_loader.datas_steps)
    
    return train_loader
    

def ewc_process(epochs, importance):
    if last_task_id < 0:
        model = AttentionModule().to(device)
    else:
        load_name = os.path.join(f"{TensorizeSet_dir}/LL_records/ewc", 
                                TNortlp, 
                                f"task{last_task_id}_models", 
                                f"{TNortlp}_model_{last_task_model_id}.pkl")
        with open(load_name, 'rb') as f:
            model = pickle.load(f).to(device)
    
    train_loader_path = os.path.join(f'{TensorizeSet_dir}/LL_records/tasks_dataset', f"{TNortlp}", f"task{new_task_id}_train.pkl.DataLoader")
    with open(train_loader_path, 'rb') as f:
        train_loader = pickle.load(f)
    
    old_small_train_loader_path = []
    for sub_task in [9, 0, 1, 2, 4]:
        if sub_task == new_task_id: break
        old_small_train_loader_path.append(
            str(os.path.join(f'{TensorizeSet_dir}/LL_records/tasks_dataset', f"{TNortlp}", f"task{sub_task}_small_train.pkl.DataLoader"))
        )
    small_train_dataloaders = []
    for pkl_dataloader_path in old_small_train_loader_path:
        pkl_dataloader_path : str
        assert pkl_dataloader_path.endswith(".pkl.DataLoader")
        with open(pkl_dataloader_path, 'rb') as f:
            train_dataloader = pickle.load(f)
        train_dataloader: SegmentDataLoader
        small_train_dataloaders.append(train_dataloader)
    small_train_main_loader = small_train_dataloaders[0]
    for i in range(len(small_train_dataloaders)):
        if i == 0: continue
        small_train_main_loader = CatSegmentDataLoaderforTrain(small_train_main_loader, small_train_dataloaders[i])
    loss_func = LambdaRankLoss(device)
    optimizer = optim.Adam(model.parameters(), lr=7e-4, weight_decay=1e-6)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=epochs // 3, gamma=1)
    
    for epoch in tqdm(range(epochs)):
        ewc_train(TNortlp, new_task_id, epoch, model, optimizer, train_loader, loss_func, lr_scheduler, EWC(model, small_train_main_loader), importance)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--last_task_id", type=int, default=0)
    parser.add_argument("--last_task_model_id", type=int, default=37)
    parser.add_argument("--new_task_id", type=int, default=0)
    parser.add_argument("--train_type", type=str, default="standard")
    parser.add_argument("--seq_len", type=int, default=179)
    parser.add_argument("--embed_size", type=int, default=27)
    args = parser.parse_args()
    
    last_task_id = args.last_task_id
    last_task_model_id = args.last_task_model_id
    new_task_id = args.new_task_id
    if args.train_type == "standard":
        standard_process(args.seq_len, args.embed_size)
    else:
        ewc_process(epochs=50, importance=0.1)