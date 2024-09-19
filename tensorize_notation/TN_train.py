import os, sys, time, pickle, random, numpy as np, argparse, glob
from torch import nn, optim
import torch.cuda
import torch.backends.cudnn
import torch.nn
import torch.optim
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from common import TensorizeSet_dir

class AttentionModule(nn.Module):  
    def __init__(self):
        super().__init__()
        self.fea_size = args.fea_size
        self.step_size = args.step_size
        self.res_block_cnt = args.res_block_cnt
        self.attention_matrix = None
        in_dim = self.fea_size
        hidden_dim = args.hidden_dim
        out_dim = args.out_dim
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
        
        self.attention = nn.MultiheadAttention(
            hidden_dim_1, args.attention_head)

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

        batch_datas_steps = batch_datas_steps[:, :self.step_size, :self.fea_size]
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


class LambdaRankLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def lamdbaRank_scheme(self, G, D, *args):
        return torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(
            G[:, :, None] - G[:, None, :])

    def forward(self, preds, labels, k=None, eps=1e-10, mu=10., sigma=1.):
        device = self.device
        preds = preds[None, :]
        labels = labels[None, :]
        y_pred = preds.clone()
        y_true = labels.clone()

        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :,
                                          None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)

        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)
        ndcg_at_k_mask = torch.zeros(
            (y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
        ndcg_at_k_mask[:k, :k] = 1

        true_sorted_by_preds.clamp_(min=0.)
        y_true_sorted.clamp_(min=0.)

        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
        D = torch.log2(1. + pos_idxs.float())[None, :]
        maxDCGs = torch.sum(
            ((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        weights = self.lamdbaRank_scheme(G, D, mu, true_sorted_by_preds)

        scores_diffs = (
            y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
        scores_diffs[torch.isnan(scores_diffs)] = 0.
        weighted_probas = (torch.sigmoid(
            sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
        losses = torch.log2(weighted_probas)
        masked_losses = losses[padded_pairs_mask & ndcg_at_k_mask]
        loss = -torch.sum(masked_losses)
        return loss





class SegmentDataLoader:
    def __init__(
            self,
            dataset,
            batch_size,
            shuffle,
    ):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.iter_order = self.pointer = None

        datas_steps = []
        labels = []
        min_latency = []
        for data_idx, data in enumerate(dataset):
            data = data[:3]
            datas_step, label, min_lat = data
            datas_steps.append(datas_step)
            labels.append(label)
            min_latency.append(min_lat)

        self.datas_steps = torch.FloatTensor(datas_steps)
        self.labels = torch.FloatTensor(labels)
        self.min_latency = torch.FloatTensor(min_latency)

        self.number = len(self.datas_steps)

    def __iter__(self):
        if self.shuffle:
            self.iter_order = torch.randperm(self.number)
        else:
            self.iter_order = torch.arange(self.number)
        self.pointer = 0

        return self

    def __next__(self):
        if self.pointer >= self.number:
            raise StopIteration

        batch_indices = self.iter_order[self.pointer: self.pointer + self.batch_size]
        self.pointer += self.batch_size
        return self._fetch_indices(batch_indices)

    def _fetch_indices(self, indices):

        batch_datas_steps = self.datas_steps[indices]
        batch_datas_steps = nn.utils.rnn.pad_sequence(
            batch_datas_steps, batch_first=True)
        batch_labels = self.labels[indices]

        return (batch_datas_steps, batch_labels)

    def __len__(self):
        return self.number



def load_datas(datasets_global):

    datasets = np.array(datasets_global, dtype=object)
    if args.data_cnt > 0:
        # train_len = int(args.data_cnt * 1000 * 0.9)
        train_len = int(args.data_cnt * 1000 * 1)
        perm = np.random.permutation(len(datasets))
        train_indices, val_indices = perm[:train_len], perm[train_len:args.data_cnt * 1000]
    else:
        # train_len = int(len(datasets) * 0.9)
        train_len = int(len(datasets) * 1)
        perm = np.random.permutation(len(datasets))
        train_indices, val_indices = perm[:train_len], perm[train_len:]

    train_datas, val_datas = datasets[train_indices], datasets[val_indices]

    n_gpu = torch.cuda.device_count()
    train_dataloader = SegmentDataLoader(
        train_datas, args.train_size_per_gpu*n_gpu, True)
    val_dataloader = SegmentDataLoader(
        val_datas, args.val_size_per_gpu*n_gpu, False)

    return train_dataloader, val_dataloader

def validate(model, valid_loader, loss_func, device):
    model.eval()
    valid_losses = []

    for batch_datas_steps, batch_labels in valid_loader:
        batch_datas_steps = batch_datas_steps.to(device)
        batch_labels = batch_labels.to(device)

        preds = model(batch_datas_steps)
        valid_losses.append(loss_func(preds, batch_labels).item())

    return np.sum(valid_losses)

def train(train_loader, val_dataloader, device, is_recover = False):
    # n_epoch = 50
    print('config train ...')
    if args.attention_class == 'default':
        args.hidden_dim = [64, 128, 256, 256]
        args.out_dim = [256, 128, 64, 1]
        # args.hidden_dim = [64, 64, 128, 128]
        # args.out_dim = [128, 128, 64, 1]
        net = AttentionModule().to(device)
        print(net.fea_size, net.step_size)
        # net = torch.nn.DataParallel(net).to(torch.cuda.current_device())

    if is_recover:
        import re
        def extract_numbers(input_string):
            numbers = re.findall(r'\d+', input_string)
            return [int(num) for num in numbers]
        TN_models = sorted(glob.glob(os.path.join(args.save_folder, "*")), 
                            key=lambda x: extract_numbers(x.split('/')[-1])[0])
        if len(TN_models) > 0:
            with open(TN_models[-1], 'rb') as f:
                net = pickle.load(f).to(device)
            print("!!!Warning!!!, recover from model ...")
        

    if args.rank_mse == 'rank':
        loss_func = LambdaRankLoss(device)
    else:
        loss_func = nn.MSELoss()

    n_epoch = args.n_epoch
    if args.optimizer == 'default':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # hyper
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=n_epoch // 3, gamma=1)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer, step_size=n_epoch // 7, gamma=0.93)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)

    train_loss = None
    print('start train...')
    print(len(train_loader), len(val_dataloader))
    print(train_loader.batch_size, train_loader.batch_size)
    for epoch in range(n_epoch):
        tic = time.time()

        net.train()
        train_loss = 0
        for batch, (batch_datas_steps, batch_labels) in enumerate(train_loader):
            batch_datas_steps = batch_datas_steps.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            loss = loss_func(net(batch_datas_steps), batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            train_loss += loss.item()
        lr_scheduler.step()

        train_time = time.time() - tic

        if epoch % 5 == 0 or epoch == n_epoch - 1 or True:

            valid_loss = validate(net, val_dataloader,
                                  loss_func, device=device)
            loss_msg = "Train Loss: %.4f\tValid Loss: %.4f" % (
                train_loss, valid_loss)
            print("Epoch: %d\tBatch: %d\t%s\tTrain Speed: %.0f" % (
                epoch, batch, loss_msg, len(train_loader) / train_time,))

        model_save_file_name = '%s/TN_model_%d.pkl' % (args.save_folder, epoch)
        with open(model_save_file_name, 'wb') as f:
            pickle.dump(net.cpu(), f)
        net = net.to(device)






def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seed(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intrin_name", type=str, default=f'avx512')
    parser.add_argument("--save_folder", type=str, default='')
    parser.add_argument("--cuda", type=str, default='cuda:0')
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--rank_mse", type=str, default='rank')
    parser.add_argument("--optimizer", type=str, default='default')
    parser.add_argument("--attention_head", type=int, default=8)
    parser.add_argument("--attention_class", type=str, default='default')
    parser.add_argument("--step_size", type=int, default=0)
    parser.add_argument("--fea_size", type=int, default=0)
    parser.add_argument("--res_block_cnt", type=int, default=2)
    parser.add_argument("--self_sup_model", type=str, default='')
    parser.add_argument("--data_cnt", type=int, default=-1)  # data_cnt * 1000

    parser.add_argument("--train_size_per_gpu", type=int, default=1024)
    parser.add_argument("--val_size_per_gpu", type=int, default=1024)
    parser.add_argument("--n_epoch", type=int, default=53)
    args = parser.parse_args()
    
    print(args)
    
    intrin_name = args.intrin_name
    measured_path = os.path.join(TensorizeSet_dir, f'{intrin_name}_dataset/measured_records') 
    
    dataset_save_path = os.path.join(TensorizeSet_dir, 'TN_records/TN_dataset') 
    # USE packed Dataloader
    train_save_dataloader_name = f'TN_{intrin_name}_train_and_val.pkl.DataLoader'
    # USE pkl
    train_save_pkl_name = f'TN_{intrin_name}_train_and_val.pkl'
    
    # eval_save_name = f'TN_{intrin_name}_test.pkl'
    
    table_save_path = os.path.join(TensorizeSet_dir, 'TN_records/TN_table') 
    table_save_name = f'TN_{intrin_name}_embedding_table.pkl'
    
    model_save_path = os.path.join(TensorizeSet_dir, f'TN_records/TN_models/{intrin_name}') 
    
    task_info_path = os.path.join(TensorizeSet_dir, 'task_info') 
    
    if args.save_folder == '':
        args.save_folder = model_save_path
    
    if os.path.exists(args.save_folder) is False:
        print('create folder', args.save_folder)
        os.makedirs(args.save_folder, exist_ok=True)
    
    # print('load data...')
    # if str(args.dataset).endswith(".pkl"):
    #     with open(args.dataset, 'rb') as f:
    #         datasets_global = pickle.load(f)
    #     print('load pkl done.')
    #     datas = load_datas(datasets_global)
    #     print('create dataloader done.')
    #     del datasets_global
    # elif str(args.dataset).endswith(".DataLoader"):
    #      with open(args.dataset, 'rb') as f:
    #         datas = pickle.load(f)
    # print('load data done.')
    
    print('load data...')
    if os.path.exists(os.path.join(dataset_save_path, train_save_dataloader_name)):
        print("using DataLoader ...")
        with open(os.path.join(dataset_save_path, train_save_dataloader_name), 'rb') as f:
            datas = pickle.load(f)
    elif os.path.exists(os.path.join(dataset_save_path, train_save_pkl_name)):
        print("using pkl ...")
        with open(os.path.join(dataset_save_path, train_save_pkl_name), 'rb') as f:
            datasets_global = pickle.load(f)
        print('load pkl done.')
        datas = load_datas(datasets_global)
        print('create dataloader done.')
        del datasets_global
    print('load data done.')
    
    train(*datas, device=args.cuda, is_recover=False)