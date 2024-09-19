#NOTE
# only for train and val pkl
import os, time, pickle, random, numpy as np, argparse, glob
import torch, sys
from torch import nn

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

        print(f"self.datas_steps.shape:\t {self.datas_steps.shape}")
        print(f"self.labels.shape:\t {self.labels.shape}")
        print(f"self.min_latency:\t {self.min_latency.shape}")
        
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
    data_cnt = -1
    if data_cnt > 0:
        # train_len = int(data_cnt * 1000 * 0.9)
        train_len = int(data_cnt * 1000 * 1.)
        perm = np.random.permutation(len(datasets))
        train_indices, val_indices = perm[:train_len], perm[train_len: data_cnt * 1000]
    else:
        # train_len = int(len(datasets) * 0.9)
        train_len = int(len(datasets) * 1.)
        perm = np.random.permutation(len(datasets))
        train_indices, val_indices = perm[:train_len], perm[train_len:]

    train_datas, val_datas = datasets[train_indices], datasets[val_indices]

    # n_gpu = torch.cuda.device_count()
    n_gpu = 1
    #NOTE hyper parameters
    val_size_per_gpu = train_size_per_gpu = 1024
    train_dataloader = SegmentDataLoader(
        train_datas, train_size_per_gpu*n_gpu, True)
    val_dataloader = SegmentDataLoader(
        val_datas, val_size_per_gpu*n_gpu, False)

    return train_dataloader, val_dataloader


if __name__ == "__main__":
    print(sys.argv[0])
    pkl_name = sys.argv[1]
    del sys.argv[1]
    TNortlp = pkl_name.split("_")[0]
    pkl_path = f"/home/zhwang/TensorizeSet/{TNortlp}_records/{TNortlp}_dataset/{pkl_name}"
    assert pkl_path.endswith("train_and_val.pkl")
    dataloader_path = pkl_path + ".DataLoader"
    print(pkl_path)
    print(dataloader_path)
    
    # if os.path.exists(dataloader_path):
    #     print(f"dataloader_path already exits, skipping ...")
    #     sys.exit()
    
    print('load data...')
    with open(pkl_path, 'rb') as f:
        datasets_global = pickle.load(f)
    print('load pkl done.')
    train_dataloader, val_dataloader = load_datas(datasets_global)
    print('create dataloader done.')
    del datasets_global
    print("del datasets_global done")
    with open(dataloader_path, 'wb') as f:
        pickle.dump((train_dataloader, val_dataloader), f)