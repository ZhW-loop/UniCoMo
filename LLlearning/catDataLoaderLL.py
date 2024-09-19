#NOTE
# for Life-Long learning make dataset

import torch, pickle, os
from torch import nn
from common import TensorizeSet_dir
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

def CatSegmentDataLoaderforTrain(loader1: SegmentDataLoader, loader2: SegmentDataLoader):
    print(f"loader1 number: {loader1.number}")
    print(f"loader2 number: {loader2.number}")
    
    if len(loader1.datas_steps) > 0:
        seq1 = len(loader1.datas_steps[0])
        ebd1 = len(loader1.datas_steps[0][0])
    else:
        seq1 = ebd1 = -1
    
    if len(loader2.datas_steps) > 0:
        seq2 = len(loader2.datas_steps[0])
        ebd2 = len(loader2.datas_steps[0][0])
    else:
        seq2 = ebd2 = -1
    
    if (seq1 != seq2 or ebd1 != ebd2) and seq1 > 0 and seq2 > 0 and ebd1 > 0 and ebd2 > 0:
        print("!!!padding!!!")
        seq = max(seq1, seq2)
        ebd = max(ebd1, ebd2)
        print(seq, ebd)
        loader1.datas_steps = torch.nn.functional.pad(loader1.datas_steps, (0, ebd - ebd1, 0, seq - seq1), "constant", 0)
        loader2.datas_steps = torch.nn.functional.pad(loader2.datas_steps, (0, ebd - ebd2, 0, seq - seq2), "constant", 0)
    
    loader1.datas_steps = torch.cat((loader1.datas_steps, loader2.datas_steps), dim=0)
    loader1.labels = torch.cat((loader1.labels, loader2.labels), dim=0)
    loader1.min_latency = torch.cat((loader1.min_latency, loader2.min_latency), dim=0)
    loader1.number = len(loader1.datas_steps)
    loader1.shuffle = True
    assert loader1.batch_size == loader2.batch_size
    loader1.batch_size = loader1.batch_size
    loader1.iter_order = loader1.pointer = None
    
    print(loader1.datas_steps.shape)
    return loader1
    
    

if __name__ == '__main__':
    TNortlp = "TN"
    new_task_id = 7
    LL_task_save_path = os.path.join(TensorizeSet_dir, f'LL_records/tasks_dataset/{TNortlp}') 
    LL_task_save_name = f"task{new_task_id}_train.pkl.DataLoader"
    # pkl_dataloader_paths = [f'/home/zhwang/TensorizeSet/{TNortlp}_records/{TNortlp}_dataset/{TNortlp}_tensorcore161616nn_train_and_val.pkl.DataLoader',
    #                         f'/home/zhwang/TensorizeSet/{TNortlp}_records/{TNortlp}_dataset/{TNortlp}_tensorcore161616nt_train_and_val.pkl.DataLoader',
    #                         f'/home/zhwang/TensorizeSet/{TNortlp}_records/{TNortlp}_dataset/{TNortlp}_tensorcore83216nn_train_and_val.pkl.DataLoader',
    #                         f'/home/zhwang/TensorizeSet/{TNortlp}_records/{TNortlp}_dataset/{TNortlp}_tensorcore83216nt_train_and_val.pkl.DataLoader',
    #                         f'/home/zhwang/TensorizeSet/{TNortlp}_records/{TNortlp}_dataset/{TNortlp}_tensorcore32816nn_train_and_val.pkl.DataLoader',
    #                         f'/home/zhwang/TensorizeSet/{TNortlp}_records/{TNortlp}_dataset/{TNortlp}_tensorcore32816nt_train_and_val.pkl.DataLoader']
    # pkl_dataloader_paths = [f'/home/zhwang/TensorizeSet/{TNortlp}_records/{TNortlp}_dataset/{TNortlp}_neon_train_and_val.pkl.DataLoader',
    #                         f'/home/zhwang/TensorizeSet/{TNortlp}_records/{TNortlp}_dataset/{TNortlp}_tensorcore161616nn_train_and_val.pkl.DataLoader',]
    
    pkl_dataloader_paths = [os.path.join(TensorizeSet_dir, f'{TNortlp}_records/{TNortlp}_dataset/{TNortlp}_tensorcore161616nn_train_and_val.pkl.DataLoader'),
                            os.path.join(TensorizeSet_dir, f'{TNortlp}_records/{TNortlp}_dataset/{TNortlp}_avx512_train_and_val.pkl.DataLoader',),
                            os.path.join(TensorizeSet_dir, f'{TNortlp}_records/{TNortlp}_dataset/{TNortlp}_sdot_train_and_val.pkl.DataLoader')]
    
    
    train_dataloaders = []

    for pkl_dataloader_path in pkl_dataloader_paths:
        pkl_dataloader_path : str
        assert pkl_dataloader_path.endswith(".pkl.DataLoader")
        with open(pkl_dataloader_path, 'rb') as f:
            train_dataloader, val_dataloader = pickle.load(f)
        train_dataloader: SegmentDataLoader
        val_dataloader: SegmentDataLoader
        train_dataloader = CatSegmentDataLoaderforTrain(train_dataloader, val_dataloader)
        train_dataloaders.append(train_dataloader)
    
    main_loader = train_dataloaders[0]
    
    for i in range(len(train_dataloaders)):
        if i == 0: continue
        main_loader = CatSegmentDataLoaderforTrain(main_loader, train_dataloaders[i])
    
    with open(os.path.join(LL_task_save_path, LL_task_save_name), 'wb') as f:
        pickle.dump(main_loader, f)