import sys 
sys.path.append("TensorizeSet_app")
from TensorizeSet_app.pkl2DataLoader import SegmentDataLoader

sys.path.append("tensorize_notation")
from tensorize_notation.TN_train import LambdaRankLoss

from copy import deepcopy
import os, pickle
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data

device = "cuda:0"

# def variable(t: torch.Tensor, use_cuda=True, **kwargs):
#     if torch.cuda.is_available() and use_cuda:
#         t = t.cuda()
#     return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model: nn.Module, train_loader: SegmentDataLoader, loss_func):

        self.model = model
        self.loss_func = loss_func
        self.dataset = train_loader

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            # self._means[n] = variable(p.data)
            self._means[n] = p.data.to(device)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            # precision_matrices[n] = variable(p.data)
            precision_matrices[n] = p.data.to(device)

        self.model.eval()
        for batch_datas_steps, batch_labels in self.dataset:
            self.model.zero_grad()
            # batch_datas_steps = variable(batch_datas_steps)
            # batch_labels = variable(batch_labels)
            batch_datas_steps = batch_datas_steps.to(device)
            batch_labels = batch_labels.to(device)
            
            loss = self.loss_func(self.model(batch_datas_steps), batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss


def normal_train(TNortlp, task_id, epoch, model: nn.Module, optimizer: torch.optim, train_loader: SegmentDataLoader, loss_func, lr_scheduler: torch.optim.lr_scheduler.StepLR):
    model.to(device)
    model.train()
    epoch_loss = 0
    for batch_datas_steps, batch_labels in train_loader:
        batch_datas_steps = batch_datas_steps.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        loss = loss_func(model(batch_datas_steps), batch_labels)
        epoch_loss += loss.item()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    lr_scheduler.step()
    model_save_file_name = os.path.join("/home/zhwang/TensorizeSet/LL_records/normal", 
                                        TNortlp, 
                                        f"task{task_id}_models", 
                                        f"{TNortlp}_model_{epoch}.pkl")
    with open(model_save_file_name, 'wb') as f:
        pickle.dump(model.cpu(), f)
    print(f"{epoch}, epoch_loss:\t {epoch_loss}")
    return epoch_loss / len(train_loader)


def ewc_train(TNortlp, task_id, epoch, model: nn.Module, optimizer: torch.optim, train_loader: SegmentDataLoader, loss_func, lr_scheduler: torch.optim.lr_scheduler.StepLR,
              ewc: EWC, importance: float):
    model = model.to(device)
    model.train()
    epoch_loss = 0
    for batch_datas_steps, batch_labels in train_loader:
        batch_datas_steps = batch_datas_steps.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        loss = loss_func(model(batch_datas_steps), batch_labels) + importance * ewc.penalty(model)
        epoch_loss += loss.item()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameter)
        optimizer.step()
    lr_scheduler.step()
    model_save_file_name = os.path.join("/home/zhwang/TensorizeSet/LL_records/ewc", 
                                        TNortlp, 
                                        f"task{task_id}_models", 
                                        f"{TNortlp}_model_{epoch}.pkl")
    with open(model_save_file_name, 'wb') as f:
        pickle.dump(model.cpu(), f)
    return epoch_loss / len(train_loader)