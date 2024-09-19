import pickle, numpy as np, torch
from torch import nn
from typing import Dict, List, NamedTuple, Optional, Tuple
from ..cost_model import PyCostModel
from ..feature_extractor import FeatureExtractor, PerBlockFeature
from ..utils import derived_object
from ..tune_context import TuneContext
from ..runner import RunnerResult
from ..search_strategy import MeasureCandidate




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


@derived_object
class UniCost(PyCostModel):
    """Segment Sum MLP Model

    Parameters
    ----------
    trainer: SegmentSumMLPTrainer
        The trainer for the model, handling the training interface.
    """

    def __init__(
        self,
        path,
        *,
        seqlen = 27,
        embsize = 179
    ):
        super().__init__()
        self.seqlen = seqlen
        self.embsize = embsize
        self.extractor = FeatureExtractor.create('per-block-feature')
        self.load(path)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            self.model = pickle.load(f).to('cuda:0')
        self.model.eval()

    def save(self, path: str) -> None:
        pass

    def update(
        self,
        context: TuneContext,
        candidates: List[MeasureCandidate],
        results: List[RunnerResult],
    ) -> None:
        pass

    def predict(self, context: TuneContext, candidates: List[MeasureCandidate]) -> np.ndarray:
        scores = self.extractor.extract_from(context, candidates)
        xs=[ x.numpy().astype("double").tolist() for x in scores ]
        datas = []
        for x in xs:
            datas.append((x, 1, 1))
        test_loader = SegmentDataLoader(datas, 4000, False)
        preds_all = []
        labels_all = []

        for batch_datas_steps, batch_labels in test_loader:
            batch_datas_steps = batch_datas_steps.to('cuda:0')
            preds = self.model(batch_datas_steps)
            preds_all.append(preds.detach().cpu())
            labels_all.append(batch_labels.detach().cpu())

        preds_all = torch.cat(preds_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)
        return preds_all.detach().cpu().numpy()
