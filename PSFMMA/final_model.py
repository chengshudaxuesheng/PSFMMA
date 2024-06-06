from torch import nn
import sys, os
import torch

sys.path.append(os.getcwd())

from t_embedding import TEmbedding
from map import Map
from map_eeg import Map_eeg
from ModalEnconder import ModalEnconder


class FinalModel(nn.Module):
    def __init__(self, device):
        super(FinalModel, self).__init__()
        self.eda_te = TEmbedding(1, 1, 1, device)
        self.ppg_te = TEmbedding(1, 1, 1, device)
        self.eeg_te = TEmbedding(32, 32, 1, device)
        self.eda_map = Map()
        self.ppg_map = Map()
        self.eeg_map = Map_eeg()
        self.ModalEnconder = ModalEnconder()

    def forward(self, eda, ppg, eeg):
        eda = self.eda_te(eda)
        ppg = self.ppg_te(ppg)
        eeg = self.eeg_te(eeg)
        eda = torch.permute(eda, (0, 2, 1))
        ppg = torch.permute(ppg, (0, 2, 1))
        eeg = torch.permute(eeg, (0, 2, 1))
        eda = self.eda_map(eda)
        ppg = self.ppg_map(ppg)
        eeg = self.eeg_map(eeg)
        sub = torch.stack((eda, ppg, eeg), 1)
        res = self.ModalEnconder(sub, None)
        return res

