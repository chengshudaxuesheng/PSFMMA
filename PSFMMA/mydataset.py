from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, eda_data, ppg_data, eeg_data, labels):
        self.eda_data = eda_data
        self.ppg_data = ppg_data
        self.eeg_data = eeg_data
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        eda_data = self.eda_data[idx]
        ppg_data = self.ppg_data[idx]
        eeg_data = self.eeg_data[idx]
        label = self.labels[idx]
        return eda_data, ppg_data, eeg_data, label
