from torch import nn


class EEGChannelAttention(nn.Module):
    def __init__(self, channel):
        super(EEGChannelAttention, self).__init__()
        self.mpool = nn.AdaptiveMaxPool1d(1)
        self.apool = nn.AdaptiveAvgPool1d(1)
        self.ser = nn.Sequential(
            nn.Conv1d(channel, channel // 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channel // 2, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        resid = x
        mr = self.mpool(x)
        ar = self.apool(x)
        max_out = self.ser(mr)
        avg_out = self.ser(ar)
        result = self.sigmoid(max_out + avg_out)
        out = resid * result
        return out + resid


class Map_eeg(nn.Module):
    def __init__(self):
        super(Map_eeg, self).__init__()
        self.eegca = EEGChannelAttention(32)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 3), padding=(0, 1), stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), padding=(0, 1), stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2, 2), padding=(0, 0), stride=(2, 2))
        self.relu4 = nn.ReLU(inplace=True)
        self.pool3 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.relu6 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.mlp = nn.Sequential(nn.Linear(2048, 2048), nn.Linear(2048, 2048))
        self.fc = nn.Linear(512, 1024)

    def forward(self, x):
        x = self.eegca(x)
        x = x.contiguous().view(-1, 1, x.shape[1], x.shape[2])
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)

        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool3(x)

        _, a, b, c = x.shape
        x = self.mlp(x.flatten(1))
        x = x.reshape(-1, a, b, c)
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.pool4(x)

        x = self.fc(x.flatten(1))
        return x
