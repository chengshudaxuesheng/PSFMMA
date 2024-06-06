from torch import nn


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Map(nn.Module):
    def __init__(self):
        super(Map, self).__init__()
        self.aw = MLP(128, 128 * 2, 128)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.mlp = nn.Sequential(nn.Linear(1024, 2048), nn.Linear(2048, 1024))
        self.fc = nn.Linear(1024, 1024)

    def forward(self, x):
        resid = x.flatten(1)
        weight = self.aw(resid).softmax(1)
        x = resid * weight + resid
        x = x.reshape(-1, 1, 128)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool1(x)
        _, a, b = x.shape
        x = self.mlp(x.flatten(1))
        x = x.reshape(-1, a, b)
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool2(x)
        x = self.fc(x.flatten(1))
        return x



