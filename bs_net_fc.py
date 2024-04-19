import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score


class BSNetFC(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(3)
        self.n_channel = 200
        self.weighter = nn.Sequential(
            nn.BatchNorm1d(200),
            nn.Linear(200, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.channel_weight_layer = nn.Sequential(
            nn.Linear(128, 200),
            nn.Sigmoid()
        )
        self.encoder = nn.Sequential(
            nn.Linear(200, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 200),
            nn.BatchNorm1d(200),
            nn.Sigmoid()
        )

    def forward(self, X):
        channel_weights = self.weighter(X)
        channel_weights = self.channel_weight_layer(channel_weights)
        channel_weights_ = torch.reshape(channel_weights, (-1, self.n_channel))
        reweight_out = X * channel_weights_
        output = self.encoder(reweight_out)
        return channel_weights, output

    def get_l1_loss(self):
        l1_reg = torch.tensor(0., requires_grad=True)
        for param in self.channel_weight_layer.parameters():
            l1_reg = l1_reg + torch.norm(param, p=1)
        return l1_reg*0.01


if __name__ == "__main__":
    f = BSNetFC()
    num_params = sum(p.numel() for p in f.parameters() if p.requires_grad)
    print("Number of learnable parameters:", num_params)
    t = torch.randn(160000, 200)
    x,y = f(t)
    print(x,y)



