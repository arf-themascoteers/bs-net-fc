from preproc import Processor
from bs_net_fc import BSNetFC
import torch
from dset import DSet
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import minmax_scale
from utility import eval_band_rf


class Fitter:
    def __init__(self, X, n_selected_band, img=None, gt=None):
        self.n_selected_band = n_selected_band
        self.img = img
        self.gt = gt
        self.model = BSNetFC()
        self.batch_size = 64
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00002)
        self.dset = DSet(X, X)
        self.dataloader = DataLoader(self.dset, batch_size=self.batch_size, shuffle=True)
        self.criterion = torch.nn.MSELoss(reduction='sum')

    def fit(self):
        channel_weights = torch.ones((1000,200), dtype=torch.float32)
        for epoch in range(100):
            for batch, (x,y) in enumerate(self.dataloader):
                if self.img is not None:
                    mean_weight = torch.mean(channel_weights, dim=0)
                    band_indx = torch.argsort(mean_weight, descending=True)[:self.n_selected_band]
                    print(f"Selected bands: {band_indx}")
                    x_new = self.img[:, :, band_indx]
                    n_row, n_clm, n_band = x_new.shape
                    img_ = minmax_scale(x_new.reshape((n_row * n_clm, n_band))).reshape((n_row, n_clm, n_band))
                    p = Processor()
                    img_correct, gt_correct = p.get_correct(img_, gt)
                    score = eval_band_rf(img_correct, gt_correct)
                    print('acc=', score)
                channel_weights, y_hat = self.model(x)
                mse_loss = self.criterion(y_hat, y)
                l1_loss = self.model.get_l1_loss()
                loss = mse_loss + l1_loss
                print(f"Epoch={epoch} Batch={batch} - MSE={round(mse_loss.item(),5)}, L1={round(l1_loss.item(),5)}, LOSS={round(loss.item(),5)}")
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()


if __name__ == '__main__':
    root = 'dataset/'
    im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
    img_path = root + im_ + '.mat'
    gt_path = root + gt_ + '.mat'
    p = Processor()
    img, gt = p.prepare_data(img_path, gt_path)

    n_row, n_column, n_band = img.shape
    X_img = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))
    X_train, y_gt = p.get_correct(X_img, gt)
    print('training img shape: ', X_train.shape)
    fitter = Fitter(X_train, 5, img=X_img, gt=gt)
    fitter.fit()
