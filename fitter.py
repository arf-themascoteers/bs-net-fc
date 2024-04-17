from preproc import Processor
from bs_net_fc import BSNetFC
import torch
from dset import DSet
from torch.utils.data import DataLoader
import numpy as np


class Fitter:
    def __init__(self, X, img=None, gt=None):
        self.model = BSNetFC()
        self.batch_size = 64
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00002)
        dset = DSet(X, X)
        dataloader = DataLoader(dset, batch_size=self.batch_size, shuffle=True)
        criterion = torch.nn.MSELoss(reduction='mean')
        for epoch in range(100):
            for x,y in dataloader:
                optimizer.zero_grad()
                channel_weights, y_hat = self.model(x)
                loss = criterion(y_hat, y)
                if img is not None:
                    mean_weight = np.mean(channel_weights, axis=0)
                    band_indx = np.argsort(mean_weight)[::-1][:self.n_selected_band]
                    print('=============================')
                    print('SELECTED BAND: ', band_indx)
                    print('=============================')
                    x_new = img[:, :, band_indx]
                    n_row, n_clm, n_band = x_new.shape
                    img_ = minmax_scale(x_new.reshape((n_row * n_clm, n_band))).reshape((n_row, n_clm, n_band))
                    p = Processor()
                    img_correct, gt_correct = p.get_correct(img_, gt)
                    score = eval_band_cv(img_correct, gt_correct, times=20, test_size=0.95)
                    print('acc=', score)
                    score_list.append(score)
                if i_epoch % 10 == 0:
                    np.savez('history-FC.npz', loss=loss_history, score=score_list, channel_weight=channel_weight_list)
            np.savez('history-FC.npz', loss=loss_history, score=score_list, channel_weight=channel_weight_list)
            saver.save(sess, './IndianPine-model-FC.ckpt')

        def