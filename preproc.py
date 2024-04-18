from sklearn.preprocessing import minmax_scale


class Processor:
    def __init__(self):
        pass

    def prepare_data(self, img_path, gt_path):
        import scipy.io as sio
        img_mat = sio.loadmat(img_path)
        gt_mat = sio.loadmat(gt_path)
        img_keys = img_mat.keys()
        gt_keys = gt_mat.keys()
        img_key = [k for k in img_keys if k != '__version__' and k != '__header__' and k != '__globals__']
        gt_key = [k for k in gt_keys if k != '__version__' and k != '__header__' and k != '__globals__']
        img = img_mat.get(img_key[0]).astype('float64')
        gt = gt_mat.get(gt_key[0]).astype('int8')
        return img, gt

    def get_correct(self, img, gt):
        gt_1D = gt.reshape(-1)
        index = gt_1D.nonzero()
        gt_correct = gt_1D[index]
        img_2D = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
        img_correct = img_2D[index]
        return img_correct, gt_correct


if __name__ == '__main__':
    root = 'dataset/'
    im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
    img_path = root + im_ + '.mat'
    gt_path = root + gt_ + '.mat'
    p = Processor()
    img, gt = p.prepare_data(img_path, gt_path)

    n_row, n_column, n_band = img.shape
    X_img = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))
    img_correct, gt_correct = p.get_correct(X_img, gt)
    print(img_correct)
    print(gt_correct)