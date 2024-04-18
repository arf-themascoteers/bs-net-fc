from preproc import Processor
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import maxabs_scale
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def eval_band(new_img, gt, train_inx, test_idx):
    p = Processor()
    # img_, gt_ = p.get_correct(new_img, gt)
    gt_ = gt
    img_ = maxabs_scale(new_img)
    # X_train, X_test, y_train, y_test = train_test_split(img_, gt_, test_size=0.4, random_state=42)
    X_train, X_test, y_train, y_test = img_[train_inx], img_[test_idx], gt_[train_inx], gt_[test_idx]
    knn_classifier = KNN(n_neighbors=5)
    knn_classifier.fit(X_train, y_train)
    # score = cross_val_score(knn_classifier, img_, y=gt_, cv=3)
    y_pre = knn_classifier.predict(X_test)
    score = accuracy_score(y_test, y_pre)
    # score = np.mean(score)
    return score


def eval_band_cv(X, y, times=10, test_size=0.95):
    p = Processor()
    estimator = [KNN(n_neighbors=3), SVC(C=1e5, kernel='rbf', gamma=1.)]
    estimator_pre, y_test_all = [[], []], []
    for i in range(times):  # repeat N times K-fold CV
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=None, shuffle=True, stratify=y)
        y_test_all.append(y_test)
        for c in range(len(estimator)):
            estimator[c].fit(X_train, y_train)
            y_pre = estimator[c].predict(X_test)
            estimator_pre[c].append(y_pre)
    score_dic = {'knn':{'ca':[], 'oa':[], 'aa':[], 'kappa':[]},
                 'svm': {'ca': [], 'oa': [], 'aa': [], 'kappa': []}
                 }
    key_ = ['knn', 'svm']
    for z in range(len(estimator)):
        ca, oa, aa, kappa = p.save_res_4kfolds_cv(estimator_pre[z], y_test_all, file_name=None, verbose=False)
        score_dic[key_[z]]['ca'] = ca
        score_dic[key_[z]]['oa'] = oa
        score_dic[key_[z]]['aa'] = aa
        score_dic[key_[z]]['kappa'] = kappa
    return score_dic


def eval_band_rf(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95)
    metric_evaluator = RandomForestClassifier()
    metric_evaluator.fit(X_train, y_train)
    y_pred = metric_evaluator.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    return score