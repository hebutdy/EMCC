# -*- coding : utf-8-*-
# coding:unicode_escape

from utils import preprocess_features, normalize_adj
from sklearn.preprocessing import  normalize
from utils import compute_ppr
import scipy.sparse as sp
import scipy.io as sio
import numpy as np
import os
from scipy.linalg import fractional_matrix_power, inv


def load_data(dataset):

    print(dataset)
    if dataset == "ACM":
        dataset_path = "./data_multi/ACM3025.mat"
    elif dataset == "DBLP":
        dataset_path = "./data_multi/DBLP4057_GAT_with_idx.mat"
    elif dataset == "IMDB":
        dataset_path = "./data_multi/imdb5k.mat"

    data = sio.loadmat(dataset_path)
    datadir = os.path.join('data_multi', dataset)
    if not os.path.exists(datadir):
        print("Preprocess the data...")
        if dataset == "DBLP":
            labels, features = data['label'], data['features'].astype(float)
            labels = np.array(preprocess_labels(labels))
            view_adj = []
            # view_adj_with_eye = np.array([(data['PAP']).tolist(), (data['PLP']).tolist()])  #ACM
            view_adj_with_eye = np.array([(data['net_APA']).tolist(), (data['net_APCPA']).tolist(), (data['net_APTPA']).tolist()]) #DBLP
            # view_adj_with_eye = np.array([(data['MAM']).tolist(), (data['MDM']).tolist()])  #IMDB
            for v in range(view_adj_with_eye.shape[0]):
                adj_tmp = view_adj_with_eye[v]
                adj_tmp = adj_tmp - np.eye(view_adj_with_eye.shape[1])
                view_adj.append(adj_tmp)
            view_adj = np.array(view_adj)
            view_diff = []
            for i in range(view_adj.shape[0]):
                diff_tmp = compute_ppr(view_adj[i], 0.2)
                view_diff.append(diff_tmp)
            view_diff = np.array(view_diff)
            diff = view_diff.sum(0)
            adj_all = view_adj.sum(0)
            adj_all = preprocess_adj_all(adj_all)
            adj_oral = view_adj
            # features_flitter = np.array(Flitter(adj_all + np.eye(adj_all.shape[0]), features))
            adj1 = normalize_adj(adj_all + np.eye(adj_all.shape[0])).todense()
            adj = normalize(view_adj_with_eye.sum(0), norm="l1")
            os.makedirs(datadir)
            np.save(f'{datadir}/view_adj.npy', view_adj)
            np.save(f'{datadir}/adj.npy', adj1)
            np.save(f'{datadir}/diff.npy', diff)
            np.save(f'{datadir}/feat.npy', features)
            # np.save(f'{datadir}/feat_fl.npy', features_flitter)
            np.save(f'{datadir}/labels.npy', labels)
            np.save(f'{datadir}/adj_oral.npy', adj_oral)
            print('Done!')
    else:
        print("loading resource from files")
        view_adj = np.load(f'{datadir}/view_adj.npy')
        adj = np.load(f'{datadir}/adj.npy')
        diff = np.load(f'{datadir}/diff.npy')
        features = np.load(f'{datadir}/feat.npy')
        # features_flitter = np.load(f'{datadir}/feat_fl.npy')
        labels = np.load(f'{datadir}/labels.npy')
        adj_oral = np.load(f'{datadir}/adj_oral.npy')

    return view_adj, adj, diff, features, labels, adj_oral
    # return view_adj, adj, diff, features, features_flitter, labels, adj_oral



def Flitter(adj, feat):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    D = d_mat_inv_sqrt
    A = adj.dot(D).transpose().dot(D).tocoo()
    Ls = D - A
    I = np.eye(adj.shape[0])
    for i in range(2):
        feat = (I - 0.5 * Ls).dot(feat)
    return feat


def compute_ppr(adj_all, alpha=0.2, self_loop=True):
    print("alpha is:", alpha)
    a = adj_all
    if self_loop:
        a = a + np.eye(a.shape[0])
    d = np.diag(np.sum(a, 1))
    dinv = fractional_matrix_power(d, -0.5)
    at = np.matmul(np.matmul(dinv, a), dinv)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))


def normalize_adj(adj, self_loop=False):
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def adj_check(adj):
    pos = np.unravel_index(np.argmax(adj), adj.shape)
    max = adj[pos]
    return max


def preprocess_labels(labels):
    real_label = np.argmax(labels, axis=1)
    return real_label

def preprocess_adj_all(adj):
    N = adj.shape[0]
    adj_new = []
    for i in adj:
        for j in i:
            if j == 0:
                adj_new.append(0)
            else:
                adj_new.append(1)
    adj_new = np.array(adj_new).reshape(N, N)
    return adj_new


if __name__ == '__main__':
    load_data('DBLP')


