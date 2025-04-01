#-*- coding : utf-8-*-
# coding:unicode_escape

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from utils import sparse_mx_to_torch_sparse_tensor
from dataset import load_data
from sklearn.cluster import KMeans
from evaluation import eva
from time import time
import torch.nn.functional as F

def drop_feature(x):
    x = torch.Tensor(x)
    drop_prob=0.2
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=256):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        z_end = beta * z
        return (beta * z).sum(1), beta

        # return z_end, beta


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.in_features = in_ft
        self.out_features = out_ft
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()
        self.attention = Attention(out_ft)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)

        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else :
            if seq_fts.shape[0]==8:
                out = torch.bmm(adj, seq_fts)
            else:
                out = torch.mm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        out ,__ = self.attention(out)
        return self.act(out)

class GCN_2(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN_2, self).__init__()
        self.in_features = in_ft
        self.out_features = out_ft
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.fc2 = nn.Linear(out_ft, 256, bias=False)
        self.act = nn.PReLU()
        self.attention = Attention(out_ft)
        self.bias2 = nn.Parameter(torch.FloatTensor(256))
        self.bias2.data.fill_(0.0)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if seq_fts.shape[0] == 4:
            out = torch.bmm(adj, seq_fts)
        else:
            out = torch.mm(adj, seq_fts)

        if self.bias is not None:
            out += self.bias
        # out ,__ = self.attention(out)

        seq_fts2 = self.fc2(out)
        if seq_fts2.shape[0] == 4:
            out2 = torch.bmm(adj, seq_fts2)
        else:
            out2 = torch.mm(adj, seq_fts2)

        if self.bias2 is not None:
            out2 += self.bias2
        return self.act(out2)

class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)
        self.attention = Attention(out_features)


    def forward(self, features, adj, active=False):
        support = torch.mm(features, self.weight)
        output = torch.spmm(torch.Tensor(adj), support)

        if active:
            output, __ = self.attention(output)
        output = F.relu(output)
        return output

class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.median(seq, 1).values
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.mean(seq * msk, 1) / torch.sum(msk)

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c1, c2, h1, h2, h3, h4):
        c_x1 = torch.unsqueeze(c1, 1)
        c_x1 = c_x1.expand_as(h1).contiguous()
        c_x2 = torch.unsqueeze(c2, 1)
        c_x2 = c_x2.expand_as(h2).contiguous()

        # positive
        sc_1 = torch.squeeze(self.f_k(h2, c_x1), 2)
        sc_2 = torch.squeeze(self.f_k(h1, c_x2), 2)

        # negetive
        sc_3 = torch.squeeze(self.f_k(h4, c_x1), 2)
        sc_4 = torch.squeeze(self.f_k(h3, c_x2), 2)

        # logits = torch.cat((sc_1, sc_3), 1)
        logits = torch.cat((sc_1, sc_2, sc_3, sc_4), 1)
        return logits

class Contrast_2(nn.Module):
    def __init__(self, n_in,n_h):
        super(Contrast_2, self).__init__()
        self.gcn1 = GCN_2(n_in, n_h)
        self.gcn2 = GCN_2(n_in, n_h)
        self.read = Readout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(256)

    def forward(self, seq1, seq2, adj, diff, sparse, msk):
        h_1 = self.gcn1(seq1, adj, sparse)
        c_1 = self.read(h_1, msk)
        c_1 = self.sigm(c_1)

        h_2 = self.gcn2(seq1, diff, sparse)
        c_2 = self.read(h_2, msk)
        c_2 = self.sigm(c_2)

        h_3 = self.gcn1(seq2, adj, sparse)
        h_4 = self.gcn2(seq2, diff, sparse)
        ret = self.disc(c_1, c_2, h_1, h_2, h_3, h_4)

        return ret, h_1+h_2
    def embed(self, x, adj, diff, sparse):
        z1 = self.gcn1(x, adj, sparse)
        d1 = self.gcn2(x, diff, sparse)
        z = z1+d1

        return z

class Model(nn.Module):
    def __init__(self, n_in, n_h,n_cluster, view_Num):
        super(Model, self).__init__()

        self.contrast = Contrast_2(n_in,n_h)
        self.gnn1 = GNNLayer(n_in,256)
        self.gnn2 = GNNLayer(n_in,256)
        self.gnn3 = GNNLayer(n_in,256)
        self.gnn4 = GNNLayer(256,n_cluster)

        self.pred_w = nn.Parameter(torch.Tensor(view_Num, 256, 256))
        torch.nn.init.xavier_normal_(self.pred_w.data)
        self.cluster_layer = nn.Parameter(torch.Tensor(n_cluster,256))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.pred_w1 = nn.Parameter(torch.Tensor(256,256))
        torch.nn.init.xavier_normal_(self.pred_w1.data)
        self.pred_w2 = nn.Parameter(torch.Tensor(256,256))
        torch.nn.init.xavier_normal_(self.pred_w2.data)
        self.pred_w3 = nn.Parameter(torch.Tensor(256,256))
        torch.nn.init.xavier_normal_(self.pred_w3.data)
        self.attention = Attention(256)

    def forward(self, b_feat, b_feat_drop, b_adj, b_diff, sparse, feat, feat_flitter, adj_all,diff_all,view_Num,view_adj):
        logits, _ = self.contrast(b_feat, b_feat_drop, b_adj, b_diff, sparse, None)
        embeds_from_contrast = self.contrast.embed(feat, adj_all, diff_all,sparse)
        A_pred_view = []
        for i in range(view_Num):
            x = embeds_from_contrast.t()
            tmp = torch.matmul(embeds_from_contrast, self.pred_w[i])
            pred = torch.sigmoid(torch.matmul(tmp, x))
            A_pred_view.append(pred)
        A_pred = A_pred_view
        view_1 = self.gnn1(feat_flitter,view_adj[0])
        view_2 = self.gnn2(feat_flitter,view_adj[1])
        view_3 = self.gnn3(feat_flitter,view_adj[2])
        view_mix  = torch.matmul(view_1,self.pred_w1)*torch.matmul(view_2,self.pred_w2)*torch.matmul(view_3,self.pred_w3)
        view_mix_p = self.gnn4(view_mix, adj_all,active= False)
        predict = F.softmax(view_mix_p, dim=1)
        q = self.get_Q(embeds_from_contrast)

        return logits, A_pred, predict,q,embeds_from_contrast,view_mix_p




    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / 1.0)
        q = q.pow((1.0 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def parameter_print(nb_epochs, patience, lr, l2_coef, hid_units, sparse):
    print("nb_epochs is:", nb_epochs)
    print("patience is:", patience)
    print("lr is:", lr)
    print("hid_units:", hid_units)


def train(dataset, verbose=True):

    nb_epochs = 3000
    lr = 0.001
    l2_coef = 0.0
    hid_units = 512
    sparse = False
    view_adj,adj, diff, features, features_flitter, labels, adj_oral = load_data(dataset)

    # ×ª»»³ÉtensorÓÃ
    adj_t = torch.Tensor(adj)
    diff_t = torch.Tensor(diff)
    feat_t = torch.Tensor(features)
    feat_ft = torch.Tensor(features_flitter)


    feat_drop_t = drop_feature(features)

    feat_drop = feat_drop_t.numpy()


    y = labels
    view_Num = view_adj.shape[0]

    adj_label = adj_oral
    adj_label = torch.Tensor(adj_label)


    ft_size = features.shape[1]
    nb_classes = np.unique(labels).shape[0]

    sample_size = 2000
    batch_size = 4

    lbl_1 = torch.ones(batch_size, sample_size * 2)
    lbl_2 = torch.zeros(batch_size, sample_size * 2)

    lbl = torch.cat((lbl_1, lbl_2), 1)

    model = Model(ft_size, hid_units,nb_classes, view_Num)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    b_xent = nn.BCEWithLogitsLoss()

    for epoch in range(nb_epochs):

        idx = np.random.randint(0, adj.shape[-1] - sample_size + 1, batch_size)
        ba, bd, bf, bfl = [], [], [], []
        for i in idx:
            ba.append(adj[i: i + sample_size, i: i + sample_size])
            bd.append(diff[i: i + sample_size, i: i + sample_size])
            bf.append(features[i: i + sample_size])
            bfl.append(feat_drop[i: i + sample_size])

        ba = np.array(ba).reshape(batch_size, sample_size, sample_size)
        bd = np.array(bd).reshape(batch_size, sample_size, sample_size)
        bf = np.array(bf).reshape(batch_size, sample_size, ft_size)
        bfl = np.array(bfl).reshape(batch_size, sample_size, ft_size)


        if sparse:
            ba = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ba))
            bd = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(bd))
        else:
            ba = torch.FloatTensor(ba)
            bd = torch.FloatTensor(bd)

        bf = torch.FloatTensor(bf)
        bfl = torch.FloatTensor(bfl)

        model.train()
        optimiser.zero_grad()

        if epoch%3==0:
            logits, A_pred_from_con_emb, predict_from_gnn, tmp_q_from_con,emb_from_con,emb_from_gnn = model(bf, bfl, ba, bd, sparse, feat_t,feat_ft,adj_t, diff_t,view_Num,view_adj)
            tmp_q_from_c = tmp_q_from_con.detach()
            p = target_distribution(tmp_q_from_c)
        logits, A_pred, predict, q,emb_c,emb_gnn= model(bf, bfl, ba, bd, sparse, feat_t, feat_ft,  adj_t, diff_t,view_Num,view_adj)
        kmeans = KMeans(n_clusters=nb_classes, n_init=20)
        y_pred2 = kmeans.fit_predict(emb_c.data.cpu().numpy())
        acc, nmi, ari, f1 = eva(y, y_pred2, epoch)
        loss1 = b_xent(logits, lbl)
        loss2 = 0
        for i in range(view_Num):
            loss2 += (F.binary_cross_entropy(A_pred[i].view(-1), adj_label[i].view(-1)))
        loss3 = F.kl_div(predict.log(), p,reduction='batchmean')
        loss4 = F.kl_div(q.log(), p,reduction='batchmean')
        print('Epoch: {0}, ACC :{1:0.4f}, NMI :{2:0.4f}, ARI :{3:0.4f}, F1 :{4:0.4f}'.format(epoch, acc, nmi, ari, f1))

        loss = 0.1 * loss1 +  loss2 + 0.01*loss3 + 0.1*loss4

        loss.backward()
        optimiser.step()


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    t = time()
    dataset = 'DBLP'
    for train_num in range(5):
        print("train_num:", train_num)
        train(dataset)
    print('time used: %d s' % (time() - t))