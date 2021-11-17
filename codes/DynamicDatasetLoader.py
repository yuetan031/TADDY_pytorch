from codes.base_class.dataset import dataset
import torch
import numpy as np
import scipy.sparse as sp
from numpy.linalg import inv
import pickle
import os


class DynamicDatasetLoader(dataset):
    c = 0.15
    k = 5
    eps = 0.001
    window_size = 1
    data = None
    batch_size = None
    dataset_name = None
    load_all_tag = False
    compute_s = False
    anomaly_per = 0.1
    train_per = 0.5

    def __init__(self, seed=None, dName=None, dDescription=None):
        super(DynamicDatasetLoader, self).__init__(dName, dDescription)

    def load_hop_wl_batch(self):  #load the "raw" WL/Hop/Batch dict
        print('Load WL Dictionary')
        f = open('./result/WL/' + self.dataset_name, 'rb')
        wl_dict = pickle.load(f)
        f.close()

        print('Load Hop Distance Dictionary')
        f = open('./result/Hop/hop_' + self.dataset_name + '_' + str(self.k) + '_' + str(self.window_size), 'rb')
        hop_dict = pickle.load(f)
        f.close()

        print('Load Subgraph Batches')
        f = open('./result/Batch/' + self.dataset_name + '_' + str(self.k) + '_' + str(self.window_size), 'rb')
        batch_dict = pickle.load(f)
        f.close()

        return hop_dict, wl_dict, batch_dict

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix. (0226)"""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def adj_normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def accuracy(self, output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot

    def sparse_to_tuple(self, sparse_mx):
        """Convert sparse matrix to tuple representation. (0226)"""

        def to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return coords, values, shape

        if isinstance(sparse_mx, list):
            for i in range(len(sparse_mx)):
                sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
            sparse_mx = to_tuple(sparse_mx)

        return sparse_mx

    def preprocess_adj(self, adj):
        """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation. (0226)"""
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # adj_np = np.array(adj.todense())
        adj_normalized = self.normalize_adj(adj + sp.eye(adj.shape[0]))
        adj_normalized = self.sparse_mx_to_torch_sparse_tensor(adj_normalized)
        return adj_normalized

    def get_adjs(self, rows, cols, weights, nb_nodes):

        eigen_file_name = 'data/eigen/' + self.dataset_name + '_' + str(self.train_per) + '_' + str(self.anomaly_per) + '.pkl'
        if not os.path.exists(eigen_file_name):
            generate_eigen = True
            print('Generating eigen as: ' + eigen_file_name)
        else:
            generate_eigen = False
            print('Loading eigen from: ' + eigen_file_name)
            with open(eigen_file_name, 'rb') as f:
                eigen_adjs_sparse = pickle.load(f)
            eigen_adjs = []
            for eigen_adj_sparse in eigen_adjs_sparse:
                eigen_adjs.append(np.array(eigen_adj_sparse.todense()))

        adjs = []
        if generate_eigen:
            eigen_adjs = []
            eigen_adjs_sparse = []

        for i in range(len(rows)):
            adj = sp.csr_matrix((weights[i], (rows[i], cols[i])), shape=(nb_nodes, nb_nodes), dtype=np.float32)
            adjs.append(self.preprocess_adj(adj))
            if self.compute_s:
                if generate_eigen:
                    eigen_adj = self.c * inv((sp.eye(adj.shape[0]) - (1 - self.c) * self.adj_normalize(adj)).toarray())
                    for p in range(adj.shape[0]):
                        eigen_adj[p,p] = 0.
                    eigen_adj = self.normalize(eigen_adj)
                    eigen_adjs.append(eigen_adj)
                    eigen_adjs_sparse.append(sp.csr_matrix(eigen_adj))

            else:
                eigen_adjs.append(None)

        if generate_eigen:
            with open(eigen_file_name, 'wb') as f:
                pickle.dump(eigen_adjs_sparse, f, pickle.HIGHEST_PROTOCOL)

        return adjs, eigen_adjs

    def load(self):
        """Load dynamic network dataset"""

        print('Loading {} dataset...'.format(self.dataset_name))
        with open('data/percent/' + self.dataset_name + '_' + str(self.train_per) + '_' + str(self.anomaly_per) + '.pkl', 'rb') as f:
            rows, cols, labels, weights, headtail, train_size, test_size, nb_nodes, nb_edges = pickle.load(f)

        degrees = np.array([len(x) for x in headtail])
        num_snap = test_size + train_size

        edges = [np.vstack((rows[i], cols[i])).T for i in range(num_snap)]
        adjs, eigen_adjs = self.get_adjs(rows, cols, weights, nb_nodes)

        labels = [torch.LongTensor(label) for label in labels]

        snap_train = list(range(num_snap))[:train_size]
        snap_test = list(range(num_snap))[train_size:]

        idx = list(range(nb_nodes))
        index_id_map = {i:i for i in idx}
        idx = np.array(idx)

        return {'X': None, 'A': adjs, 'S': eigen_adjs, 'index_id_map': index_id_map, 'edges': edges,
                'y': labels, 'idx': idx, 'snap_train': snap_train, 'degrees': degrees,
                'snap_test': snap_test, 'num_snap': num_snap}
