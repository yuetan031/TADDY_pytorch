import torch
import torch.nn.functional as F
import torch.optim as optim

from transformers.modeling_bert import BertPreTrainedModel
from codes.BaseModel import BaseModel

import time
import numpy as np

from sklearn import metrics
from codes.utils import dicts_to_embeddings, compute_batch_hop, compute_zero_WL


class DynADModel(BertPreTrainedModel):
    learning_record_dict = {}
    lr = 0.001
    weight_decay = 5e-4
    max_epoch = 500
    spy_tag = True

    load_pretrained_path = ''
    save_pretrained_path = ''

    def __init__(self, config, args):
        super(DynADModel, self).__init__(config, args)
        self.args = args
        self.config = config
        self.transformer = BaseModel(config)
        self.cls_y = torch.nn.Linear(config.hidden_size, 1)
        self.weight_decay = config.weight_decay
        self.init_weights()

    def forward(self, init_pos_ids, hop_dis_ids, time_dis_ids, idx=None):

        outputs = self.transformer(init_pos_ids, hop_dis_ids, time_dis_ids)

        sequence_output = 0
        for i in range(self.config.k+1):
            sequence_output += outputs[0][:,i,:]
        sequence_output /= float(self.config.k+1)

        output = self.cls_y(sequence_output)

        return output

    def batch_cut(self, idx_list):
        batch_list = []
        for i in range(0, len(idx_list), self.config.batch_size):
            batch_list.append(idx_list[i:i + self.config.batch_size])
        return batch_list

    def evaluate(self, trues, preds):
        aucs = {}
        for snap in range(len(self.data['snap_test'])):
            auc = metrics.roc_auc_score(trues[snap],preds[snap])
            aucs[snap] = auc

        trues_full = np.hstack(trues)
        preds_full = np.hstack(preds)
        auc_full = metrics.roc_auc_score(trues_full, preds_full)
        
        return aucs, auc_full

    def generate_embedding(self, edges):
        num_snap = len(edges)
        # WL_dict = compute_WL(self.data['idx'], np.vstack(edges[:7]))
        WL_dict = compute_zero_WL(self.data['idx'],  np.vstack(edges[:7]))
        batch_hop_dicts = compute_batch_hop(self.data['idx'], edges, num_snap, self.data['S'], self.config.k, self.config.window_size)
        raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = \
            dicts_to_embeddings(self.data['X'], batch_hop_dicts, WL_dict, num_snap)
        return raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings

    def negative_sampling(self, edges):
        negative_edges = []
        node_list = self.data['idx']
        num_node = node_list.shape[0]
        for snap_edge in edges:
            num_edge = snap_edge.shape[0]

            negative_edge = snap_edge.copy()
            fake_idx = np.random.choice(num_node, num_edge)
            fake_position = np.random.choice(2, num_edge).tolist()
            fake_idx = node_list[fake_idx]
            negative_edge[np.arange(num_edge), fake_position] = fake_idx

            negative_edges.append(negative_edge)
        return negative_edges

    def train_model(self, max_epoch):

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = self.generate_embedding(self.data['edges'])
        self.data['raw_embeddings'] = None

        ns_function = self.negative_sampling

        for epoch in range(max_epoch):
            t_epoch_begin = time.time()

            # -------------------------
            negatives = ns_function(self.data['edges'][:max(self.data['snap_train']) + 1])
            raw_embeddings_neg, wl_embeddings_neg, hop_embeddings_neg, int_embeddings_neg, \
            time_embeddings_neg = self.generate_embedding(negatives)
            self.train()

            loss_train = 0
            for snap in self.data['snap_train']:

                if wl_embeddings[snap] is None:
                    continue
                int_embedding_pos = int_embeddings[snap]
                hop_embedding_pos = hop_embeddings[snap]
                time_embedding_pos = time_embeddings[snap]
                y_pos = self.data['y'][snap].float()

                int_embedding_neg = int_embeddings_neg[snap]
                hop_embedding_neg = hop_embeddings_neg[snap]
                time_embedding_neg = time_embeddings_neg[snap]
                y_neg = torch.ones(int_embedding_neg.size()[0])

                int_embedding = torch.vstack((int_embedding_pos, int_embedding_neg))
                hop_embedding = torch.vstack((hop_embedding_pos, hop_embedding_neg))
                time_embedding = torch.vstack((time_embedding_pos, time_embedding_neg))
                y = torch.hstack((y_pos, y_neg))

                optimizer.zero_grad()

                output = self.forward(int_embedding, hop_embedding, time_embedding).squeeze()
                loss = F.binary_cross_entropy_with_logits(output, y)
                loss.backward()
                optimizer.step()

                loss_train += loss.detach().item()

            loss_train /= len(self.data['snap_train']) - self.config.window_size + 1
            print('Epoch: {}, loss:{:.4f}, Time: {:.4f}s'.format(epoch + 1, loss_train, time.time() - t_epoch_begin))

            if ((epoch + 1) % self.args.print_feq) == 0:
                self.eval()
                preds = []
                for snap in self.data['snap_test']:
                    int_embedding = int_embeddings[snap]
                    hop_embedding = hop_embeddings[snap]
                    time_embedding = time_embeddings[snap]

                    with torch.no_grad():
                        output = self.forward(int_embedding, hop_embedding, time_embedding, None)
                        output = torch.sigmoid(output)
                    pred = output.squeeze().numpy()
                    preds.append(pred)

                y_test = self.data['y'][min(self.data['snap_test']):max(self.data['snap_test'])+1]
                y_test = [y_snap.numpy() for y_snap in y_test]

                aucs, auc_full = self.evaluate(y_test, preds)

                for i in range(len(self.data['snap_test'])):
                    print("Snap: %02d | AUC: %.4f" % (self.data['snap_test'][i], aucs[i]))
                print('TOTAL AUC:{:.4f}'.format(auc_full))

    def run(self):
        self.train_model(self.max_epoch)
        return self.learning_record_dict