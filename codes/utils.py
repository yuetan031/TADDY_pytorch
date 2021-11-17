import numpy as np
import networkx as nx
import torch

# WL dict
def WL_setting_init(node_list, link_list):
    node_color_dict = {}
    node_neighbor_dict = {}

    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for pair in link_list:
        u1, u2 = pair
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1

    return node_color_dict, node_neighbor_dict

def compute_zero_WL(node_list, link_list):
    WL_dict = {}
    for i in node_list:
        WL_dict[i] = 0
    return WL_dict

# batching + hop + int + time
def compute_batch_hop(node_list, edges_all, num_snap, Ss, k=5, window_size=1):

    batch_hop_dicts = [None] * (window_size-1)
    s_ranking = [0] + list(range(k+1))

    Gs = []
    for snap in range(num_snap):
        G = nx.Graph()
        G.add_nodes_from(node_list)
        G.add_edges_from(edges_all[snap])
        Gs.append(G)

    for snap in range(window_size - 1, num_snap):
        batch_hop_dict = {}
        # S = Ss[snap]
        edges = edges_all[snap]

        # G = nx.Graph()
        # G.add_nodes_from(node_list)
        # G.add_edges_from(edges)

        for edge in edges:
            edge_idx = str(snap) + '_' + str(edge[0]) + '_' + str(edge[1])
            batch_hop_dict[edge_idx] = []
            for lookback in range(window_size):
                # s = np.array(Ss[snap-lookback][edge[0]] + Ss[snap-lookback][edge[1]].todense()).squeeze()
                s = Ss[snap - lookback][edge[0]] + Ss[snap - lookback][edge[1]]
                s[edge[0]] = -1000 # don't pick myself
                s[edge[1]] = -1000 # don't pick myself
                top_k_neighbor_index = s.argsort()[-k:][::-1]

                indexs = np.hstack((np.array([edge[0], edge[1]]), top_k_neighbor_index))

                for i, neighbor_index in enumerate(indexs):
                    try:
                        hop1 = nx.shortest_path_length(Gs[snap-lookback], source=edge[0], target=neighbor_index)
                    except:
                        hop1 = 99
                    try:
                        hop2 = nx.shortest_path_length(Gs[snap-lookback], source=edge[1], target=neighbor_index)
                    except:
                        hop2 = 99
                    hop = min(hop1, hop2)
                    batch_hop_dict[edge_idx].append((neighbor_index, s_ranking[i], hop, lookback))
        batch_hop_dicts.append(batch_hop_dict)

    return batch_hop_dicts

# Dict to embeddings
def dicts_to_embeddings(feats, batch_hop_dicts, wl_dict, num_snap, use_raw_feat=False):

    raw_embeddings = []
    wl_embeddings = []
    hop_embeddings = []
    int_embeddings = []
    time_embeddings = []

    for snap in range(num_snap):

        batch_hop_dict = batch_hop_dicts[snap]

        if batch_hop_dict is None:
            raw_embeddings.append(None)
            wl_embeddings.append(None)
            hop_embeddings.append(None)
            int_embeddings.append(None)
            time_embeddings.append(None)
            continue

        raw_features_list = []
        role_ids_list = []
        position_ids_list = []
        hop_ids_list = []
        time_ids_list = []

        for edge_idx in batch_hop_dict:

            neighbors_list = batch_hop_dict[edge_idx]
            edge = edge_idx.split('_')[1:]
            edge[0], edge[1] = int(edge[0]), int(edge[1])

            raw_features = []
            role_ids = []
            position_ids = []
            hop_ids = []
            time_ids = []

            for neighbor, intimacy_rank, hop, time in neighbors_list:
                if use_raw_feat:
                    raw_features.append(feats[snap-time][neighbor])
                else:
                    raw_features.append(None)
                role_ids.append(wl_dict[neighbor])
                hop_ids.append(hop)
                position_ids.append(intimacy_rank)
                time_ids.append(time)

            raw_features_list.append(raw_features)
            role_ids_list.append(role_ids)
            position_ids_list.append(position_ids)
            hop_ids_list.append(hop_ids)
            time_ids_list.append(time_ids)

        if use_raw_feat:
            raw_embedding = torch.FloatTensor(raw_features_list)
        else:
            raw_embedding = None
        wl_embedding = torch.LongTensor(role_ids_list)
        hop_embedding = torch.LongTensor(hop_ids_list)
        int_embedding = torch.LongTensor(position_ids_list)
        time_embedding = torch.LongTensor(time_ids_list)

        raw_embeddings.append(raw_embedding)
        wl_embeddings.append(wl_embedding)
        hop_embeddings.append(hop_embedding)
        int_embeddings.append(int_embedding)
        time_embeddings.append(time_embedding)

    return raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings
