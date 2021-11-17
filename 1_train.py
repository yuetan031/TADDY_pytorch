from codes.DynamicDatasetLoader import DynamicDatasetLoader
from codes.Component import MyConfig
from codes.DynADModel import DynADModel
from codes.Settings import Settings
import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['uci', 'digg', 'btc_alpha', 'btc_otc'], default='uci')
parser.add_argument('--anomaly_per', choices=[0.01, 0.05, 0.1], type=float, default=0.1)
parser.add_argument('--train_per', type=float, default=0.5)

parser.add_argument('--neighbor_num', type=int, default=5)
parser.add_argument('--window_size', type=int, default=2)

parser.add_argument('--embedding_dim', type=int, default=32)
parser.add_argument('--num_hidden_layers', type=int, default=2)
parser.add_argument('--num_attention_heads', type=int, default=2)

parser.add_argument('--max_epoch', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=5e-4)

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_feq', type=int, default=10)


args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

print('$$$$ Start $$$$')
data_obj = DynamicDatasetLoader()
data_obj.dataset_name = args.dataset
data_obj.k = args.neighbor_num
data_obj.window_size = args.window_size
data_obj.anomaly_per = args.anomaly_per
data_obj.train_per = args.train_per
data_obj.load_all_tag = False
data_obj.compute_s = True

my_config = MyConfig(k=args.neighbor_num, window_size=args.window_size, hidden_size=args.embedding_dim,
                     intermediate_size=args.embedding_dim, num_attention_heads=args.num_attention_heads,
                     num_hidden_layers=args.num_hidden_layers, weight_decay=args.weight_decay)

method_obj = DynADModel(my_config, args)
method_obj.spy_tag = True
method_obj.max_epoch = args.max_epoch
method_obj.lr = args.lr

setting_obj = Settings()

setting_obj.prepare(data_obj, method_obj)
setting_obj.run()

print('$$$$ Finish $$$$')