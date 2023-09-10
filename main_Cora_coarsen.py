#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 14:23:46 2023

@author: chris
"""

import datasets
import utility
import torch
import numpy as np
import models
import train_SIGN
import train_GCN
import writer
import argparse
import GraphCoarsening.utils as gc

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', nargs='?', const='QSIGN', type=str, default='QSIGN')
parser.add_argument('--coarsen_ratio', nargs='?', const=0.96, type=int, default=0.96) #0.96 corresponds to ~500 nodes
parser.add_argument('--max_hop', nargs='?', const=3, type=int, default=3)
parser.add_argument('--layers', nargs='?', const=3, type=int, default=3)
parser.add_argument('--hidden_channels', nargs='?', const=1536, type=int, default=1536)
parser.add_argument('--dropout', nargs='?', const=0.5, type=float, default=0.5)
parser.add_argument('--batch_norm', nargs='?', const=False, type=bool, default=False)
parser.add_argument('--lr', nargs='?', const=0.0005, type=float, default=0.0005)
parser.add_argument('--num_batch', nargs='?', const=5, type=int, default=5)
parser.add_argument('--num_epoch', nargs='?', const=400, type=int, default=400)
parser.add_argument('--multilabel', nargs='?', const=False, type=bool, default=False)
parser.add_argument('--do_eval', nargs='?', const=True, type=bool, default=True)
parser.add_argument('--residual', nargs='?', const=False, type=bool, default=False)
parser.add_argument('--print_result', nargs='?', const=True, type=bool, default=True)

args = parser.parse_args()
print(args)

model_type = args.model_type
coarsening_ratio = args.coarsen_ratio
max_hop = args.max_hop
layers = args.layers
hidden_channels = args.hidden_channels
dropout = args.dropout
batch_norm = args.batch_norm
lr = args.lr
num_batch = args.num_batch
num_epoch = args.num_epoch
multilabel = args.multilabel
do_evaluation = args.do_eval
residual = args.residual
print_result = args.print_result

dataset_name = 'Cora'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Prepare Dataset
# get dataset
(x, y, edge_index, train_mask, val_mask, test_mask) = datasets.get_Planetoid(name="Cora")
num_features = x.shape[1]
num_classes = max(y) + 1

# construct networkx graph
G = utility.construct_graph(x, y, edge_index, train_mask, val_mask, 
                                test_mask)

#%% split graph to get uncontracted train, val, and test graphs
(x_train, y_train, edge_train, train_mask, x_val, y_val, edge_val, val_mask, x_test, 
 y_test, edge_test, test_mask) = utility.split_graph(G, multilabel = multilabel)

# get initial label distribution
Y_dist_before = utility.get_label_distribution_tensor(y_train[train_mask], multilabel)

#%% coarsening train graph (all training nodes)
edge_train = edge_train.T
edge_train = torch.tensor(edge_train)
edge_train_flip = torch.flip(edge_train,[0]) # re-adds flipped edges that were removed by networkx
edge_train = torch.cat((edge_train, edge_train_flip), 1)
x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train)
train_mask = torch.tensor(train_mask)

candidate, C_list, Gc_list = gc.coarsening(edge_train, coarsening_ratio, coarsening_method='variation_neighborhoods')

(x_train, y_train, train_mask, _, _, edge_train) = gc.load_data(train_mask, ~train_mask, 
                                                       x_train, y_train, 
                                                       num_classes, candidate, 
                                                       C_list, Gc_list)

# converting training tensors back to networkx to remove duplicate edges
edge_train = edge_train.numpy()
edge_train = edge_train.T
G = utility.construct_graph(x_train, y_train, edge_train, train_mask, ~train_mask, 
                            ~train_mask)

#%% convert train networkx graph to tensor
x_train, y_train, edge_train, train_mask, _, _ = utility.convert_graph_to_tensor(G, multilabel)

# get final label distribution
Y_dist_after = utility.get_label_distribution_tensor(y_train[train_mask], multilabel)

# get label distribution errors
Y_dist_error = utility.compute_label_distribution_error(Y_dist_before, Y_dist_after)
print(f"Contraction Label Distribution Avg Error: {np.mean(Y_dist_error)}")

#%% Normalize Adjacency Matrices
adj_train = utility.construct_normalized_adj(edge_train, x_train.shape[0])
adj_val = utility.construct_normalized_adj(edge_val, x_val.shape[0])
adj_test = utility.construct_normalized_adj(edge_test, x_test.shape[0])

#%% Convert feature and labels to torch tensor
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train)
train_mask = torch.tensor(train_mask)
x_val = torch.tensor(x_val, dtype=torch.float32)
y_val = torch.tensor(y_val)
val_mask = torch.tensor(val_mask)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test)
test_mask = torch.tensor(test_mask)

#%% Preparing Model
if (model_type == 'SIGN' or model_type == 'QSIGN'):
    # precomputing SIGN aggregation
    print("Pre-aggregating embeddings ...")
    x_train_aggregated = models.precompute_SIGN_aggregation(x_train, adj_train, 
                                                            max_hop)
    x_val_aggregated = models.precompute_SIGN_aggregation(x_val, adj_val, 
                                                            max_hop)
    x_test_aggregated = models.precompute_SIGN_aggregation(x_test, adj_test, 
                                                            max_hop)
    print(f"Pre-aggregated embedding size: {x_train_aggregated.shape[1]}")
    
    # in SIGN, after aggregation phase, data points are decoupled, we only need
    # to pass labeled data
    x_train_aggregated = x_train_aggregated[train_mask]
    y_train = y_train[train_mask]
    x_val_aggregated = x_val_aggregated[val_mask]
    y_val = y_val[val_mask]
    x_test_aggregated = x_test_aggregated[test_mask]
    y_test = y_test[test_mask]
    
    if model_type == 'QSIGN':
        model = models.QMLP(hidden_channels=hidden_channels, num_layers= layers, 
                            in_channels= x_train_aggregated.shape[1],
                            out_channels=num_classes, batch_norm=batch_norm, 
                            dropout=dropout)
    elif model_type == 'SIGN':
        model = models.MLP(hidden_channels=hidden_channels, num_layers= layers, 
                            in_channels= x_train_aggregated.shape[1],
                            out_channels=num_classes, batch_norm=batch_norm, 
                            dropout=dropout)
        
elif (model_type == 'GCN'):
    model = models.GCN(hidden_channels=hidden_channels, num_layers= layers, 
                        in_channels= x_train.shape[1], out_channels=num_classes, 
                        batch_norm=batch_norm, dropout=dropout, residual=residual)
    
print(model)

#%% Training
if model_type == 'SIGN' or model_type == 'QSIGN':
    if do_evaluation:
        (max_val_acc, max_val_f1, max_val_sens, max_val_spec, max_val_test_acc, 
         max_val_test_f1, max_val_test_sens, max_val_test_spec, session_memory, 
         train_memory, train_time_avg) = train_SIGN.train(model, device, 
                                                          x_train=x_train_aggregated, 
                                                          y_train=y_train, 
                                                          x_val=x_val_aggregated, 
                                                          y_val=y_val, 
                                                          x_test=x_test_aggregated, 
                                                          y_test=y_test, 
                                                          multilabel=multilabel, 
                                                          lr=lr, num_batch=num_batch, 
                                                          num_epoch=num_epoch)
    else:
        (max_val_acc, max_val_f1, max_val_sens, max_val_spec, max_val_test_acc, 
         max_val_test_f1, max_val_test_sens, max_val_test_spec, session_memory, 
         train_memory, train_time_avg) = train_SIGN.train(model, device, 
                                                          x_train=x_train_aggregated, 
                                                          y_train=y_train, 
                                                          x_val=None, 
                                                          y_val=None, 
                                                          x_test=None, 
                                                          y_test=None, 
                                                          multilabel=multilabel, 
                                                          lr=lr, num_batch=num_batch, 
                                                          num_epoch=num_epoch)

elif model_type == 'GCN':
    if do_evaluation:
        (max_val_acc, max_val_f1, max_val_sens, max_val_spec, max_val_test_acc, 
         max_val_test_f1, max_val_test_sens, max_val_test_spec, session_memory, 
         train_memory, train_time_avg) = train_GCN.train(model, device, 
                                                          x_train=x_train, 
                                                          y_train=y_train, 
                                                          adj_train=adj_train,
                                                          train_mask=train_mask,
                                                          x_val=x_val, 
                                                          y_val=y_val, 
                                                          adj_val=adj_val,
                                                          val_mask=val_mask,
                                                          x_test=x_test, 
                                                          y_test=y_test, 
                                                          adj_test=adj_test,
                                                          test_mask=test_mask,
                                                          multilabel=multilabel, 
                                                          lr=lr, num_epoch=num_epoch)
    else:
        (max_val_acc, max_val_f1, max_val_sens, max_val_spec, max_val_test_acc, 
         max_val_test_f1, max_val_test_sens, max_val_test_spec, session_memory, 
         train_memory, train_time_avg) = train_GCN.train(model, device, 
                                                          x_train=x_train, 
                                                          y_train=y_train, 
                                                          adj_train=adj_train,
                                                          train_mask=train_mask,
                                                          x_val=None, 
                                                          y_val=None, 
                                                          adj_val=None,
                                                          val_mask=None,
                                                          x_test=None, 
                                                          y_test=None, 
                                                          adj_test=None,
                                                          test_mask=None,
                                                          multilabel=multilabel, 
                                                          lr=lr, num_epoch=num_epoch)
    
#%% Printing Result
if print_result:
    writer.write_result(dataset_name, model_type, "Coarsening", x_train.shape[0], 
                        max_val_acc, max_val_f1, max_val_sens, max_val_spec, 
                        max_val_test_acc, max_val_test_f1, max_val_test_sens, 
                        max_val_test_spec, session_memory, train_memory, 
                        train_time_avg)
    
    writer.write_label_dist_error(dataset_name, "Coarsening", x_train.shape[0], 
                                  0, Y_dist_error)