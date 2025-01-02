import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
import numpy as np
from utils.backbone import HeteroNet


class scHeteroNet(nn.Module):
    def __init__(self, d, c, edge_index, num_nodes, args):
        super(scHeteroNet, self).__init__()
        self.encoder =  HeteroNet(d, args.hidden_channels, c, edge_index=edge_index,
                        num_nodes=num_nodes,
                        num_layers=args.num_layers, dropout=args.dropout, use_bn=args.use_bn,
                        dec_dim=[32, 64, 128])

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        '''return predicted logits'''
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def propagation(self, e, edge_index, prop_layers=1, alpha=0.5):
        '''energy belief propagation, return the energy after propagation'''
        e = e.unsqueeze(1)
        N = e.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm = 1. / d[col]
        value = torch.ones_like(row) * d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))  # normolized adjacency matrix
        for _ in range(prop_layers):  # iterative propagation
            e = e * alpha + matmul(adj, e) * (1 - alpha)
        return e.squeeze(1)
    
    def two_hop_propagation(self, e, edge_index, prop_layers=1, alpha=0.5):
        e = e.unsqueeze(1)
        N = e.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm = 1. / d[col]
        value = torch.ones_like(row) * d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))  # normalized adjacency matrix

        # Compute the two-hop adjacency matrix
        adj_2hop = adj @ adj

        for _ in range(prop_layers):  # iterative propagation
            e = e * alpha + matmul(adj_2hop, e) * (1 - alpha)
        return e.squeeze(1)

    def detect(self, dataset, node_idx, device, args):
        '''return negative energy, a vector for all input nodes'''
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits = self.encoder(x, edge_index)
        if args.dataset in ('proteins', 'ppi'): # for multi-label binary classification
            logits = torch.stack([logits, torch.zeros_like(logits)], dim=2)
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1).sum(dim=1)
        else: # for single-label multi-class classification
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1)
        if args.use_prop: # use energy belief propagation
            if args.use_2hop:
                neg_energy = self.two_hop_propagation(neg_energy, edge_index, args.oodprop, args.oodalpha)
            else:
                neg_energy = self.propagation(neg_energy, edge_index, args.oodprop, args.oodalpha)
        return neg_energy[node_idx]

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):
        '''return loss for training'''
        x_in, edge_index_in = dataset_ind.x.to(device), dataset_ind.edge_index.to(device)
        x_out, edge_index_out = dataset_ood.x.to(device), dataset_ood.edge_index.to(device)
        train_in_idx, train_ood_idx = dataset_ind.splits['train'], dataset_ood.node_idx
        logits_in, _mean, _disp, _pi = [i[train_in_idx] for i in self.encoder(dataset_ind.x.to(device), dataset_ind.edge_index.to(device), decoder=args.use_zinb)]
        logits_out = self.encoder(x_out, edge_index_out)
        # compute supervised training loss
        pred_in = F.log_softmax(logits_in, dim=1)
        sup_loss = criterion(pred_in, dataset_ind.y[train_in_idx].squeeze(1).to(device))    
        loss = sup_loss

        return loss, _mean, _disp, _pi, train_in_idx, logits_in
