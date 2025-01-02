import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import scipy.sparse


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index=None):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class HetConv(nn.Module):
    """ Neighborhood aggregation step """
    def __init__(self):
        super(HetConv, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, x, adj_t, adj_t2):
        x1 = matmul(adj_t, x)
        x2 = matmul(adj_t2, x)
        return torch.cat([x1, x2], dim=1)


class ZINBDecoder(nn.Module):
    """
    Parameters
    ----------
    input_dim : int
        dimension of input feature.
    n_z : int
        dimension of latent embedding.
    n_dec_1 : int optional
        number of nodes of decoder layer 1.
    n_dec_2 : int optional
        number of nodes of decoder layer 2.
    n_dec_3 : int optional
        number of nodes of decoder layer 3.

    """

    def __init__(self, input_dim, n_z, n_dec_1=128, n_dec_2=256, n_dec_3=512):
        super().__init__()
        self.n_dec_3 = n_dec_3
        self.input_dim = input_dim
        self.dec_1 = nn.Linear(n_z, n_dec_1)
        self.dec_2 = nn.Linear(n_dec_1, n_dec_3)
        self.dec_3 = nn.Linear(n_dec_2, n_dec_3)
        self.dec_mean = nn.Sequential(nn.Linear(self.n_dec_3, self.input_dim), MeanAct())
        self.dec_disp = nn.Sequential(nn.Linear(self.n_dec_3, self.input_dim), DispAct())
        self.dec_pi = nn.Sequential(nn.Linear(self.n_dec_3, self.input_dim), nn.Sigmoid())

    def forward(self, z):
        """Forward propagation.

        Parameters
        ----------
        z :
            embedding.

        Returns
        -------
        _mean :
            data mean from ZINB.
        _disp :
            data dispersion from ZINB.
        _pi :
            data dropout probability from ZINB.

        """
        dec_h1 = F.relu(self.dec_1(z))
        dec_h3 = F.relu(self.dec_2(dec_h1))
        # dec_h3 = F.relu(self.dec_3(dec_h2))

        _mean = self.dec_mean(dec_h3)
        _disp = self.dec_disp(dec_h3)
        _pi = self.dec_pi(dec_h3)
        return _mean, _disp, _pi


class MeanAct(nn.Module):
    """Mean activation class."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    """Dispersion activation class."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


class HeteroNet(nn.Module):
    """ our implementation with ZINB"""
    def __init__(self, in_channels, hidden_channels, out_channels, edge_index, num_nodes,
                    num_layers=2, dropout=0.5, save_mem=False, num_mlp_layers=1,
                    use_bn=True, conv_dropout=True, dec_dim=[]):
        super(HeteroNet, self).__init__()

        self.feature_embed = MLP(in_channels, hidden_channels,
                hidden_channels, num_layers=num_mlp_layers, dropout=dropout)


        self.convs = nn.ModuleList()
        self.convs.append(HetConv())

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*2*len(self.convs) ) )

        for l in range(num_layers - 1):
            self.convs.append(HetConv())
            if l != num_layers-2:
                self.bns.append(nn.BatchNorm1d(hidden_channels*2*len(self.convs) ) )

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.conv_dropout = conv_dropout # dropout neighborhood aggregation steps

        self.jump = JumpingKnowledge('cat')
        last_dim = hidden_channels*(2**(num_layers+1)-1)
        self.final_project = nn.Linear(last_dim, out_channels)

        self.num_nodes = num_nodes
        self.init_adj(edge_index)
        
        self.ZINB = ZINBDecoder(in_channels, last_dim, n_dec_1=dec_dim[0], n_dec_2=dec_dim[1], n_dec_3=dec_dim[2])

    def reset_parameters(self):
        self.feature_embed.reset_parameters()
        self.final_project.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def init_adj(self, edge_index):
        """ cache normalized adjacency and normalized strict two-hop adjacency,
        neither has self loops
        """
        n = self.num_nodes
        
        if isinstance(edge_index, SparseTensor):
            dev = edge_index.device
            adj_t = edge_index
            adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
            adj_t[adj_t > 0] = 1
            adj_t[adj_t < 0] = 0
            adj_t = SparseTensor.from_scipy(adj_t).to(dev)
        elif isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=None, sparse_sizes=(n, n))

        adj_t.remove_diag(0)
        adj_t2 = matmul(adj_t, adj_t)
        adj_t2.remove_diag(0)
        adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
        adj_t2 = scipy.sparse.csr_matrix(adj_t2.to_scipy())
        adj_t2 = adj_t2 - adj_t
        adj_t2[adj_t2 > 0] = 1
        adj_t2[adj_t2 < 0] = 0

        adj_t = SparseTensor.from_scipy(adj_t)
        adj_t2 = SparseTensor.from_scipy(adj_t2)
        
        adj_t = gcn_norm(adj_t, None, n, add_self_loops=False)
        adj_t2 = gcn_norm(adj_t2, None, n, add_self_loops=False)

        self.adj_t = adj_t.to(edge_index.device)
        self.adj_t2 = adj_t2.to(edge_index.device)

    def forward(self, x, edge_index, decoder=False):

        adj_t = self.adj_t
        adj_t2 = self.adj_t2
        
        x = self.feature_embed(x)
        x = self.activation(x)
        xs = [x]
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, adj_t2) 
            if self.use_bn:
                x = self.bns[i](x)
            xs.append(x)
            if self.conv_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, adj_t2)
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(x)
        x = self.jump(xs)
        if not self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        if decoder:
            _mean, _disp, _pi = self.ZINB(x)
            h = self.final_project(x)
            return h, _mean, _disp, _pi
        x = self.final_project(x)
        return x


if __name__ == '__main__':
    pass
