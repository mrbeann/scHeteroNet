import torch
import scanpy as sc
import numpy as np
import pandas as pd
import anndata 
from sklearn import preprocessing
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from torch_geometric.data import Data
import time


dataset_prefix = '_processed'


class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name
        self.graph = {}
        self.label = None

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))

def build_graph(adata, radius=None, knears=None, distance_metrics='l2', use_repo='spatial'):
    """
    based on https://github.com/hannshu/st_datasets/blob/master/utils/preprocess.py
    """
    if (isinstance(adata, np.ndarray)):
        coor = pd.DataFrame(adata)
    elif ('X' == use_repo):
        coor = pd.DataFrame(adata.X.todense())
    else:
        coor = pd.DataFrame(adata.obsm[use_repo])
        coor.index = adata.obs.index
        coor.columns = ['row', 'col']

    if (radius):
        nbrs = NearestNeighbors(radius=radius, metric=distance_metrics).fit(coor)
        _, indices = nbrs.radius_neighbors(coor, return_distance=True)
    else:
        nbrs = NearestNeighbors(n_neighbors=knears+1, metric=distance_metrics).fit(coor)
        _, indices = nbrs.kneighbors(coor)

    edge_list = np.array([[i, j] for i, sublist in enumerate(indices) for j in sublist])
    return edge_list


def load_dataset_fixed(args, ignore_first=False, ood=False):
    dataset = NCDataset(args.dataset)
    ref_adata = anndata.read_h5ad(f'../data/processed_datasets/{args.dataset}{dataset_prefix}.h5ad')
    # encoding label to id
    le = preprocessing.LabelEncoder()
    y = ref_adata.obs['cell'].copy()
    X = ref_adata.X.copy()
    le.fit(y)
    y = torch.as_tensor(le.transform(y))
    features = torch.as_tensor(X)
    labels = y
    num_nodes = features.shape[0]

    dataset.graph = {'edge_index': None,
                     'edge_feat': None,
                     'node_feat': features,
                     'num_nodes': num_nodes}
    dataset.num_nodes = len(labels)
    dataset.label = torch.LongTensor(labels)
    if args.spatial:
        adj_knn = kneighbors_graph(dataset.graph['node_feat'], n_neighbors=args.knn_num, include_self=True)
        edge_index = torch.tensor(adj_knn.nonzero(), dtype=torch.long)
    else:
        edge_index = edge_index = build_graph(ref_adata, knears=args.knn_num)
        edge_index = torch.tensor(edge_index.T, dtype=torch.long)
    dataset.edge_index = dataset.graph['edge_index']=edge_index
    dataset.x = features
    # if ignore some class.  not fully match
    if ignore_first:
        dataset.label[dataset.label==0] = -1
    dataset.splits = {'train': ref_adata.uns['train_idxs'],
                      'valid': ref_adata.uns['val_idxs'],
                      'test': ref_adata.uns['test_idxs']}

    ref_adata.var['gene_name'] = ref_adata.var.index
    return dataset, ref_adata, le


def load_cell_graph_fixed(args):
    """
    given the dataset split data into: ID (train,val,test) and OOD.
    Since we do not use OOD train, we make it equal to OOD test
    """
    dataset, ref_adata, le = load_dataset_fixed(args, ood=True)
    dataset.y = dataset.label
    dataset.node_idx = torch.arange(dataset.num_nodes)
    dataset_ind = dataset  # in distribution dataset
    number_class = dataset.y.max().item() + 1
    print('number of classes', number_class)
    # class_t = number_class//2-1  # ood training classes
    dataset_ind_list, dataset_ood_tr_list, dataset_ood_te_list = [], [], []
    for run in range(args.runs):
        train_idx, val_idx, test_idx = ref_adata.uns['train_idxs'][str(run)], ref_adata.uns['val_idxs'][str(run)], ref_adata.uns['test_idxs'][str(run)]        
        ood_idx = ref_adata.uns["ood_idxs"][str(run)]
        id_idx = ref_adata.uns["id_idxs"][str(run)]
        
        dataset_ind.node_idx = id_idx
        dataset_ind.splits = {'train': train_idx,
                              'valid': val_idx,
                              'test': test_idx}
        dataset_ood_tr = Data(x=dataset.graph['node_feat'], 
                            edge_index=dataset.graph['edge_index'], y=dataset.y)
        dataset_ood_te = Data(x=dataset.graph['node_feat'], 
                            edge_index=dataset.graph['edge_index'], y=dataset.y)
        
        dataset_ood_tr.node_idx = dataset_ood_te.node_idx = ood_idx
        dataset_ind_list.append(dataset_ind)
        dataset_ood_tr_list.append(dataset_ood_tr)
        dataset_ood_te_list.append(dataset_ood_te)
    return dataset_ind_list, dataset_ood_tr_list, dataset_ood_te_list, ref_adata, le


if __name__ == '__main__':
    pass
