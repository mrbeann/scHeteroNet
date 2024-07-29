import scanpy as sc
from sklearn.model_selection import train_test_split
import numpy as np
import anndata
import random
SAVE_PATH = "./data/processed_datasets"


def normalize_adata(adata, size_factors=True, normalize_input=True, logtrans_input=True):
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata, min_counts=0)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    
    if logtrans_input:
        sc.pp.log1p(adata)
    
    if normalize_input:
        sc.pp.scale(adata)
    
    return adata

def filter_cellType(adata):
    adata_copied = adata.copy()
    cellType_Number = adata_copied.obs.cell.value_counts()
    celltype_to_remove = cellType_Number[cellType_Number <= 10].index
    adata_copied = adata_copied[~adata_copied.obs.cell.isin(celltype_to_remove), :]

    return adata_copied

def filter_data(X, highly_genes=4000):
    X = np.ceil(X).astype(int)
    adata = sc.AnnData(X, dtype=np.float32)
    sc.pp.filter_genes(adata, min_counts=3)
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=4, flavor='cell_ranger', min_disp=0.5,
                                n_top_genes=highly_genes, subset=True)
    genes_idx = np.array(adata.var_names.tolist()).astype(int)
    cells_idx = np.array(adata.obs_names.tolist()).astype(int)

    return genes_idx, cells_idx


def get_genename(raw_adata):
    if "gene_id" in raw_adata.var.keys():
        gene_name = raw_adata.var["gene_id"].values
    elif "symbol" in raw_adata.var.keys():
        gene_name = raw_adata.var["symbol"].values
    else:
        gene_name = raw_adata.var.index
    return gene_name


def gen_split(dataset, normalize_input=False, logtrans_input=True):     # 
    raw_adata = anndata.read_h5ad('./data/'+dataset+'.h5ad')
    raw_adata.obs['cell'] = raw_adata.obs['cell_type']
    # delete cell_type column
    if 'cell_type' in raw_adata.obs.keys():
        del raw_adata.obs['cell_type']
    raw_adata = raw_adata[raw_adata.obs['assay'] == '10x 3\' v2']
    print("filtering cells whose cell type number is less than 10")
    raw_adata = filter_cellType(raw_adata)
    # fileter and normalize (bio processing)
    X = raw_adata.X.toarray()
    y = raw_adata.obs["cell"]
    genes_idx, cells_idx = filter_data(X)
    X = X[cells_idx][:, genes_idx]
    y = y[cells_idx]
    adata = sc.AnnData(X, dtype=np.float32)
    # sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    # sc.pp.log1p(adata)
    adata = normalize_adata(adata, size_factors=True, normalize_input=normalize_input, logtrans_input=logtrans_input)
    for obs in raw_adata.obs.keys():
        if obs in ["cell"]:
            adata.obs[obs + "type_raw"] = raw_adata.obs[obs].values[cells_idx]
        print("copy", obs)
        adata.obs[obs] = raw_adata.obs[obs].values[cells_idx]
    adata.obs["cell"] = y.values
    adata.var["gene_name"] = get_genename(raw_adata)[genes_idx]
    
    train_idxs = []
    val_idxs = []
    test_idxs = []
    ood_idxs = []
    id_idxs = []
    for seed in [42, 66, 88, 2023, 2024]:
        ood_class = y.value_counts().idxmin()
        ood_idx = [i for i, value in enumerate(y) if value == ood_class]
        id_idx = [i for i, value in enumerate(y) if value != ood_class]
        full_indices = np.arange(adata.shape[0])
        train_idx, test_idx = train_test_split(full_indices, test_size=0.2, random_state=seed)
        train_val_indices = train_idx
        train_idx, val_idx = train_test_split(train_val_indices, test_size=0.25, random_state=seed)
        train_idx = [i for i in train_idx if i not in ood_idx]
        val_idx = [i for i in val_idx if i not in ood_idx]
        test_idx = [i for i in test_idx if i not in ood_idx]
        train_idxs.append(train_idx)
        val_idxs.append(val_idx)
        test_idxs.append(test_idx)
        ood_idxs.append(ood_idx)
        id_idxs.append(id_idx)
    adata.uns["train_idxs"] = {str(key): value for key, value in enumerate(train_idxs)}
    adata.uns["val_idxs"] = {str(key): value for key, value in enumerate(val_idxs)}
    adata.uns["test_idxs"] = {str(key): value for key, value in enumerate(test_idxs)}
    adata.uns["ood_idxs"] = {str(key): value for key, value in enumerate(ood_idxs)}
    adata.uns["id_idxs"] = {str(key): value for key, value in enumerate(id_idxs)}
    print(adata)
    adata.write(f"{SAVE_PATH}/{dataset}_processed.h5ad")


if __name__ == '__main__':
    gen_split('10x5cl')
