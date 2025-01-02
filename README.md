# scHeteroNet
 
## Requirement

- Scanpy 
- Pytorch 
- Numpy
- Pandas
- torch_geometric


## Usage

### 1. Setting environmernt
Setting the conda environment first.
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
You can use your own dataset in h5ad format. Or you can download the dataset from figureshare. The data should be placed in `./data/processed_datasets`


If you use your own dataset, you should firstly split the dataset into train, valid and test set. You can use the `split.py` to split the dataset.



### 3. Running the code
Run with 10x5cl dataset
```bash
python main.py --dataset 10x_5cl --epochs 200 --use_zinb --use_prop --use_2hop 
```

Run with spatial transcriptomics
```bash
python main.py --dataset dlpfc_151670 --epochs 200 --use_zinb --use_prop --use_2hop --spatial
```

Run with contrastive learning.
```bash
python main.py --dataset 10x_5cl --epochs 200 --use_zinb --use_prop --use_2hop --cl_weight 0.05
```

## Citation
