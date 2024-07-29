import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import scipy
from utils.logger import Logger_detect
from utils.evaluate import evaluate_detect, eval_acc
from utils.dataset import load_cell_graph_fixed
from utils.parse import parser_add_main_args
from utils.scHeteroNet import scHeteroNet
from utils.losses import ZINBLoss
import warnings
warnings.filterwarnings("ignore")


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

dataset_ind_list, dataset_ood_tr_list, dataset_ood_te_list, adata, le = load_cell_graph_fixed(args)
eval_func = eval_acc
logger = Logger_detect(args.runs, args)


for run in range(args.runs):
    dataset_ind, dataset_ood_tr, dataset_ood_te = dataset_ind_list[run], dataset_ood_tr_list[run], dataset_ood_te_list[run]
    if len(dataset_ind.y.shape) == 1:
        dataset_ind.y = dataset_ind.y.unsqueeze(1)
    if len(dataset_ood_tr.y.shape) == 1:
        dataset_ood_tr.y = dataset_ood_tr.y.unsqueeze(1)
    if isinstance(dataset_ood_te, list):
        for data in dataset_ood_te:
            if len(data.y.shape) == 1:
                data.y = data.y.unsqueeze(1)
    else:
        if len(dataset_ood_te.y.shape) == 1:
            dataset_ood_te.y = dataset_ood_te.y.unsqueeze(1)

    c = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])
    d = dataset_ind.graph['node_feat'].shape[1]
    model = scHeteroNet(d, c, dataset_ind.edge_index.to(device), dataset_ind.num_nodes, args).to(device)

    criterion = nn.NLLLoss()
    model.train()

    model.reset_parameters()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float('-inf')
    min_loss = 100000
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss, _mean, _disp, _pi, train_idx  = model.loss_compute(dataset_ind, dataset_ood_tr, criterion, device, args)
        if args.use_zinb:
            zinb_loss = ZINBLoss().to(device)
            x_raw = adata.raw.X
            if scipy.sparse.issparse(x_raw):
                x_raw = x_raw.toarray()
            x_raw =  torch.Tensor(x_raw)[train_idx].to(device)
            zinb_loss = zinb_loss(x_raw, _mean, _disp, _pi, torch.tensor(adata.obs.size_factors)[train_idx].to(device))
            loss += args.zinb_weight * zinb_loss 
        loss.backward()
        optimizer.step()
        
        result, test_ind_score, test_ood_score, representations = evaluate_detect(model, dataset_ind, dataset_ood_te, criterion, eval_func, args, device, return_score=True)
        logger.add_result(run, result)
        if result[-1] < min_loss:
            min_loss = result[-1]
            selected_run = run
            selected_representation = representations.copy()
            selected_full_preds = selected_representation.argmax(axis=-1)
            selected_full_preds = le.inverse_transform(selected_full_preds)
            selected_test_ind_score = test_ind_score.numpy().copy()
            selected_test_ood_score = test_ood_score.numpy().copy()
        
        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'AUROC: {100 * result[0]:.2f}%, '
                    f'AUPR: {100 * result[1]:.2f}%, '
                    f'FPR95: {100 * result[2]:.2f}%, '
                    f'Test Score: {100 * result[-2]:.2f}%')
    logger.print_statistics(run)

results = logger.print_statistics()
