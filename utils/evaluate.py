import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level= 0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))
    if np.array_equal(classes, [1]):
        return thresholds[cutoff]  # return threshold

    return fps[cutoff] / (np.sum(np.logical_not(y_true))), thresholds[cutoff]


def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = roc_auc_score(labels, examples)
    aupr = average_precision_score(labels, examples)
    fpr, threshould = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr, threshould


def eval_f1(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_pred.shape == y_true.shape:
        y_pred = y_pred.detach().cpu().numpy()
    else:
        y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        f1 = f1_score(y_true, y_pred, average='micro')
        acc_list.append(f1)

    return sum(acc_list)/len(acc_list)


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_pred.shape == y_true.shape:
        y_pred = y_pred.detach().cpu().numpy()
    else:
        y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)


# @torch.no_grad()  # this seems quite important, the orignal impl donot set this.
def evaluate_detect(model, dataset_ind, dataset_ood, criterion, eval_func, args, device, return_score=False):
    model.eval()

    with torch.no_grad():
        test_ind_score = model.detect(dataset_ind, dataset_ind.splits['test'], device, args).cpu()
    if isinstance(dataset_ood, list):
        result = []
        for d in dataset_ood:
            with torch.no_grad():
                test_ood_score = model.detect(d, d.node_idx, device, args).cpu()
            auroc, aupr, fpr, _ = get_measures(test_ind_score, test_ood_score)
            result += [auroc] + [aupr] + [fpr]
    else:
        with torch.no_grad():
            test_ood_score = model.detect(dataset_ood, dataset_ood.node_idx, device, args).cpu()
        # print(test_ind_score, test_ood_score)
        auroc, aupr, fpr, _ = get_measures(test_ind_score, test_ood_score)
        result = [auroc] + [aupr] + [fpr]

    with torch.no_grad(): 
        out = model(dataset_ind, device).cpu()
        test_idx = dataset_ind.splits['test']
        test_score = eval_func(dataset_ind.y[test_idx], out[test_idx])

        valid_idx = dataset_ind.splits['valid']
        if args.dataset in ('proteins', 'ppi'):
            valid_loss = criterion(out[valid_idx], dataset_ind.y[valid_idx].to(torch.float))
        else:
            valid_out = F.log_softmax(out[valid_idx], dim=1)
            valid_loss = criterion(valid_out, dataset_ind.y[valid_idx].squeeze(1))

        result += [test_score] + [valid_loss]

        if return_score:
            return result, test_ind_score, test_ood_score, out.detach().cpu().numpy()
        else:
            return result

