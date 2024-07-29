import torch


class Logger_detect(object):
    """ logger for ood detection task, reporting test auroc/aupr/fpr95 for ood detection """
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) % 3 == 2
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            ood_result, test_score, valid_loss = result[:, :-2], result[:, -2], result[:, -1]
            argmin = valid_loss.argmin().item()
            print(f'Run {run + 1:02d}:')
            print(f'Chosen epoch: {argmin + 1}')
            for k in range(result.shape[1] // 3):
                print(f'OOD Test {k+1} Final AUROC: {ood_result[argmin, k*3]:.2f}')
                print(f'OOD Test {k+1} Final AUPR: {ood_result[argmin, k*3+1]:.2f}')
                print(f'OOD Test {k+1} Final FPR95: {ood_result[argmin, k*3+2]:.2f}')
            print(f'IND Test Score: {test_score[argmin]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            ood_te_num = result.shape[2] // 3

            best_results = []
            for r in result:
                ood_result, test_score, valid_loss = r[:, :-2], r[:, -2], r[:, -1]
                score_val = test_score[valid_loss.argmin()].item()
                ood_result_val = []
                for k in range(ood_te_num):
                    auroc_val = ood_result[valid_loss.argmin(), k*3].item()
                    aupr_val = ood_result[valid_loss.argmin(), k*3+1].item()
                    fpr_val = ood_result[valid_loss.argmin(), k*3+2].item()
                    ood_result_val += [auroc_val, aupr_val, fpr_val]
                best_results.append(ood_result_val + [score_val])

            best_result = torch.tensor(best_results)

            if best_result.shape[0] == 1:
                print(f'All runs:')
                for k in range(ood_te_num):
                    r = best_result[:, k * 3]
                    print(f'OOD Test {k + 1} Final AUROC: {r.mean():.2f}')
                    r = best_result[:, k * 3 + 1]
                    print(f'OOD Test {k + 1} Final AUPR: {r.mean():.2f}')
                    r = best_result[:, k * 3 + 2]
                    print(f'OOD Test {k + 1} Final FPR: {r.mean():.2f}')
                r = best_result[:, -1]
                print(f'IND Test Score: {r.mean():.2f}')
            else:
                print(f'All runs:')
                for k in range(ood_te_num):
                    r = best_result[:, k*3]
                    print(f'OOD Test {k+1} Final AUROC: {r.mean():.2f} ± {r.std():.2f}')
                    r = best_result[:, k*3+1]
                    print(f'OOD Test {k+1} Final AUPR: {r.mean():.2f} ± {r.std():.2f}')
                    r = best_result[:, k*3+2]
                    print(f'OOD Test {k+1} Final FPR: {r.mean():.2f} ± {r.std():.2f}')
                r = best_result[:, -1]
                print(f'IND Test Score: {r.mean():.2f} ± {r.std():.2f}')

            return best_result
