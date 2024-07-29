import torch
import torch.nn as nn
import torch.nn.functional as F


class ZINBLoss(nn.Module):
    """ZINB loss class."""

    def __init__(self):
        super().__init__()

    def forward(self, x, mean, disp, pi, scale_factor, ridge_lambda=0.0):
        """Forward propagation.

        Parameters
        ----------
        x :
            input features.
        mean :
            data mean.
        disp :
            data dispersion.
        pi :
            data dropout probability.
        scale_factor : list
            scale factor of mean.
        ridge_lambda : float optional
            ridge parameter.

        Returns
        -------
        result : float
            ZINB loss.

        """
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor
        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge

        result = torch.mean(result)
        return result