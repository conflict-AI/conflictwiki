
import torch

def weighted_binary_cross_entropy(self, output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2

        output = torch.clamp(output, min=1e-2, max=1)
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
    return torch.neg(torch.mean(loss))

