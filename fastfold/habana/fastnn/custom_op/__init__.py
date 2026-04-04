import torch
import torch.nn.functional as F


def fused_softmax(logits, mask, dim):
    """Drop-in replacement for the R1.7.1 custom TPC fused_softmax kernel."""
    return F.softmax(logits + mask, dim=dim)


def fused_softmax_bias(logits, mask, bias, dim):
    """Drop-in replacement for the R1.7.1 custom TPC fused_softmax_bias kernel."""
    return F.softmax(logits + mask + bias, dim=dim)
