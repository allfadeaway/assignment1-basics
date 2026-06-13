import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max, _ = torch.max(x, dim = dim, keepdim = True)
    exp = torch.exp(x - max)   # avoiding situation like inf/inf
    return exp / torch.sum(exp, dim = dim, keepdim = True)

def cross_entropy(inputs, targets): # inputs(batch_size, max_seq_len, vocab_size)
    max, _ = torch.max(inputs, dim = -1, keepdim = True)
    rem = inputs - max
    log_softmax = torch.log(torch.sum(torch.exp(rem), dim = -1, keepdim = False))
    target_logits = torch.gather(rem, dim = -1, index = targets.unsqueeze(-1)).squeeze(-1)
    loss_per_token = log_softmax - target_logits
    return loss_per_token.mean()

def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    grads = [p.grad for p in parameters if p.grad is not None]
    
    total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))
    
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)
