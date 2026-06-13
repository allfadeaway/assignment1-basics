from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr = 1e-3):
        if lr < 0:
            raise ValueError(f"Inbalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss
    
def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate + (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (max_learning_rate - min_learning_rate) / 2
    else:
        return min_learning_rate

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr = 1e-3, weight_decay = 0.01, betas = (0.9, 0.95), eps = 1e-8):
        if lr < 0:
            raise ValueError(f"Inbalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "dr": weight_decay, "eps": eps}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            dr = group["dr"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))

                grad = p.grad.data
                lrt = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= lr * dr * p.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad
                p.data -= lrt * m / (torch.sqrt(v) + eps)

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss