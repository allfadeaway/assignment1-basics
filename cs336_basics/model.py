import torch
import torch.nn as nn
import math

from nn_utils import softmax

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        std = math.sqrt(2.0 / (in_features + out_features))
        self.weight = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(out_features, in_features, device=device, dtype=dtype),
                mean = 0, std = std, a = -3 * std, b = 3 * std
            )
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T # (..., d_in) @ (d_in, d_out) -> (..., d_out)

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.embeddings = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype), 
                mean = 0, std = 1, a = -3, b = 3
            )
        )
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d = d_model
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor: # x: (batch_size, sequence_length, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.sum(x ** 2, dim = -1, keepdim = True) / self.d + self.eps)
        result = x / rms * self.g

        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # x: (batch_size, sequence_length, d_model) 
        tmp = self.W1(x)
        return self.W2((tmp * torch.sigmoid(tmp)) * self.W3(x))

def scaled_dot_product_attention(Q, K, V, mask) -> torch.Tensor: # Q,K,V(batch_size, ..., seq_len, d_k), mask(seq_len, seq_len)
    d_k = Q.shape[-1]
    attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k) # attn(batch_size, ..., seq_len, seq_len)
    masked_attn = attn.masked_fill(mask == False, -torch.inf) 
    softmax_attn = softmax(masked_attn, dim = -1)
    return torch.matmul(softmax_attn, V)                          # (..., seq_len, d_k)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        k = torch.arange(0, d_k, 2, device=device)
        tmp = 1.0 / (theta ** (k / d_k))

        positions = torch.arange(max_seq_len)
        theta_ik = torch.outer(positions, tmp)

        self.register_buffer("embedding_cos", torch.cos(theta_ik)) 
        self.register_buffer("embedding_sin", torch.sin(theta_ik)) 

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x0 = x[..., 0::2]
        x1 = x[..., 1::2] 
        cos = self.embedding_cos[token_positions]
        sin = self.embedding_sin[token_positions]

        # (cos, -sin) x[0]  ->  x[0] cos - x[1] sin 
        # (sin,  cos) x[1]  ->  x[0] sin + x[1] cos
        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos

        output = torch.stack([y0, y1], dim = -1).flatten(-2)

        return output

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        self.h = num_heads
        self.d_k = d_model // num_heads

        self.w_q = Linear(in_features = d_model, out_features = num_heads * self.d_k, device=device, dtype=dtype)
        self.w_k = Linear(in_features = d_model, out_features = num_heads * self.d_k, device=device, dtype=dtype)
        self.w_v = Linear(in_features = d_model, out_features = num_heads * self.d_k, device=device, dtype=dtype)
        self.w_o = Linear(in_features = num_heads * self.d_k, out_features = d_model, device=device, dtype=dtype)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        batch_size = x.shape[0]
        max_seq_len = x.shape[1]

        # (batch_size, max_seq_len, d) -> (batch_size, num_heads, max_seq_len, d_k) 
        Q = Q.view(batch_size, max_seq_len, self.h, self.d_k).transpose(-3, -2)
        K = K.view(batch_size, max_seq_len, self.h, self.d_k).transpose(-3, -2)
        V = V.view(batch_size, max_seq_len, self.h, self.d_k).transpose(-3, -2)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, device=x.device, dtype=torch.bool))

        attn = scaled_dot_product_attention(Q, K, V, mask)
        concat_attn = attn.transpose(-3, -2).reshape(batch_size, max_seq_len, self.h * self.d_k)
        output = self.w_o(concat_attn)

        return output

class MultiheadSelfAttentionWithRope(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, device=None, dtype=None):
        super().__init__()
        self.h = num_heads
        self.d_k = d_model // num_heads

        self.w_q = Linear(in_features = d_model, out_features = num_heads * self.d_k, device=device, dtype=dtype)
        self.w_k = Linear(in_features = d_model, out_features = num_heads * self.d_k, device=device, dtype=dtype)
        self.w_v = Linear(in_features = d_model, out_features = num_heads * self.d_k, device=device, dtype=dtype)
        self.w_o = Linear(in_features = num_heads * self.d_k, out_features = d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device, dtype=dtype)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        batch_size = x.shape[0]
        max_seq_len = x.shape[1]

        # (batch_size, max_seq_len, d) -> (batch_size, num_heads, max_seq_len, d_k) 
        Q = Q.view(batch_size, max_seq_len, self.h, self.d_k).transpose(-3, -2)
        K = K.view(batch_size, max_seq_len, self.h, self.d_k).transpose(-3, -2)
        V = V.view(batch_size, max_seq_len, self.h, self.d_k).transpose(-3, -2)

        Q = self.rope(x = Q, token_positions = token_positions)
        K = self.rope(x = K, token_positions = token_positions)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, device=x.device, dtype=torch.bool))

        attn = scaled_dot_product_attention(Q, K, V, mask)
        concat_attn = attn.transpose(-3, -2).reshape(batch_size, max_seq_len, self.h * self.d_k)
        output = self.w_o(concat_attn)

        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None, dtype=None):
        super().__init__()
        self.rms_norm_1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn_rope = MultiheadSelfAttentionWithRope(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.rms_norm_2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.swiglu = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x): # x(batch_size, max_seq_len, d_model)
        token_positions = torch.arange(0, x.shape[-2], 1, device=x.device)

        x_norm = self.rms_norm_1(x) 
        y = x + self.attn_rope(x_norm, token_positions)

        y_norm = self.rms_norm_2(y)
        output = y + self.swiglu(y_norm)

        return output

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device=None, dtype=None):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.transformer_layers = nn.ModuleList(
            TransformerBlock(d_model = d_model, num_heads = num_heads, d_ff =d_ff, max_seq_len = context_length, theta = rope_theta, device=device, dtype=dtype) for _ in range(num_layers)
        )
        self.rms_norm = RMSNorm(d_model = d_model, device=device, dtype=dtype)
        self.linear = Linear(d_model, vocab_size, device=device, dtype=dtype)
    def forward(self, in_indices):
        x = self.embedding(in_indices)
        for layer in self.transformer_layers:
            x = layer(x)
        norm = self.rms_norm(x)
        linear = self.linear(norm)

        return linear
