import numpy as np
from computation.tensor import Tensor


class MultiHeadAttention:
    def __init__(self, hidden_size, num_heads):
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.depth = hidden_size // num_heads

        self.wq = Tensor(np.random.randn(hidden_size, hidden_size), requires_grad=True)
        self.wk = Tensor(np.random.randn(hidden_size, hidden_size), requires_grad=True)
        self.wv = Tensor(np.random.randn(hidden_size, hidden_size), requires_grad=True)
        self.wo = Tensor(np.random.randn(hidden_size, hidden_size), requires_grad=True)

    def scaled_dot_product_attention(self, query, key, value, mask=None): # self 추가
        """Scaled Dot-Product Attention"""
        if len(key.data.shape) >= 2:
            matmul_qk = query.matmul(key.transpose(-2, -1))
        else:
            # key 텐서가 2차원 미만인 경우 전치 없이 계산
            matmul_qk = query.matmul(key)

        d_k = Tensor(np.array(query.data.shape[-1]))
        scaled_attention_logits = matmul_qk / d_k.sqrt()

        if mask is not None:
            if not isinstance(mask, Tensor):
                mask = Tensor(mask)

            # (B, T) → (B, 1, 1, T) 로 reshape
            if len(mask.data.shape) == 2:
                mask = mask.reshape(mask.data.shape[0], 1, 1, mask.data.shape[1])

            scaled_attention_logits += (mask * Tensor(-1e9))

        attention_weights = scaled_attention_logits.softmax(axis=-1)
        output = attention_weights.matmul(value)
        return output, attention_weights

    def __call__(self, value, key, query, mask=None):
        batch_size = query.data.shape[0]

        query_len = len(query.data.shape)
        key_len = len(key.data.shape)
        value_len = len(value.data.shape)

        query = query.matmul(self.wq).reshape(batch_size, -1, self.num_heads, self.depth).transpose(0, 2, 1, 3)
        key = key.matmul(self.wk).reshape(batch_size, -1, self.num_heads, self.depth).transpose(0, 2, 1, 3)
        value = value.matmul(self.wv).reshape(batch_size, -1, self.num_heads, self.depth).transpose(0, 2, 1, 3)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(query, key, value, mask) # self 추가
        scaled_attention = scaled_attention.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.hidden_size)
        output = scaled_attention.matmul(self.wo)
        return output, attention_weights

class FeedForwardNetwork:
    def __init__(self, hidden_size: int, ff_dim: int):
        self.w1 = Tensor(np.random.randn(hidden_size, ff_dim) * np.sqrt(2.0 / hidden_size), requires_grad=True)
        self.w2 = Tensor(np.random.randn(ff_dim, hidden_size) * np.sqrt(2.0 / ff_dim), requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        x = x.matmul(self.w1).relu()
        x = x.matmul(self.w2)
        return x

class EncoderLayer:
    def __init__(self, hidden_size, num_heads, ff_dim):
        self.mha = MultiHeadAttention(hidden_size, num_heads)
        self.ffn = FeedForwardNetwork(hidden_size, ff_dim)

        self.layernorm1 = LayerNorm(hidden_size)
        self.layernorm2 = LayerNorm(hidden_size)

    def __call__(self, x, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2
    
class LayerNorm:
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma = Tensor(np.ones((1, 1, hidden_size)), requires_grad=True)
        self.beta = Tensor(np.zeros((1, 1, hidden_size)), requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=-1, keepdims=True)
        variance = ((x - mean) * (x - mean)).mean(axis=-1, keepdims=True)
        norm = (x - mean) / (variance + self.eps).sqrt()
        return norm * self.gamma + self.beta