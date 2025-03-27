import numpy as np
from computation.tensor import Tensor
from model.layer import EncoderLayer

class BERTModel:
    def __init__(self, vocab_size, max_length, hidden_size, num_heads, num_layers, ff_dim):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_length = max_length

        # 학습 가능한 embedding 테이블들
        self.token_embedding = Tensor(np.random.randn(vocab_size, hidden_size) * 0.02, requires_grad=True)
        self.segment_embedding = Tensor(np.random.randn(2, hidden_size) * 0.02, requires_grad=True)
        self.position_embedding = Tensor(np.random.randn(max_length, hidden_size) * 0.02, requires_grad=True)

        # 인코더 레이어들
        self.encoder_layers = [EncoderLayer(hidden_size, num_heads, ff_dim) for _ in range(num_layers)]

        # MLM/NSP head
        self.mlm_head = Tensor(np.random.randn(hidden_size, vocab_size) * 0.02, requires_grad=True)
        self.nsp_head = Tensor(np.random.randn(hidden_size, 2) * 0.02, requires_grad=True)

    def __call__(self, input_ids, segment_ids, attention_mask):
        batch_size, seq_len = input_ids.shape

        # 임베딩 처리
        token_embed = self.token_embedding.data[input_ids.data]                 # (B, T, H)
        segment_embed = self.segment_embedding.data[segment_ids]          # (B, T, H)
        position_embed = self.position_embedding.data[:seq_len]           # (T, H)
        position_embed = np.broadcast_to(position_embed, (batch_size, seq_len, self.hidden_size))

        embeddings = Tensor(token_embed + segment_embed + position_embed, requires_grad=True)

        # 인코더 통과
        x = embeddings
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)

        # MLM head (token-wise linear)
        mlm_logits = x.matmul(self.mlm_head)

        # NSP head (first token representation만 추출)
        cls_token = x.data[:, 0]                   # (B, H)
        nsp_logits = Tensor(cls_token, requires_grad=True).matmul(self.nsp_head)

        return mlm_logits, nsp_logits