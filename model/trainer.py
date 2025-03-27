import numpy as np
import time

from computation.tensor import Tensor
from model.loss import cross_entropy_loss
from model.utils import print_progress_stats
from model.test import compute_mlm_accuracy, compute_nsp_accuracy
from tokenization.bert_preprocessor import prepare_bert_inputs


def train_loop(model, corpus, vocab, optimizer, max_length, mask_token="[MASK]", epochs=1):
    params = [
        model.token_embedding,
        model.segment_embedding,
        model.position_embedding,
        model.mlm_head,
        model.nsp_head
    ]
    for layer in model.encoder_layers:
        params.extend([
            layer.mha.wq, layer.mha.wk, layer.mha.wv, layer.mha.wo,
            layer.ffn.w1, layer.ffn.w2,
            layer.layernorm1.gamma, layer.layernorm1.beta,
            layer.layernorm2.gamma, layer.layernorm2.beta,
        ])

    for epoch in range(epochs):
        total_loss = 0.0
        total_steps = (len(corpus) - 1) // 2
        start_time = time.time()

        for step, i in enumerate(range(0, len(corpus) - 1, 2)):
            sent_a = corpus[i]
            sent_b = corpus[i + 1] if np.random.rand() > 0.5 else corpus[np.random.randint(0, len(corpus))]
            is_next = int(sent_b == corpus[i + 1])

            # ✅ 전처리 함수 호출 (prepare_bert_inputs)
            input_ids_tensor, segment_ids_tensor, attention_mask_tensor, mlm_labels = prepare_bert_inputs(
                sent_a, sent_b, vocab, max_length, mask_token=mask_token
            )
            nsp_labels = np.array([is_next])

            # 순전파
            mlm_logits, nsp_logits = model(input_ids_tensor, segment_ids_tensor, attention_mask_tensor)
            mlm_loss = cross_entropy_loss(mlm_logits, mlm_labels)
            nsp_loss = cross_entropy_loss(nsp_logits, nsp_labels)
            loss = mlm_loss + nsp_loss
            total_loss += loss.data

            # 역전파 + 업데이트
            loss.backward()
            optimizer.update(params)
            for p in params:
                p.grad = np.zeros_like(p.grad)

            # 진행률 표시
            print_progress_stats(step + 1, total_steps, start_time)

        # Epoch 결과 출력
        mlm_acc = compute_mlm_accuracy(mlm_logits, mlm_labels)
        nsp_acc = compute_nsp_accuracy(nsp_logits, nsp_labels)
        print(f"[Epoch {epoch+1}] Total Loss: {total_loss:.4f} | MLM Acc: {mlm_acc:.2%} | NSP Acc: {nsp_acc:.2%}")
