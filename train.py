import numpy as np
from computation.tensor import Tensor
from model.bertmodel import BERTModel
from model.loss import Adam
from model.test import compute_mlm_accuracy, compute_nsp_accuracy
from model.utils import collect_parameters, save_model, load_model
from dataset.dataloader import load_dataset_by_name
from model.settings import *
from model.trainer import train_loop
from tokenization.wordpiece_utils import preprocess, tokenize
from tokenization.bert_input_utils import (
    convert_tokens_to_ids,
    create_segment_ids,
    pad_sequences,
    create_attention_mask
)

# -------------------------------
# 데이터 및 토크나이저
# -------------------------------
corpus = load_dataset_by_name("wikitext2", min_length=min_sentence_length)
from tokenization.wordpiece_utils import get_word_counts, get_initial_vocab, create_vocab
word_counts = get_word_counts(corpus)
initial_vocab = get_initial_vocab(word_counts)
vocab = create_vocab(word_counts, initial_vocab, vocab_size)
mask_token_id = vocab.index(mask_token) if mask_token in vocab else len(vocab) - 1

# -------------------------------
# 모델 및 옵티마이저 초기화
# -------------------------------
model = BERTModel(len(vocab), max_length, hidden_size, num_heads, num_layers, ff_dim)
optimizer = Adam(learning_rate=learning_rate)

# -------------------------------
# 학습
# -------------------------------
train_loop(
    model=model,
    corpus=corpus,
    vocab=vocab,
    optimizer=optimizer,
    max_length=max_length,
    mask_token=mask_token,
    epochs=epochs
)

# -------------------------------
# 저장 / 로드
# -------------------------------
params = collect_parameters(model)
# save_model(save_path, params)
load_model(save_path, params)

# -------------------------------
# 테스트 예시
# -------------------------------
text = "The capital of France is [MASK]."
text = preprocess(text)
tokens = ["[CLS]"] + tokenize(text, vocab) + ["[SEP]"]

if len(tokens) > max_length:
    if "[MASK]" in tokens:
        mask_i = tokens.index("[MASK]")
        half = max_length // 2
        start = max(0, mask_i - half)
        tokens = tokens[start:start + max_length]
        print(f"💡 tokens 잘라냄: 포함 범위 [{start}:{start+max_length}]")
    else:
        tokens = tokens[:max_length]

input_ids = convert_tokens_to_ids(tokens, vocab)
segment_ids = create_segment_ids(tokens, tokens.index("[SEP]"))
attention_mask = create_attention_mask(input_ids, max_length)
input_ids = pad_sequences(input_ids, max_length)
segment_ids = pad_sequences(segment_ids, max_length)
attention_mask = pad_sequences(attention_mask, max_length)

try:
    mask_index = input_ids.index(mask_token_id)
except ValueError:
    print("⚠️ [MASK]가 패딩 후 시퀀스에서 제거되었습니다.")
    print("tokens:", tokens)
    mask_index = None

input_ids_tensor = Tensor(np.array([input_ids]), requires_grad=False)
segment_ids_tensor = np.array([segment_ids])
attention_mask_tensor = np.array([attention_mask])

mlm_logits, _ = model(input_ids_tensor, segment_ids_tensor, attention_mask_tensor)

if mask_index is not None:
    logits_at_mask = mlm_logits.data[0, mask_index]
    top_k_ids = np.argsort(logits_at_mask)[-5:][::-1]
    top_k_tokens = [vocab[i] for i in top_k_ids]
    print(f"Top predictions for [MASK]: {top_k_tokens}")
else:
    print("추론 실패: [MASK]가 잘려나감.")
