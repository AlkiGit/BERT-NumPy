import numpy as np
from computation.tensor import Tensor
from tokenization.wordpiece_utils import tokenize, preprocess
from tokenization.bert_input_utils import (
    convert_tokens_to_ids,
    create_segment_ids,
    create_attention_mask,
    pad_sequences
)


def prepare_bert_inputs(sent_a, sent_b, vocab, max_length, mask_token="[MASK]", mask_prob=0.15):
    tokens_a = tokenize(sent_a, vocab)
    tokens_b = tokenize(sent_b, vocab)
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]

    segment_ids = create_segment_ids(tokens, tokens.index("[SEP]"))
    input_ids = convert_tokens_to_ids(tokens, vocab)
    attention_mask = create_attention_mask(input_ids, max_length)

    # MLM Masking
    mlm_labels = np.full_like(input_ids, -100)
    masked_input_ids = input_ids.copy()
    mask_token_id = vocab.index(mask_token) if mask_token in vocab else len(vocab) - 1

    for i in range(len(input_ids)):
        if tokens[i] in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        if np.random.rand() < mask_prob:
            mlm_labels[i] = input_ids[i]
            prob = np.random.rand()
            if prob < 0.8:
                masked_input_ids[i] = mask_token_id
            elif prob < 0.9:
                masked_input_ids[i] = np.random.randint(0, len(vocab))

    # Padding
    masked_input_ids = pad_sequences(masked_input_ids, max_length)
    segment_ids = pad_sequences(segment_ids, max_length)
    attention_mask = pad_sequences(attention_mask, max_length)
    mlm_labels = pad_sequences(mlm_labels.tolist(), max_length)

    # Transforms
    input_ids_tensor = Tensor(np.array([masked_input_ids]), requires_grad=False)
    segment_ids_tensor = np.array([segment_ids])
    attention_mask_tensor = np.array([attention_mask])
    mlm_labels = np.array([mlm_labels])

    return input_ids_tensor, segment_ids_tensor, attention_mask_tensor, mlm_labels
