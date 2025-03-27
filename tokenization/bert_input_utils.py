# bert_input_utils.py
def convert_tokens_to_ids(tokens, vocab):
    """토큰을 토큰 ID로 변환하는 함수"""
    token_ids = []
    for token in tokens:
        if token in vocab:
            token_ids.append(vocab.index(token))
        else:
            token_ids.append(vocab.index("[UNK]"))
    return token_ids

def create_segment_ids(token_ids, sep_index):
    """세그먼트 ID 생성 함수"""
    segment_ids = [0] * (sep_index + 1) + [1] * (len(token_ids) - sep_index - 1)
    return segment_ids

def create_position_ids(token_ids):
    """위치 ID 생성 함수"""
    position_ids = list(range(len(token_ids)))
    return position_ids

def pad_sequences(token_ids, max_length, pad_id=0):
    """패딩 추가 함수"""
    padded_ids = token_ids + [pad_id] * (max_length - len(token_ids))
    return padded_ids[:max_length]

def create_attention_mask(token_ids, max_length):
    """어텐션 마스크 생성 함수"""
    attention_mask = [1] * len(token_ids) + [0] * (max_length - len(token_ids))
    return attention_mask[:max_length]