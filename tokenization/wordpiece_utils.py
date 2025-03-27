import re
from collections import defaultdict

def preprocess(text):
    """텍스트 전처리 함수"""
    special_tokens = ["[MASK]", "[CLS]", "[SEP]", "[PAD]", "[UNK]"]
    for token in special_tokens:
        text = text.replace(token, f" {token} ")
    text = re.sub(r"([.,!?])", r" \1", text)
    text = re.sub(r"(\d+)", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def get_word_counts(corpus):
    """말뭉치에서 단어 빈도를 계산하는 함수"""
    word_counts = defaultdict(int)
    for sentence in corpus:
        words = sentence.split()
        for word in words:
            word_counts[word] += 1
    return word_counts

def get_initial_vocab(word_counts):
    """초기 어휘 집합을 생성하는 함수"""
    vocab = set()
    for word in word_counts:
        for char in word:
            vocab.add(char)
    vocab.add("[UNK]")  # Unknown 토큰
    vocab.add("[CLS]")  # Classification 토큰
    vocab.add("[SEP]")  # Separator 토큰
    vocab.add("[PAD]")  # Padding 토큰
    vocab.add("[MASK]")  # Masking 토큰
    return sorted(list(vocab))

def get_pairs(word_counts):
    """단어 쌍 빈도 계산"""
    pairs = defaultdict(int)
    for word, count in word_counts.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += count
    return pairs

def merge_vocab(pair, word_counts):
    """단어 쌍 병합"""
    new_word_counts = {}
    pair_str = ' '.join(pair)
    for word in word_counts:
        new_word = word.replace(pair_str, pair_str.replace(" ", ""))
        new_word_counts[new_word] = word_counts[word]
    return new_word_counts

def create_vocab(word_counts, initial_vocab, vocab_size):
    """어휘 집합 생성"""
    vocab = initial_vocab[:]
    word_counts = {word: count for word, count in word_counts.items()}

    while len(vocab) < vocab_size:
        pairs = get_pairs(word_counts)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        word_counts = merge_vocab(best_pair, word_counts)
        vocab.append(best_pair[0] + best_pair[1])

    return vocab

def tokenize(text, vocab, do_preprocess=True):
    if do_preprocess:
        text = preprocess(text)

    special_tokens = {"[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]"}
    tokens = []

    words = text.split()
    for word in words:
        if word in special_tokens:
            tokens.append(word)
            continue

        if word in vocab:
            tokens.append(word)
            continue

        # WordPiece 분해
        sub_tokens = []
        start = 0
        while start < len(word):
            found = False
            for end in range(len(word), start, -1):
                subword = word[start:end]
                if subword in vocab:
                    if start > 0:
                        sub_tokens.append("##" + subword)
                    else:
                        sub_tokens.append(subword)
                    start = end
                    found = True
                    break
            if not found:
                sub_tokens.append("[UNK]")
                start += 1

        tokens.extend(sub_tokens)

    return tokens