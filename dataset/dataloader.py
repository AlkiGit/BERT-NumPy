from datasets import load_dataset

def load_dataset_wikitext2(min_length=10):
    """
    HuggingFace datasets에서 wikitext-2 corpus를 불러오고,
    빈 줄 및 너무 짧은 문장을 제거한 후 리스트로 반환합니다.
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    raw_text = dataset["train"]["text"]

    corpus = [line.strip() for line in raw_text if len(line.strip()) >= min_length]
    print(f"[✔] Wikitext-2 loaded. Total {len(corpus)} non-empty lines.")
    return corpus

def load_dataset_by_name(name: str, **kwargs):
    if name == "wikitext2":
        return load_dataset_wikitext2(**kwargs)
    else:
        raise ValueError(f"Unknown dataset name: {name}")