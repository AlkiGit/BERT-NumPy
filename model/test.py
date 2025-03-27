import numpy as np

def compute_mlm_accuracy(logits, labels):
    """
    Masked Language Modeling Accuracy 계산
    logits: (B, T, V)
    labels: (B, T), -100은 무시됨
    """
    preds = np.argmax(logits.data, axis=-1)
    mask = (labels != -100)
    correct = (preds == labels) & mask
    total = np.sum(mask)
    correct = np.sum(correct)
    return correct / total if total > 0 else 0.0

def compute_nsp_accuracy(logits, labels):
    """
    Next Sentence Prediction Accuracy 계산
    logits: (B, 2)
    labels: (B,)
    """
    preds = np.argmax(logits.data, axis=-1)
    correct = np.sum(preds == labels)
    return correct / len(labels)