import numpy as np
import time

def save_model(path, params):
    """
    Tensor 파라미터 리스트를 .npz 파일로 저장
    """
    weights = {f"param_{i}": p.data for i, p in enumerate(params)}
    np.savez(path, **weights)
    print(f"[✔] 모델 저장 완료 → {path}")

def load_model(path, params):
    """
    .npz 파일에서 Tensor 파라미터 데이터를 불러옴
    """
    data = np.load(path)
    for i, p in enumerate(params):
        p.data = data[f"param_{i}"]
    print(f"[✔] 모델 로드 완료 ← {path}")

def collect_parameters(model):
    """
    BERTModel에서 학습 가능한 모든 Tensor 파라미터를 리스트로 반환
    """
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
    return params

def print_progress_bar(step, total, start_time, bar_length=40, epoch=None):
    """
    ETA, steps/sec 포함한 tqdm 유사 진행률 바 출력 (표준 print 기반)
    """
    now = time.time()
    elapsed = now - start_time
    steps_per_sec = step / elapsed if elapsed > 0 else 0
    eta = (total - step) / steps_per_sec if steps_per_sec > 0 else 0

    percent = int(100 * step / total)
    filled_len = int(bar_length * step / total)
    bar = '=' * filled_len + '-' * (bar_length - filled_len)

    msg = f"|{bar}| {percent:3d}% ({step}/{total})"
    msg += f" | {steps_per_sec:.2f} it/s | ETA: {eta:.1f}s"

    if epoch is not None:
        print(f"Epoch {epoch} {msg}", end='\r')
    else:
        print(f"{msg}", end='\r')

    if step == total:
        print()  # 줄바꿈

def print_progress_stats(step, total, start_time):
    """
    간단한 진행 정보 출력: 현재 step / total, it/s, ETA만
    """
    now = time.time()
    elapsed = now - start_time
    steps_per_sec = step / elapsed if elapsed > 0 else 0
    eta = (total - step) / steps_per_sec if steps_per_sec > 0 else 0

    print(f"{step}/{total} | {steps_per_sec:.2f} it/s | ETA: {eta:.1f}s", end='\r')
    if step == total:
        print()  # 줄바꿈