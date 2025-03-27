# model/setting.py

# 모델 구조
vocab_size = 1000
hidden_size = 64
num_heads = 8
num_layers = 2
ff_dim = 256
max_length = 16

# 학습 설정
batch_size = 2
epochs = 3
learning_rate = 1e-3
mask_token = "[MASK]"
mask_prob = 0.15

# 토크나이저 설정
min_sentence_length = 10

# 경로 설정
save_path = "bert_model.npz"