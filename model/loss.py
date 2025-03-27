from computation.tensor import Tensor
import numpy as np

def cross_entropy_loss(logits: Tensor, labels: np.ndarray, axis=-1) -> Tensor:
    """
    logits: Tensor of shape (B, T, V) or (B, V)
    labels: np.ndarray of int indices, shape (B, T) or (B,)
    """
    # softmax & log
    log_probs = logits.softmax(axis=axis).log()

    # one-hot 변환
    one_hot = np.zeros_like(log_probs.data)
    np.put_along_axis(one_hot, labels[..., np.newaxis], 1, axis=axis)

    one_hot_tensor = Tensor(one_hot, requires_grad=False)

    # loss = - sum(one_hot * log_probs) / N
    loss = -(one_hot_tensor * log_probs).sum() / labels.size
    return loss

class Adam:
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = {}  # 1st moment
        self.v = {}  # 2nd moment
        self.t = 0   # time step

    def update(self, params: list):
        self.t += 1
        for i, param in enumerate(params):
            if not param.requires_grad:
                continue

            key = id(param)
            grad = param.grad
            if grad is None:
                continue

            if key not in self.m:
                self.m[key] = np.zeros_like(grad)
                self.v[key] = np.zeros_like(grad)

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)