import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.grad = np.zeros_like(self.data)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op='+')

        def _backward():
            if self.requires_grad:
                grad = out.grad
                while grad.ndim > self.grad.ndim:
                    grad = grad.sum(axis=0)
                for i in range(grad.ndim):
                    if grad.shape[i] != self.grad.shape[i]:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad += grad

            if other.requires_grad:
                grad = out.grad
                while grad.ndim > other.grad.ndim:
                    grad = grad.sum(axis=0)
                for i in range(grad.ndim):
                    if grad.shape[i] != other.grad.shape[i]:
                        grad = grad.sum(axis=i, keepdims=True)
                other.grad += grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op='*')

        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                self.grad += self.data * out.grad
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, _children=(self,), _op='sum')

        def _backward():
            if self.requires_grad:
                if axis is None:
                    self.grad += out.grad * np.ones_like(self.data)
                else:
                    dims_to_add = len(self.data.shape) - len(out.grad.shape)
                    if dims_to_add > 0:
                        grad = np.expand_dims(out.grad, axis=tuple(range(len(out.grad.shape), len(self.data.shape))))
                    else:
                        grad = out.grad
                    self.grad += grad * np.ones_like(self.data)
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, _children=(self,), _op='mean')

        def _backward():
            if self.requires_grad:
                if axis is None:
                    self.grad += out.grad * np.ones_like(self.data) / self.data.size
                else:
                    dims_to_add = len(self.data.shape) - len(out.grad.shape)
                    if dims_to_add > 0:
                        grad = np.expand_dims(out.grad, axis=tuple(range(len(out.grad.shape), len(self.data.shape))))
                    else:
                        grad = out.grad
                    self.grad += grad * np.ones_like(self.data) / self.data.size
        out._backward = _backward
        return out

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            np.matmul(self.data, other.data),
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op='matmul'
        )

        def _backward():
            if out.grad is None:
                return

            A = self.data
            B = other.data
            dOut = out.grad

            # case: both A and B are 2D → simple matmul
            if A.ndim == 2 and B.ndim == 2:
                if self.requires_grad:
                    self.grad += np.matmul(dOut, B.T)
                if other.requires_grad:
                    other.grad += np.matmul(A.T, dOut)

            # case: batched matmul (3D or more)
            elif A.ndim >= 3 or B.ndim >= 3:
                # A: (..., m, k), B: (..., k, n), dOut: (..., m, n)
                if self.requires_grad:
                    # ∂L/∂A = dOut @ Bᵗ → (..., m, n) @ (..., n, k) → (..., m, k)
                    B_T = np.swapaxes(B, -1, -2)  # last two dims
                    self.grad += np.matmul(dOut, B_T)

                if other.requires_grad:
                    A_T = np.swapaxes(A, -1, -2)               # Aᵗ: (batch, k, m)
                    grad_B = np.matmul(A_T, dOut)              # (batch, k, n)
                    if grad_B.shape != other.grad.shape:
                        grad_B = np.sum(grad_B, axis=0)        # sum over batch
                    other.grad += grad_B

            else:
                raise NotImplementedError("Unhandled matmul input dimensions")

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), requires_grad=self.requires_grad, _children=(self,), _op='relu')

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (self.data > 0)
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad, _children=(self,), _op='exp')

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * out.data
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad, _children=(self,), _op='log')

        def _backward():
            if self.requires_grad:
                self.grad += out.grad / self.data
        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad, _children=(self,), _op='reshape')

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def transpose(self, *axes):
        # (-2, -1) 같은 경우를 tuple로 받아 처리
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])

        rank = len(self.data.shape)

        if len(axes) == 2 and all(isinstance(a, int) for a in axes):
            # 음수 인덱스 양수화
            a1 = axes[0] if axes[0] >= 0 else rank + axes[0]
            a2 = axes[1] if axes[1] >= 0 else rank + axes[1]
            full_axes = list(range(rank))
            full_axes[a1], full_axes[a2] = full_axes[a2], full_axes[a1]
            axes = tuple(full_axes)

        out = Tensor(self.data.transpose(axes), requires_grad=self.requires_grad, _children=(self,), _op='transpose')

        def _backward():
            if self.requires_grad:
                inv_axes = tuple(np.argsort(axes))
                self.grad += out.grad.transpose(inv_axes)

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def sqrt(self):
        data = np.sqrt(self.data)
        out = Tensor(data, requires_grad=self.requires_grad, _children=(self,), _op='sqrt')

        def _backward():
            if self.requires_grad:
                self.grad += (0.5 / np.sqrt(self.data)) * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        out = Tensor(self.data / other_data, requires_grad=self.requires_grad, _children=(self,), _op='truediv')

        def _backward():
            if self.requires_grad:
                self.grad += (1 / other_data) * out.grad

            if isinstance(other, Tensor) and other.requires_grad:
                grad = -(self.data / (other_data ** 2)) * out.grad

                # 브로드캐스트된 차원 제거
                while grad.ndim > other.grad.ndim:
                    grad = grad.sum(axis=0)
                for i in range(grad.ndim):
                    if grad.shape[i] != other.grad.shape[i]:
                        grad = grad.sum(axis=i, keepdims=True)

                other.grad += grad

        out._backward = _backward
        return out
    
    def softmax(self, axis=-1):
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        exp_data = np.exp(shifted)
        sum_exp = np.sum(exp_data, axis=axis, keepdims=True)
        out_data = exp_data / sum_exp

        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op='softmax')

        def _backward():
            if self.requires_grad:
                grad = out.grad  # [*, N]
                s = out.data     # [*, N]

                # 정확한 Jacobian-vector product: grad * s - sum(grad * s) * s
                dot = np.sum(grad * s, axis=axis, keepdims=True)  # [*, 1]
                dx = s * (grad - dot)  # [*, N]

                self.grad += dx

        out._backward = _backward
        return out
    
    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, requires_grad={self.requires_grad})"
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op='-')

        def _backward():
            if self.requires_grad:
                self.grad += out.grad

            if other.requires_grad:
                grad = -out.grad

                # broadcasting 대응
                while grad.ndim > other.grad.ndim:
                    grad = grad.sum(axis=0)
                for i in range(grad.ndim):
                    if grad.shape[i] != other.grad.shape[i]:
                        grad = grad.sum(axis=i, keepdims=True)

                other.grad += grad

        out._backward = _backward
        return out
        
    def __neg__(self):
        return self * -1

    def __rsub__(self, other):
        return Tensor(other) - self
    
    def __rmul__(self, other):
        return self * other
    
    @property
    def shape(self):
        return self.data.shape