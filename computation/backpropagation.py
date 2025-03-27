import numpy as np

def backward(tensor):
    """역전파 알고리즘"""
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(tensor)

    tensor.grad = np.ones_like(tensor.data)
    for v in reversed(topo):
        v._backward()