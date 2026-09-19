import numpy as np


DEFAULT_DIM = 512
DEFAULT_DTYPE = np.float32


def to_fixed_dim(x, dim=DEFAULT_DIM, dtype=DEFAULT_DTYPE):
    x = np.asarray(x, dtype=dtype).flatten()
    if len(x) > dim:
        return x[:dim]
    if len(x) < dim:
        result = np.zeros(dim, dtype=dtype)
        result[:len(x)] = x
        return result
    return x


def to_fixed_shape(x, shape, dtype=DEFAULT_DTYPE):
    x = np.asarray(x, dtype=dtype).flatten()
    total = int(np.prod(shape))
    if len(x) > total:
        x = x[:total]
    elif len(x) < total:
        result = np.zeros(total, dtype=dtype)
        result[:len(x)] = x
        x = result
    return x.reshape(shape)


def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x)
    if n > 1e-10:
        x = x / n
    return x.astype(DEFAULT_DTYPE)


def cosine_sim(a, b):
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-10:
        return 0.0
    return float(np.dot(a, b) / denom)


def safe_stack(arrays, dim=DEFAULT_DIM, dtype=DEFAULT_DTYPE):
    if not arrays:
        return np.zeros((0, dim), dtype=dtype)
    fixed = [to_fixed_dim(a, dim, dtype) for a in arrays]
    return np.stack(fixed)


def blend_vectors(v1, v2, alpha=0.5, dim=DEFAULT_DIM, dtype=DEFAULT_DTYPE):
    v1 = to_fixed_dim(v1, dim, dtype)
    v2 = to_fixed_dim(v2, dim, dtype)
    return (1.0 - alpha) * v1 + alpha * v2


def extract_topk_neurons(vector, k=10, total_neurons=1000):
    vector = np.asarray(vector).flatten()
    abs_vals = np.abs(vector)
    indices = np.argsort(abs_vals)[::-1]
    result = []
    for idx in indices[:k]:
        nid = int(abs(vector[idx]) * total_neurons) % total_neurons
        strength = float(abs(vector[idx]))
        result.append((nid, strength))
    return result