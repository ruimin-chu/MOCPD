import numpy as np

def preprocess(data, fixed_t):
    del_idx = []
    for i in range (data.shape[0]):
        if abs(data[i, 1]) > fixed_t:
            del_idx.append(i)
    return np.delete(data, del_idx, axis=0)

def sliding_window(elements, window_size, step):
    if len(elements) <= window_size:
        return elements
    new = np.empty((0, window_size))
    for i in range(0, len(elements) - window_size + 1, step):
        new = np.vstack((new, elements[i:i+window_size]))
    return new