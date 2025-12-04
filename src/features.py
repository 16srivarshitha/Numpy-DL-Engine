import numpy as np

def get_raw(coords):
    return coords.copy()

def get_polynomial(coords, order=5): 
    # Scale to [-1, 1]
    x = 2 * coords[:, 0:1] - 1 
    y = 2 * coords[:, 1:2] - 1 
    
    features = []
    for i in range(1, order + 1):
        features.append(x ** i)
        features.append(y ** i)
    
    return np.concatenate(features, axis=1)

def get_fourier(coords, freq=10):
    x = coords[:, 0:1]
    y = coords[:, 1:2]
    
    features = [np.ones((coords.shape[0], 1))]
    
    for i in range(1, freq + 1):
        features.append(np.sin(2 * np.pi * i * x))
        features.append(np.cos(2 * np.pi * i * x))
    
    for i in range(1, freq + 1):
        features.append(np.sin(2 * np.pi * i * y))
        features.append(np.cos(2 * np.pi * i * y))
    
    return np.concatenate(features, axis=1)

def normalize_features(features, method):
    if method == 'raw':
        return features
    elif method == 'polynomial':
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        std = np.maximum(std, 1e-8)
        return (features - mean) / std
    elif method == 'fourier':
        return features
    else:
        raise ValueError(f"Unknown method: {method}")