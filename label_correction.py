import numpy as np

def cos_similarity(x1, x2, eps=1e-8):
    """Compute the cosine similarity matrix"""

    norm_x1 = np.linalg.norm(x1, axis=1, keepdims=True)
    norm_x2 = np.linalg.norm(x2, axis=1, keepdims=True)
    return (x1 @ x2.T) / (norm_x1 @ norm_x2.T + eps)

def calculate_S(z):
    """Compute the sample similarity matrix"""
    return cos_similarity(z, z)

def calculate_rho(S, rate=0.4):
    """Calculate sample density \rho"""
    m = S.shape[0]
    Sc = np.quantile(S, 1 - rate, interpolation='higher')
    return (S > Sc).sum(axis=1) - (np.diag(S) > Sc)

def get_prototype_index(S, rho, p):
    """Getting the prototype sample index"""
    rho_max = rho.max()
    m = S.shape[0]
    
    eta = np.where(
        rho == rho_max,
        S.min(axis=1),
        np.fmax(S.diagonal(), 
                np.max(S * (rho[:, None] > rho), axis=1))
    )
    return np.argpartition(eta, p)[:p]

def get_prototypes(z, y, k, p, samples_n=1280):
    """Get samples of prototypes for each category"""
    prototypes = []
    for c in range(k):
        class_samples = z[y == c]
        n_samples = min(samples_n, len(class_samples))
        
        if len(class_samples) > samples_n:
            selected = np.random.choice(len(class_samples), n_samples, replace=False)
        else:
            selected = np.arange(len(class_samples))
        
        z_samples = class_samples[selected]
        S = calculate_S(z_samples)
        rho = calculate_rho(S)
        proto_idx = get_prototype_index(S, rho, p)
        prototypes.append(z_samples[proto_idx])
    return prototypes

def produce_pseudo_labels(z, y, k, p=14, samples_n=1280):
    """Generate pseudo-labels"""
    prototypes_list = get_prototypes(z, y, k, p, samples_n)
    
    sigma = np.stack([
        cos_similarity(z, protos).mean(axis=1)
        for protos in prototypes_list
    ], axis=1)
    
    return np.argmax(sigma, axis=1)
