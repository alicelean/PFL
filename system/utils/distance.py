import numpy as np
# from scipy.spatial.distance import hellinger
from scipy.special import kl_div
from scipy.spatial.distance import jensenshannon

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


# def hellinger_distance(a, b):
#     return hellinger(np.array(a), np.array(b))


def kl_divergence(a, b):
    return kl_div(np.array(a), np.array(b)).sum()

def jensen_shannon_distance(a, b):
    return jensenshannon(np.array(a), np.array(b))