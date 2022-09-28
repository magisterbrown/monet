from  pytorch_fid.fid_score import  calculate_frechet_distance
import numpy as np

def fid_dis(vec1: np.array, vec2: np.array):
    mu1 = np.mean(vec1, axis=0)
    sigma1 = np.cov(vec1, rowvar=False)
    mu2 = np.mean(vec2, axis=0)
    sigma2 = np.cov(vec2, rowvar=False)

    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

def mfid(x_real: np.array, y_pred: np.array, eps = 0.1):
    fid = fid_dis(x_real, y_pred)
    coss_dis = cosine_distance(x_real, y_pred)
    coss_tresh = coss_dis if coss_dis < eps else 1

    print("FID_public: ", fid, "distance_public: ", coss_dis, "multiplied_public: ", fid/(coss_tresh + eps))

    return fid

def cosine_distance(features1, features2):
    features1_nozero = features1[np.sum(features1, axis=1) != 0]
    features2_nozero = features2[np.sum(features2, axis=1) != 0]
    norm_f1 = normalize_rows(features1_nozero)
    norm_f2 = normalize_rows(features2_nozero)

    d = 1.0-np.abs(np.matmul(norm_f1, norm_f2.T))
    mean_min_d = np.mean(np.min(d, axis=1))
    return mean_min_d

def normalize_rows(x: np.ndarray):
    """
    function that normalizes each row of the matrix x to have unit length.

    Args:
     ``x``: A numpy matrix of shape (n, m)

    Returns:
     ``x``: The normalized (by row) numpy matrix.
    """
    return np.nan_to_num(x/np.linalg.norm(x, ord=2, axis=1, keepdims=True))

