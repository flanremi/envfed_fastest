import os
import time

import numpy as np
import torch
from sklearn.decomposition import PCA


def get_pca_by_model(models:list[np.ndarray]):
    tmp = np.array(models)
    pca = PCA(n_components=10)  # 降到10维
    results = pca.fit_transform(tmp).tolist()
    return results


