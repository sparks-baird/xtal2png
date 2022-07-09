import numpy as np
from numpy.random import default_rng
from scipy.spatial import distance_matrix
from sklearn.manifold import MDS


def reconstruct_distances(coords, labels):
    dm = distance_matrix(coords, coords)
    dm_lbl = np.zeros_like(dm, dtype="object")
    for i, x in enumerate(labels):
        for j, y in enumerate(labels):
            dm_lbl[i, j] = (x, y)

    dm1 = np.array([labels] * dm.shape[0])
    dm2 = np.array([labels] * dm.shape[1]).T

    ulbls = np.unique(labels)
    nlbls = len(ulbls)
    avg_dm = np.zeros((nlbls, nlbls))
    std_dm = np.zeros((nlbls, nlbls))
    for i, lbl1 in enumerate(ulbls):
        for j, lbl2 in enumerate(ulbls):
            sub_dm = dm[(dm1 == lbl1) & (dm2 == lbl2)]
            avg_dm[i, j] = np.mean(sub_dm)
            std_dm[i, j] = np.std(sub_dm)

    template_dm = np.zeros((2, *dm.shape))
    for i, lbl1 in enumerate(labels):
        for j, lbl2 in enumerate(labels):
            avg = avg_dm[lbl1, lbl2]
            std = std_dm[lbl1, lbl2]
            template_dm[0, i, j] = avg
            template_dm[1, i, j] = std

    embedder = MDS(n_components=3, metric=True, dissimilarity="precomputed")
    rng = default_rng()
    pred_dms = []
    for _ in range(10):
        mu, sigma = template_dm
        sample_dm = rng.standard_normal(size=mu.shape)
        sample_dm = np.multiply(sigma, sample_dm) + mu
        # https://stackoverflow.com/questions/28904411/making-a-numpy-ndarray-matrix-symmetric
        sample_dm = np.tril(sample_dm) + np.triu(sample_dm.T, 1)
        np.fill_diagonal(sample_dm, 0.0)
        embedding = embedder.fit_transform(sample_dm)
        pred_dm = distance_matrix(embedding, embedding)
        pred_dms.append(pred_dm)
    pred_dm = np.mean(pred_dms, axis=0)
    embedding = embedder.fit_transform(sample_dm)
    return embedding
