import pickle
import random
import numpy as np
import tqdm
from sklearn.metrics import f1_score


def ranking_rprecision_score(y_true, y_score, k=10):
    """Precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) == 1:
        return ValueError("The score cannot be approximated.")
    elif len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
    return float(n_relevant) / min(k, n_pos)


def mean_rprecision_k(y_true, y_score, k=10):
    """Mean precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    mean precision @k : float
    """

    p_ks = []
    for y_t, y_s in zip(y_true, y_score):
        if np.sum(y_t == 1):
            p_ks.append(ranking_rprecision_score(y_t, y_s, k=k))

    return np.mean(p_ks)


def get_score(gold, predictions, metric='r-precision'):
    if metric == 'r-precision':
        return mean_rprecision_k(gold,predictions, k=5)
    else:
        pred_targets = (predictions > 0.5).astype('int32')
        return f1_score(y_true=gold, y_pred=pred_targets, average='micro')


goldf = '/home/ichalkidis/nlp.research/TEMP_GOLD.predictions'
sysAf = '/home/ichalkidis/nlp.research/TEMP_GRUS_1.predictions'
sysBf = '/home/ichalkidis/nlp.research/TEMP_CNNs_1.predictions'
print(goldf)
print(sysAf)
print(sysBf)

with open(goldf, 'rb') as f:
  gold = pickle.load(f)
with open(sysAf, 'rb') as f:
  sysA = pickle.load(f)
with open(sysBf, 'rb') as f:
  sysB = pickle.load(f)

sysA_metric = get_score(gold, sysA)
print(sysA_metric)
sysB_metric = get_score(gold, sysB)
print(sysB_metric)
orig_diff = abs(sysA_metric - sysB_metric)
print(orig_diff)

N = 10000
num_invalid = 0

for n in tqdm.tqdm(range(1, N+1)):
    with open(sysAf, 'rb') as f:
        sysA2 = pickle.load(f)
    with open(sysBf, 'rb') as f:
        sysB2 = pickle.load(f)
    for i in range(gold.shape[0]):
        rval = random.random()
        if rval < 0.5:
            AD = [pred for pred in sysA[i]]
            BD = [pred for pred in sysB[i]]
            sysA2[i] = [pred for pred in BD]
            sysB2[i] = [pred for pred in AD]

    new_sysA_metric = get_score(gold, sysA2)
    new_sysB_metric = get_score(gold, sysB2)
    new_diff = abs(new_sysA_metric - new_sysB_metric)

    if new_diff >= orig_diff:
        num_invalid += 1

    if n % 20 == 0 and n > 0:
        print('Random Iteration {}: {}'.format(n, float(num_invalid) / float(n)))

print('Overall: {}'.format(float(num_invalid) / float(N)))
