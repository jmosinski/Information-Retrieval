import numpy as np


def average_precision_score(ranking_relevancy):
    ranking = np.array(ranking_relevancy)
    relevant, = np.where(ranking>0)
    if not relevant.size: return 0
    ranks = relevant + 1
    precisions = np.cumsum(ranking[relevant]) / ranks
    return np.mean(precisions)

def norm_disc_cum_gain_score(ranking_relevancy, k=None):
    if sum(ranking_relevancy) == 0: return 0
    ranking = np.array(ranking_relevancy)
    if k is None:
        k = ranking.size
    else:
        k = min(ranking.size, k)
    ranks = np.arange(k) + 1
    denominator = np.log2(ranks+1)
    
    dcg = np.sum(ranking[:k] / denominator)
    
    ideal_ranking = ranking[np.argsort(-ranking)][:k]
    idcg = np.sum(ideal_ranking / denominator)
    
    return dcg / idcg