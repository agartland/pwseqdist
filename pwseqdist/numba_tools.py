import numpy as np
import scipy.special
import itertools
import numba as nb


__all__ = ['nb_distance_vec']


@nb.jit(nopython=True, parallel=False, nogil=True)
def nb_distance_vec(indices, seqs_mat, seqs_L, nb_metric, *args):
    """Compute distances between pairs of sequences in seqs_mat specified by indices.

    Note: numba raised errors when this function tried to use *args inside the prange
    function. Without *args the prange function appears to fully parallelize the computation
    but ultimately args were a requirement for functionality, so parallel = False.

    Because these were not able to be parallelized by numba, they would need to be parallelized using
    multi-processing (like the python metrics).

    Parameters
    ----------
    indices : np.ndarray [nseqs, 2]
        Indices into seqs_mat indicating pairs of sequences to compare.
    seqs_mat : np.ndarray dtype=int16 [nseqs, seq_length]
        Created by pwsd.seqs2mat with padding to accomodate
        sequences of different lengths (-1 padding)
    seqs_L : np.ndarray [nseqs]
        A vector containing the length of each sequence,
        without the padding in seqs_mat
    nb_metric : nb.jitt'ed function
        Any function that has been numba-compiled, taking
        two numpy vector representations of sequences and *args"""

    assert seqs_mat.shape[0] == seqs_L.shape[0]
    
    dist = np.zeros(indices.shape[0], dtype=np.int16)
    for ind_i in nb.prange(indices.shape[0]):
        dist[ind_i] = nb_metric(seqs_mat[indices[ind_i, 0], :seqs_L[indices[ind_i, 0]]],
                                seqs_mat[indices[ind_i, 1], :seqs_L[indices[ind_i, 1]]], *args)
    return dist


def _nb_pairwise_sq(seqs, nb_metric, *args):
    """Calculate distance between all pairs of seqs using metric
    and kwargs provided to nb_metric. Will use multiprocessing Pool
    if ncpus > 1.

    nb_metric must be a numba-compiled function

    Parameters
    ----------
    seqs : list
        List of sequences provided to metric in pairs.
    metric : numba-compiled function
        A distance function of the form
        func(seq1, seq2, **kwargs)
    **kwargs : keyword arguments
        Additional keyword arguments are supplied to the metric.

    Returns
    -------
    dvec : np.ndarray, length n*(n - 1) / 2
        Vector form of the pairwise distance matrix.
        Use scipy.distance.squareform to convert to a square matrix"""
    nb_seqs = nb.typed.List()
    for s in seqs:
        nb_seqs.append(s)

    dvec = np.zeros(int(scipy.special.comb(len(seqs), 2)))
    indices = np.zeros((int(scipy.special.comb(len(seqs), 2)), 2), dtype=np.int)
    for veci, ij in enumerate(itertools.combinations(range(len(seqs)), 2)):
        indices[veci, :] = ij

    distance_vec(dvec, indices, nb_seqs, nb_metric, *args)
    return dvec

def _nb_dict_from_matrix(subst):
    """Converts a parasail substitution matrix into a nb.typed.Dict
    that can be used with he numba metrics"""
    subst_dict = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                               value_type=nb.types.int32)
    for k,v in dict_from_matrix(subst).items():
        key = '|'.join(k)
        subst_dict[key] = v
    return subst_dict
