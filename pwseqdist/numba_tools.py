import numpy as np
import scipy.special
import itertools
import numba

from .matrices import dict_from_matrix

__all__ = ['nb_pairwise_sq',
            'nb_hamming_distance',
            'nb_subst_metric',
            'nb_dict_from_matrix']

try:
    import numba
    # print('pwseqdist: Successfully imported numba version %s' % (numba.__version__))
    NB_SUCCESS = True
except ImportError:
    NB_SUCCESS = False
    print('pwseqdist: Could not import numba')

def nb_dict_from_matrix(subst):
    """Converts a parasail substitution matrix into a numba.typed.Dict
    that can be used with he numba metrics"""
    subst_dict = numba.typed.Dict.empty(key_type=numba.types.unicode_type,
                               value_type=numba.types.int32)
    for k,v in dict_from_matrix(subst).items():
        key = '|'.join(k)
        subst_dict[key] = v
    return subst_dict

@numba.jit(nopython=True, parallel=False)
def distance_vec(dvec, indices, seqs, nb_metric, *args):
    for veci in numba.prange(len(indices)):
        si = seqs[indices[veci, 0]]
        sj = seqs[indices[veci, 1]]
        d = nb_metric(si, sj, *args)
        dvec[veci] = d

def nb_pairwise_sq(seqs, nb_metric, *args):
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
    nb_seqs = numba.typed.List()
    for s in seqs:
        nb_seqs.append(s)

    dvec = np.zeros(int(scipy.special.comb(len(seqs), 2)))
    indices = np.zeros((int(scipy.special.comb(len(seqs), 2)), 2), dtype=np.int)
    for veci, ij in enumerate(itertools.combinations(range(len(seqs)), 2)):
        indices[veci, :] = ij

    distance_vec(dvec, indices, nb_seqs, nb_metric, *args)
    return dvec

if not NB_SUCCESS:
    nb_hamming_distance = None
    nb_subst_metric = None
else:
    @numba.jit(nopython=True)
    def nb_hamming_distance(str1, str2):
        assert len(str1) == len(str2)

        tot = 0
        for i in range(len(str1)):
            if str1[i] != str2[i]:
                tot += 1
        return tot

    @numba.jit(nopython=True)
    def nb_subst_metric(seq1, seq2, subst_dict, as_similarity=False):
        """Computes sequence similarity based on the substitution matrix."""
        assert len(seq1) == len(seq2)

        def _sim_func(s1, s2, subst):
            sim12 = 0.
            for i in range(len(s1)):
                k1 = s1[i] + '|' + s2[i]
                k2 = s2[i] + '|' + s1[i]
                sim12 += subst.get(k1, subst.get(k2, subst['n|a']))
            return sim12

        """Site-wise similarity between seq1 and seq2 using the substitution matrix subst"""
        sim12 = _sim_func(seq1, seq2, subst_dict)

        if as_similarity:
            return sim12
        else:
            L = len(seq1)
            sim11 = _sim_func(seq1, seq1, subst_dict)
            sim22 = _sim_func(seq2, seq2, subst_dict)
            D = sim11 + sim22 - 2 * sim12
            return D