import numpy as np
import scipy.special
import itertools
import numba as nb

from .matrices import dict_from_matrix, parasail_aa_alphabet

__all__ = ['nb_dict_from_matrix',
           'nb_distance_vec',
           'nb_distance_rect',
           'nb_editdistance']

default_distance_matrix = np.ones((len(parasail_aa_alphabet), len(parasail_aa_alphabet)), dtype=np.int32)
default_distance_matrix[np.diag_indices_from(default_distance_matrix)] = 0

def nb_dict_from_matrix(subst):
    """Converts a parasail substitution matrix into a nb.typed.Dict
    that can be used with he numba metrics"""
    subst_dict = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                               value_type=nb.types.int32)
    for k,v in dict_from_matrix(subst).items():
        key = '|'.join(k)
        subst_dict[key] = v
    return subst_dict

@nb.jit(nopython=True, parallel=False)
def nb_distance_vec(seqs_mat, seqs_L, indices, nb_metric, *args):
    """
    Parameters
    ----------
    seqs_mat : np.ndarray dtype=int16 [nseqs, seq_length]
        Created by pwsd.seqs2mat with padding to accomodate
        sequences of different lengths (-1 padding)
    seqs_L : np.ndarray [nseqs]
        A vector containing the length of each sequence,
        without the padding in seqs_mat
    indices : np.ndarray [nseqs, 2]
        Indices into seqs_mat indicating pairs of sequences to compare.
    nb_metric : nb.jitt'ed function
        Any function that has been numba-compiled, taking
        two numpy vector representations of sequences and *args"""

    assert seqs_mat.shape[0] == seqs_L.shape[0]
        
    dist = np.zeros(indices.shape[0], dtype=np.int16)
    for ind_i in nb.prange(indices.shape[0]):
        dist[ind_i] = nb_metric(seqs_mat[indices[ind_i, 0], :seqs_L[indices[ind_i, 0]]],
                                seqs_mat[indices[ind_i, 1], :seqs_L[indices[ind_i, 1]]], *args)
    return dist

@nb.jit(nopython=True, parallel=False)
def nb_distance_rect(seqs_mat1, seqs_L1, seqs_mat2, seqs_L2, indices, nb_metric, *args):
    """Parameters
    ----------
    seqs_mat : np.ndarray dtype=int16 [nseqs, seq_length]
        Created by pwsd.seqs2mat with padding to accomodate
        sequences of different lengths (-1 padding)
    seqs_L : np.ndarray [nseqs]
        A vector containing the length of each sequence,
        without the padding in seqs_mat
    indices : np.ndarray [nseqs, 2]
        Indices into seqs_mat1 and seqs_mat2 indicating pairs of sequences to compare.
    nb_metric : nb.jitt'ed function
        Any function that has been numba-compiled, taking
        two numpy vector representations of sequences and *args"""

    assert seqs_mat1.shape[0] == seqs_L1.shape[0]
    assert seqs_mat2.shape[0] == seqs_L2.shape[0]
        
    dist = np.zeros(indices.shape[0], dtype=np.int16)
    for ind_i in nb.prange(indices.shape[0]):
        dist[ind_i] = nb_metric(seqs_mat1[indices[ind_i, 0], :seqs_L1[indices[ind_i, 0]]],
                                seqs_mat2[indices[ind_i, 1], :seqs_L2[indices[ind_i, 1]]], *args)
    return dist

@nb.jit(nopython=True)
def nb_editdistance(seq_vec1, seq_vec2, distance_matrix=default_distance_matrix, gap_penalty=1):
    """Computes the Levenshtein edit distance between two sequences, with the AA substitution
    distances provided in distance_matrix.

    The default distance matrix has a 1 for mismatches and 0 for matches.

    Parameters
    ----------
    seqs_vec : np.ndarray dtype=int16
        Vector/integer representation of a sequence, created by
        pwsd.seqs2mat. Padding should have been removed by pwsd.distance_vec
    distance_matrix : np.ndarray [alphabet, alphabet] dtype=int32
        A square distance matrix (NOT a similarity matrix).
        Matrix must match the alphabet that was used to create
        seqs_mat, where each AA is represented by an index into the alphabet.
    gap_penalty : int
        Penalty for insertions and deletions in the optimal alignment.

    Returns
    -------
    dist : int16
        Edit distance between seq1 and seq2"""
    
    q_L = seq_vec1.shape[0]
    s_L = seq_vec2.shape[0]
    if q_L == s_L:
        """No gaps: substitution distance
        This will make it differ from a strict edit-distance since
        the optimal edit-distance may insert same number of gaps in both sequences"""
        dist = 0
        for i in range(q_L):
            dist += distance_matrix[seq_vec1[i], seq_vec2[i]]
        return dist

    ldmat = np.zeros((q_L, s_L), dtype=np.int16)
    for row in range(1, q_L):
        ldmat[row, 0] = row * gap_penalty

    for col in range(1, s_L):
        ldmat[0, col] = col * gap_penalty
        
    for col in range(1, s_L):
        for row in range(1, q_L):
            ldmat[row, col] = min(ldmat[row-1, col] + gap_penalty,
                                 ldmat[row, col-1] + gap_penalty,
                                 ldmat[row-1, col-1] + distance_matrix[seq_vec1[row-1], seq_vec2[col-1]]) # substitution
    return ldmat[row, col]

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





@nb.jit(nopython=True)
def _nb_vector_editdistance(seqs_mat, seqs_L, indices, distance_matrix=default_distance_matrix, gap_penalty=1):
    """Computes the Levenshtein edit distance for sequences in seqs_mat indicated
    by pairs of indices.

    Parameters
    ----------
    seqs_mat : np.ndarray dtype=int16 [nseqs, seq_length]
        Created by pwsd.seqs2mat with padding to accomodate
        sequences of different lengths (-1 padding)
    seqs_L : np.ndarray [nseqs]
        A vector containing the length of each sequence,
        without the padding in seqs_mat
    indices : np.ndarray [nseqs, 2]
        Indices into seqs_mat indicating pairs of sequences to compare.
    distance_matrix : np.ndarray [alphabet, alphabet] dtype=int32
        A square distance matrix (NOT a similarity matrix).
        Matrix must match the alphabet that was used to create
        seqs_mat, where each AA is represented by an index into the alphabet.
    gap_penalty : int
        Penalty for insertions and deletions in the optimal alignment."""

    assert seqs_mat.shape[0] == seqs_L.shape[0]
    
    dist = np.zeros(indices.shape[0], dtype=np.int16)
    for ind_i in nb.prange(indices.shape[0]):
        query_i = indices[ind_i, 0]
        seq_i = indices[ind_i, 1]
        q_L = seqs_L[query_i]
        s_L = seqs_L[seq_i]
        if q_L == s_L:
            """No gaps: substitution distance
            This will make it differ from a strict edit-distance since
            the optimal edit-distance may insert same number of gaps in both sequences"""
            tmp_dist = 0
            for i in range(q_L):
                tmp_dist += distance_matrix[seqs_mat[query_i, i], seqs_mat[seq_i, i]]
            dist[ind_i] = tmp_dist
            continue
    
        ldmat = np.zeros((q_L, s_L), dtype=np.int16)
        for row in range(1, q_L):
            ldmat[row, 0] = row * gap_penalty

        for col in range(1, s_L):
            ldmat[0, col] = col * gap_penalty
            
        for col in range(1, s_L):
            for row in range(1, q_L):
                ldmat[row, col] = min(ldmat[row-1, col] + gap_penalty,
                                     ldmat[row, col-1] + gap_penalty,
                                     ldmat[row-1, col-1] + distance_matrix[seqs_mat[query_i, row-1], seqs_mat[seq_i, col-1]]) # substitution
        dist[ind_i] = ldmat[row, col]
    return dist

@nb.jit(nopython=True)
def _nb_hamming_distance(str1, str2):
    assert len(str1) == len(str2)

    tot = 0
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            tot += 1
    return tot

@nb.jit(nopython=True)
def _nb_subst_metric(seq1, seq2, subst_dict, as_similarity=False):
    """Computes sequence similarity based on the substitution matrix.
    Requires that sequences are pre-aligned and equal length.
    Operates on strings and a dict substitution matrix"""
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
