import parasail
import numpy as np
import operator
from .nb_metrics import nb_editdistance, nb_vector_editdistance, nb_tcrdist_distance, nb_vector_tcrdist_distance, nb_hamming_distance

__all__ = ['compute_many',
           'compute_many_rect',
           'str_subst_metric',
           'hamming_distance',
           'nw_metric',
           'nw_hamming_metric',
           'nb_hamming_distance',
           'nb_editdistance',
           'nb_vector_editdistance',
           'nb_tcrdist_distance',
           'nb_vector_tcrdist_distance']

def compute_many(indices, metric, seqs, dtype, **kwargs):
    return  np.array([metric(seqs[i], seqs[j], **kwargs) for i,j in indices], dtype=dtype)

def compute_many_rect(indices, metric, seqs1, seqs2, dtype, **kwargs):
    return np.array([metric(seqs1[i], seqs2[j], **kwargs) for i,j in indices], dtype=dtype)

def _str_sim(s1, s2, subst, na_penalty):
    a = np.array([i for i in map(lambda a, b: subst.get((a, b), subst.get((b, a), na_penalty)), s1, s2)])
    return np.sum(a)

def str_subst_metric(seq1, seq2, subst_dict, as_similarity=False, na_penalty=None):
    """Distance between seq1 and seq2 using a substitution similarity matrix.
    Requires that seq1 and seq2 have the same length (see nw_metric for alternative)
    
    Distance is a function of these similarities:
    
        D = sim11 + sim22 - 2 * sim12

    Parameters
    ----------
    seq1: string
        string containing amino acid letters
    seq2: string
        string containing amino acid letters
    subst_dict : dict
        A substitution dictionary with keys (AA1, AA2).
        Can be created from any parasail matrix by calling
        pwseqdist.matrices.dict_from_matrix,
        e.g.  subst_dict = dict_from_matrix(parasail.blosum62)
    as_similarity : bool
        Returns values as summed raw, similarities as opposed to distances.
    na_penalty : int
        Optionally provide a penalty for any AA symbols not in the parasail
        alphabet: ARNDCQEGHILKMFPSTWYVBZX. This value is also coded in 
        matrices that are translated from parasail.

    Returns
    -------
    D : float
        distance between strings"""
    msg = "len of seq1 (%d) and seq2 (%d) are different" % (len(seq1), len(seq2))
    assert len(seq1) == len(seq2), msg
    
    if na_penalty is None:
        na_penalty = subst_dict['na']

    """Site-wise similarity between seq1 and seq2 using the substitution matrix subst"""
    sim12 = _str_sim(seq1, seq2, subst_dict, na_penalty)

    if as_similarity:
        return sim12
    else:
        L = len(seq1)
        sim11 = _str_sim(seq1, seq1, subst_dict, na_penalty)
        sim22 = _str_sim(seq2, seq2, subst_dict, na_penalty)
        D = sim11 + sim22 - 2 * sim12
        return D

def hamming_distance(s1, s2, check_lengths=False):
    """Hamming distance between str1 and str2.
    Requires that str1 and sr2 are equal length

    Parameters
    ----------
    s1: string
        string containing amino acid letters
    s2: string
        string containing amino acid letters

    Returns
    -------
    D : float
        distance between strings (Hamming Distance: number of mismatched positions)"""
    if check_lengths:
        assert len(s1) == len(s2), "Inputs must have the same length."
    return np.sum([i for i in map(operator.__ne__, s1, s2)])

def nw_metric(s1, s2, matrix='blosum62', open=3, extend=3, return_similarity=False):
    """Function applying Parasail's Needleman-Wuncsh Algorithm to compute
    a distance between any two sequences.

    Parameters
    ----------
    s1: string
        string containing amino acid letters
    s2: string
        string containing amino acid letters
    matrix : str
        Attribute of parasail that names a substitution matrix

    Returns
    -------
    D : float
        distance via reciprocal alignment scores.

    Notes
    -----

    .. code-block:: python

      xx = parasail.nw_stats(s1, s1, open=open, extend=extend, matrix=matrix).score
      yy = parasail.nw_stats(s2, s2, open=open, extend=extend, matrix=matrix).score
      xy = parasail.nw_stats(s1, s2, open=open, extend=extend, matrix=matrix).score
      D = xx + yy - 2 * xy
      return D


    May or may not produce a true metric. Details in:
    E. Halpering, J. Buhler, R. Karp, R. Krauthgamer, and B. Westover.
    Detecting protein sequence conservation via metric embeddings.
    Bioinformatics, 19 (sup 1) 2003
    """
    p_matrix = getattr(parasail, matrix)

    xy = parasail.nw_stats(s1, s2, open=open, extend=extend, matrix=p_matrix).score
    
    if return_similarity:
        return xy
    else:
        xx = parasail.nw_stats(s1, s1, open=open, extend=extend, matrix=p_matrix).score
        yy = parasail.nw_stats(s2, s2, open=open, extend=extend, matrix=p_matrix).score
        
        D = xx + yy - 2 * xy
        return D

def nw_hamming_metric(s1, s2, matrix='blosum62', open=3, extend=3):
    """Function applying Parasail's Needleman-Wuncsh Algorithm to align and
    compute a Hamming Distance between any two sequences: number of
    mismatched positions. Gaps count as a mismatch. Penalties and matrix
    are used for alignment purposes, not in the distance calculation.

    Parameters
    ----------
    s1: string
        string containing amino acid letters
    s2: string
        string containing amino acid letters
    matrix : str
        Attribute of parasail that names a substitution matrix

    Returns
    -------
    D : float
        distance between strings (Hamming Distance: number of mismatched positions)

    Notes
    -----
    .. code-block:: python

        xy = parasail.nw_stats(s1, s2, open=open, extend=extend, matrix=matrix)
        xy_t = parasail.nw_trace(s1, s2, open=open, extend=extend, matrix=matrix)
        hamming_distance = len(xy_t.traceback.comp)-xy.matches
        return hamming_distance"""
    p_matrix = getattr(parasail, matrix)
    xy = parasail.nw_stats(s1, s2, open=open, extend=extend, matrix=p_matrix)
    xy_t = parasail.nw_trace(s1, s2, open=open, extend=extend, matrix=p_matrix)
    D = len(xy_t.traceback.comp)-xy.matches
    return D

def _np_subst_metric(seq1, seq2, subst_mat, as_similarity=False):
    """Numpy version of str_subst_metric, however it is not yet compatible with
    apply_pairwise_sq because of the need to convert strings to vectors of integers.
    It may not be faster than the string version, so there is no plan to further develop."""
    """Computes sequence similarity or distance based on the substitution matrix."""
    msg = "Sequences must be the same length (%d != %d)." % (seq1.shape[0], seq2.shape[0])
    assert seq1.shape[0] == seq2.shape[0], msg

    """Similarity between seq1 and seq2 using the substitution matrix subst"""
    sim12 = np.sum(subst_mat[seq1, seq2])
    if as_similarity:
        return sim12
    else:
        L = seq1.shape[0]
        sim11 = np.sum(subst_mat[seq1, seq1])
        sim22 = np.sum(subst_mat[seq2, seq2])
        D = sim11 + sim22 - 2 * sim12
        return D