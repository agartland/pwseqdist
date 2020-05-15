import parasail
import numpy as np
import itertools

__all__ = ['parasail_aa_alphabet',
           'seq2vec',
           'vec2seq',
           'seqs2mat',
           'mat2seqs',
           'dict_from_matrix']

"""Parasail uses a 23 AA alphabet for its substitution matrix.
A final column/row includes a value for any symbol not included
in the alphabet"""
parasail_aa_alphabet = 'ARNDCQEGHILKMFPSTWYVBZX'
"""Used for reconstruction of sequences from number representation"""
parasail_aa_alphabet_with_unknown = 'ARNDCQEGHILKMFPSTWYVBZX*'

def dict_from_matrix(mat):
    """Converts substitution matrix to a dict for use
    by str_subst_metric. Does not store (aa1, aa2) and
    (aa2, aa1), so user needs to check for both. Similarity
    for any symbol not present in the aplhabet is stored in
    key "na" 

    Parameters
    ----------
    mat : parasail substitution Matrix object
        Example is parasail.blosum62"""
    d = {(aa1, aa2):mat.matrix[parasail_aa_alphabet.index(aa1), parasail_aa_alphabet.index(aa2)] for aa1, aa2 in itertools.product(parasail_aa_alphabet, parasail_aa_alphabet)}
    d.update({'na':mat.matrix[-1, 0]})
    return d

def seq2vec(seq):
    """Convert AA string sequence into numpy vector of integers"""
    vec = np.zeros(len(seq), dtype=np.int8)
    for aai, aa in enumerate(seq):
        try:
            vec[aai] = parasail_aa_alphabet.index(aa)
        except ValueError:
            """Unknown symbols given value for last column/row of matrix"""
            vec[aai] = len(parasail_aa_alphabet)
    return vec

def vec2seq(vec):
    """Convert a numpy array of integers back into a AA string sequence.
    (opposite of seq2vec())"""
    return ''.join([parasail_aa_alphabet_with_unknown[aai] for aai in vec])

def seqs2mat(seqs):
    """Convert a collection of AA sequences into a
    numpy matrix of integers for fast comparison.

    Requires all seqs to have the same length."""
    L1 = len(seqs[0])
    mat = np.zeros((len(seqs), L1), dtype=np.int8)
    for si, s in enumerate(seqs):
        msg = "All sequences must have the same length: L1 = %d, but L%d = %d" % (L1, si, len(s))
        assert L1 == len(s), msg
        mat[si, :] = seq2vec(s)
    return mat

def mat2seqs(mat):
    """Convert a matrix of integers into AA sequences."""
    return [vec2seq(mat[i,:]) for i in range(mat.shape[0])]