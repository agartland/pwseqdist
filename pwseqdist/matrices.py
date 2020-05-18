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


def dict_from_matrix(mat, alphabet = parasail_aa_alphabet):
    """
    Converts substitution matrix to a dict for use
    by str_subst_metric. Does not store (aa1, aa2) and
    (aa2, aa1), so user needs to check for both. Similarity
    for any symbol not present in the aplhabet is stored in
    key "na" 

    Parameters
    ----------
    mat : parasail substitution Matrix object
        Example is parasail.
    alphabet : str
        string of characters usely specifying the 23 AA Parasail 
        uses for its substitution matrix.

    Returns
    -------
    d : dict
    """
    d = {(aa1, aa2):mat.matrix[alphabet.index(aa1), alphabet.index(aa2)] for aa1, aa2 in itertools.product(alphabet, alphabet)}
    d.update({'na':mat.matrix[-1, 0]})
    return d

def seq2vec(seq, alphabet = parasail_aa_alphabet_with_unknown ):
    """
    Convert AA string sequence into numpy vector of integers

    Parameters
    ----------
    seq : string

    alphabet : str
        string of characters usely specifying the 23 AA Parasail 
        uses for its substitution matrix.

    Returns
    -------
    vec : np.ndarray

    Examples 
    -------_
    >>> seq2vec("CAT")
    array([ 4,  0, 16], dtype=int8)
    >>> seq2vec("ARNDCQEGHILKMFPSTWYVBZX")
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22], dtype=int8)
    """
    vec = np.zeros(len(seq), dtype=np.int8)
    for aai, aa in enumerate(seq):
        try:
            vec[aai] =alphabet.index(aa)
        except ValueError("Unknown symbols given value for last column/row of matrix"):
            """Unknown symbols given value for last column/row of matrix"""
            vec[aai] = len(parasail_aa_alphabet)
    return vec

def vec2seq(vec, alphabet = parasail_aa_alphabet_with_unknown  ):
    """
    Convert a numpy array of integers back into a AA string sequence.
    (opposite of seq2vec())
    
    Parameters
    ----------
    vec : np.array

    alphabet : str
        string of characters usely specifying the 23 AA Parasail 
        uses for its substitution matrix.

    Returns
    -------
    seq : string

    Examples 
    --------
    >>> vec2seq([ 4,  0, 16])
    'CAT'
    """
    try:
        seq =  ''.join([alphabet[aai] for aai in vec])
    except IndexError:
        max_int = len(alphabet)-1
        raise ValueError(f"vex2seq only works with integers 0 to {max_int} corresponding with {alphabet}")
    return seq

def seqs2mat(seqs):
    """
    Convert a collection of AA sequences into a
    numpy matrix of integers for fast comparison.

    Parameters
    ----------
    seqs : list 
        list of strings of equal length

    Returns
    -------
    mat : np.array  

    Examples
    --------
    >>> seqs2mat(["CAT","HAT"])
    array([[ 4,  0, 16],
           [ 8,  0, 16]], dtype=int8)

    Notes
    -----

    Requires all seqs to have the same length.

    To avoid confusion, mat in this context is an np.array 
    wherase mat in dict_from_matrix() is a parasail substitution Matrix object
    """
    L1 = len(seqs[0])
    mat = np.zeros((len(seqs), L1), dtype=np.int8)
    for si, s in enumerate(seqs):
        msg = "All sequences must have the same length: L1 = %d, but L%d = %d" % (L1, si, len(s))
        assert L1 == len(s), msg
        mat[si, :] = seq2vec(s)
    return mat

def mat2seqs(mat):
    """
    Convert a matrix of integers into AA sequences.
    
    Parameters
    ----------
    mat : np.array 
        2D np.array of integers

    Returns 
    -------
    seqs : list
        list of strings
    
    Examples 
    --------
    >>> mat2seqs(np.array([[ 4,  0, 16],[ 8,  0, 16]] ))
    ['CAT', 'HAT']
    """
    seqs = [vec2seq(mat[i,:]) for i in range(mat.shape[0])]
    return seqs
