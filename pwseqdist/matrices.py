import parasail
import numpy as np
import itertools

__all__ = ['parasail_aa_alphabet',
           'parasail_aa_alphabet_with_unknown',
           'seq2vec',
           'vec2seq',
           'seqs2mat',
           'mat2seqs',
           'dict_from_matrix',
           'make_numba_matrix',
           'identity_nb_distance_matrix',
           'tcr_nb_distance_matrix']


"""Parasail uses a 23 AA alphabet for its substitution matrix.
A final column/row includes a value for any symbol not included
in the alphabet"""
parasail_aa_alphabet = 'ARNDCQEGHILKMFPSTWYVBZX'
"""Used for reconstruction of sequences from number representation"""
parasail_aa_alphabet_with_unknown = 'ARNDCQEGHILKMFPSTWYVBZX*'

identity_nb_distance_matrix = np.ones((len(parasail_aa_alphabet_with_unknown), len(parasail_aa_alphabet_with_unknown)), dtype=np.int32)
identity_nb_distance_matrix[np.diag_indices_from(identity_nb_distance_matrix)] = 0


def make_numba_matrix(distance_matrix, alphabet=parasail_aa_alphabet_with_unknown):
    """Make a numba compatible distance matrix from dict of tuples, e.g. key=('A', 'C')
    A numba compatible distance matrix is the same as a parasail compatible matrix
    if the default alphabet is used (including the dtype).

    If you have a parasail SIMILARITY matrix you'd like to use with a numba distance
    metric you can pass the matrix attribute (e.g. parasail.blosum62.matrix) directly to
    the metric as they are both formatted the same (though make sure you use the parasail
    alphabet with seqs2mat when encoding the sequences.

    However, note that metrics may want a DISTANCE matrix as opposed to a SIMILARITY
    matrix. For example, nb_editdistance uses a distance matrix with zeros along the
    diagonal, which means a BLOSUM type similarity matrix would have to be converted
    to a distance matrix (see kernel2dist below).

    Parameters
    ----------
    distance_matrix : dict
        Keys are tuples like ('A', 'C') with values containing an integer.
    alphabet : str

    Returns
    -------
    distance_matrix : np.ndarray, dtype=np.int32"""
    
    dm = np.zeros((len(alphabet), len(alphabet)), dtype=np.int32)
    for (aa1, aa2), d in distance_matrix.items():
        dm[alphabet.index(aa1), alphabet.index(aa2)] = d
        dm[alphabet.index(aa2), alphabet.index(aa1)] = d
    return dm

tcr_dict_distance_matrix = {('A', 'A'): 0,  ('A', 'C'): 4,  ('A', 'D'): 4,  ('A', 'E'): 4,  ('A', 'F'): 4,  ('A', 'G'): 4,  ('A', 'H'): 4,  ('A', 'I'): 4,  ('A', 'K'): 4,  ('A', 'L'): 4,  ('A', 'M'): 4,  ('A', 'N'): 4,  ('A', 'P'): 4,  ('A', 'Q'): 4,  ('A', 'R'): 4,  ('A', 'S'): 3,  ('A', 'T'): 4,  ('A', 'V'): 4,  ('A', 'W'): 4,  ('A', 'Y'): 4,  ('C', 'A'): 4,  ('C', 'C'): 0,  ('C', 'D'): 4,  ('C', 'E'): 4,  ('C', 'F'): 4,  ('C', 'G'): 4,  ('C', 'H'): 4,  ('C', 'I'): 4,  ('C', 'K'): 4,  ('C', 'L'): 4,  ('C', 'M'): 4,  ('C', 'N'): 4,  ('C', 'P'): 4,  ('C', 'Q'): 4,  ('C', 'R'): 4,  ('C', 'S'): 4,  ('C', 'T'): 4,  ('C', 'V'): 4,  ('C', 'W'): 4,  ('C', 'Y'): 4,  ('D', 'A'): 4,  ('D', 'C'): 4,  ('D', 'D'): 0,  ('D', 'E'): 2,  ('D', 'F'): 4,  ('D', 'G'): 4,  ('D', 'H'): 4,  ('D', 'I'): 4,  ('D', 'K'): 4,  ('D', 'L'): 4,  ('D', 'M'): 4,  ('D', 'N'): 3,  ('D', 'P'): 4,  ('D', 'Q'): 4,  ('D', 'R'): 4,  ('D', 'S'): 4,  ('D', 'T'): 4,  ('D', 'V'): 4,  ('D', 'W'): 4,  ('D', 'Y'): 4,  ('E', 'A'): 4,  ('E', 'C'): 4,  ('E', 'D'): 2,  ('E', 'E'): 0,  ('E', 'F'): 4,  ('E', 'G'): 4,  ('E', 'H'): 4,  ('E', 'I'): 4,  ('E', 'K'): 3,  ('E', 'L'): 4,  ('E', 'M'): 4,  ('E', 'N'): 4,  ('E', 'P'): 4,  ('E', 'Q'): 2,  ('E', 'R'): 4,  ('E', 'S'): 4,  ('E', 'T'): 4,  ('E', 'V'): 4,  ('E', 'W'): 4,  ('E', 'Y'): 4,  ('F', 'A'): 4,  ('F', 'C'): 4,  ('F', 'D'): 4,  ('F', 'E'): 4,  ('F', 'F'): 0,  ('F', 'G'): 4,  ('F', 'H'): 4,  ('F', 'I'): 4,  ('F', 'K'): 4,  ('F', 'L'): 4,  ('F', 'M'): 4,  ('F', 'N'): 4,  ('F', 'P'): 4,  ('F', 'Q'): 4,  ('F', 'R'): 4,  ('F', 'S'): 4,  ('F', 'T'): 4,  ('F', 'V'): 4,  ('F', 'W'): 3,  ('F', 'Y'): 1,  ('G', 'A'): 4,  ('G', 'C'): 4,  ('G', 'D'): 4,  ('G', 'E'): 4,  ('G', 'F'): 4,  ('G', 'G'): 0,  ('G', 'H'): 4,  ('G', 'I'): 4,  ('G', 'K'): 4,  ('G', 'L'): 4,  ('G', 'M'): 4,  ('G', 'N'): 4,  ('G', 'P'): 4,  ('G', 'Q'): 4,  ('G', 'R'): 4,  ('G', 'S'): 4,  ('G', 'T'): 4,  ('G', 'V'): 4,  ('G', 'W'): 4,  ('G', 'Y'): 4,  ('H', 'A'): 4,  ('H', 'C'): 4,  ('H', 'D'): 4,  ('H', 'E'): 4,  ('H', 'F'): 4,  ('H', 'G'): 4,  ('H', 'H'): 0,  ('H', 'I'): 4,  ('H', 'K'): 4,  ('H', 'L'): 4,  ('H', 'M'): 4,  ('H', 'N'): 3,  ('H', 'P'): 4,  ('H', 'Q'): 4,  ('H', 'R'): 4,  ('H', 'S'): 4,  ('H', 'T'): 4,  ('H', 'V'): 4,  ('H', 'W'): 4,  ('H', 'Y'): 2,  ('I', 'A'): 4,  ('I', 'C'): 4,  ('I', 'D'): 4,  ('I', 'E'): 4,  ('I', 'F'): 4,  ('I', 'G'): 4,  ('I', 'H'): 4,  ('I', 'I'): 0,  ('I', 'K'): 4,  ('I', 'L'): 2,  ('I', 'M'): 3,  ('I', 'N'): 4,  ('I', 'P'): 4,  ('I', 'Q'): 4,  ('I', 'R'): 4,  ('I', 'S'): 4,  ('I', 'T'): 4,  ('I', 'V'): 1,  ('I', 'W'): 4,  ('I', 'Y'): 4,  ('K', 'A'): 4,  ('K', 'C'): 4,  ('K', 'D'): 4,  ('K', 'E'): 3,  ('K', 'F'): 4,  ('K', 'G'): 4,  ('K', 'H'): 4,  ('K', 'I'): 4,  ('K', 'K'): 0,  ('K', 'L'): 4,  ('K', 'M'): 4,  ('K', 'N'): 4,  ('K', 'P'): 4,  ('K', 'Q'): 3,  ('K', 'R'): 2,  ('K', 'S'): 4,  ('K', 'T'): 4,  ('K', 'V'): 4,  ('K', 'W'): 4,  ('K', 'Y'): 4,  ('L', 'A'): 4,  ('L', 'C'): 4,  ('L', 'D'): 4,  ('L', 'E'): 4,  ('L', 'F'): 4,  ('L', 'G'): 4,  ('L', 'H'): 4,  ('L', 'I'): 2,  ('L', 'K'): 4,  ('L', 'L'): 0,  ('L', 'M'): 2,  ('L', 'N'): 4,  ('L', 'P'): 4,  ('L', 'Q'): 4,  ('L', 'R'): 4,  ('L', 'S'): 4,  ('L', 'T'): 4,  ('L', 'V'): 3,  ('L', 'W'): 4,  ('L', 'Y'): 4,  ('M', 'A'): 4,  ('M', 'C'): 4,  ('M', 'D'): 4,  ('M', 'E'): 4,  ('M', 'F'): 4,  ('M', 'G'): 4,  ('M', 'H'): 4,  ('M', 'I'): 3,  ('M', 'K'): 4,  ('M', 'L'): 2,  ('M', 'M'): 0,  ('M', 'N'): 4,  ('M', 'P'): 4,  ('M', 'Q'): 4,  ('M', 'R'): 4,  ('M', 'S'): 4,  ('M', 'T'): 4,  ('M', 'V'): 3,  ('M', 'W'): 4,  ('M', 'Y'): 4,  ('N', 'A'): 4,  ('N', 'C'): 4,  ('N', 'D'): 3,  ('N', 'E'): 4,  ('N', 'F'): 4,  ('N', 'G'): 4,  ('N', 'H'): 3,  ('N', 'I'): 4,  ('N', 'K'): 4,  ('N', 'L'): 4,  ('N', 'M'): 4,  ('N', 'N'): 0,  ('N', 'P'): 4,  ('N', 'Q'): 4,  ('N', 'R'): 4,  ('N', 'S'): 3,  ('N', 'T'): 4,  ('N', 'V'): 4,  ('N', 'W'): 4,  ('N', 'Y'): 4,  ('P', 'A'): 4,  ('P', 'C'): 4,  ('P', 'D'): 4,  ('P', 'E'): 4,  ('P', 'F'): 4,  ('P', 'G'): 4,  ('P', 'H'): 4,  ('P', 'I'): 4,  ('P', 'K'): 4,  ('P', 'L'): 4,  ('P', 'M'): 4,  ('P', 'N'): 4,  ('P', 'P'): 0,  ('P', 'Q'): 4,  ('P', 'R'): 4,  ('P', 'S'): 4,  ('P', 'T'): 4,  ('P', 'V'): 4,  ('P', 'W'): 4,  ('P', 'Y'): 4,  ('Q', 'A'): 4,  ('Q', 'C'): 4,  ('Q', 'D'): 4,  ('Q', 'E'): 2,  ('Q', 'F'): 4,  ('Q', 'G'): 4,  ('Q', 'H'): 4,  ('Q', 'I'): 4,  ('Q', 'K'): 3,  ('Q', 'L'): 4,  ('Q', 'M'): 4,  ('Q', 'N'): 4,  ('Q', 'P'): 4,  ('Q', 'Q'): 0,  ('Q', 'R'): 3,  ('Q', 'S'): 4,  ('Q', 'T'): 4,  ('Q', 'V'): 4,  ('Q', 'W'): 4,  ('Q', 'Y'): 4,  ('R', 'A'): 4,  ('R', 'C'): 4,  ('R', 'D'): 4,  ('R', 'E'): 4,  ('R', 'F'): 4,  ('R', 'G'): 4,  ('R', 'H'): 4,  ('R', 'I'): 4,  ('R', 'K'): 2,  ('R', 'L'): 4,  ('R', 'M'): 4,  ('R', 'N'): 4,  ('R', 'P'): 4,  ('R', 'Q'): 3,  ('R', 'R'): 0,  ('R', 'S'): 4,  ('R', 'T'): 4,  ('R', 'V'): 4,  ('R', 'W'): 4,  ('R', 'Y'): 4,  ('S', 'A'): 3,  ('S', 'C'): 4,  ('S', 'D'): 4,  ('S', 'E'): 4,  ('S', 'F'): 4,  ('S', 'G'): 4,  ('S', 'H'): 4,  ('S', 'I'): 4,  ('S', 'K'): 4,  ('S', 'L'): 4,  ('S', 'M'): 4,  ('S', 'N'): 3,  ('S', 'P'): 4,  ('S', 'Q'): 4,  ('S', 'R'): 4,  ('S', 'S'): 0,  ('S', 'T'): 3,  ('S', 'V'): 4,  ('S', 'W'): 4,  ('S', 'Y'): 4,  ('T', 'A'): 4,  ('T', 'C'): 4,  ('T', 'D'): 4,  ('T', 'E'): 4,  ('T', 'F'): 4,  ('T', 'G'): 4,  ('T', 'H'): 4,  ('T', 'I'): 4,  ('T', 'K'): 4,  ('T', 'L'): 4,  ('T', 'M'): 4,  ('T', 'N'): 4,  ('T', 'P'): 4,  ('T', 'Q'): 4,  ('T', 'R'): 4,  ('T', 'S'): 3,  ('T', 'T'): 0,  ('T', 'V'): 4,  ('T', 'W'): 4,  ('T', 'Y'): 4,  ('V', 'A'): 4,  ('V', 'C'): 4,  ('V', 'D'): 4,  ('V', 'E'): 4,  ('V', 'F'): 4,  ('V', 'G'): 4,  ('V', 'H'): 4,  ('V', 'I'): 1,  ('V', 'K'): 4,  ('V', 'L'): 3,  ('V', 'M'): 3,  ('V', 'N'): 4,  ('V', 'P'): 4,  ('V', 'Q'): 4,  ('V', 'R'): 4,  ('V', 'S'): 4,  ('V', 'T'): 4,  ('V', 'V'): 0,  ('V', 'W'): 4,  ('V', 'Y'): 4,  ('W', 'A'): 4,  ('W', 'C'): 4,  ('W', 'D'): 4,  ('W', 'E'): 4,  ('W', 'F'): 3,  ('W', 'G'): 4,  ('W', 'H'): 4,  ('W', 'I'): 4,  ('W', 'K'): 4,  ('W', 'L'): 4,  ('W', 'M'): 4,  ('W', 'N'): 4,  ('W', 'P'): 4,  ('W', 'Q'): 4,  ('W', 'R'): 4,  ('W', 'S'): 4,  ('W', 'T'): 4,  ('W', 'V'): 4,  ('W', 'W'): 0,  ('W', 'Y'): 2,  ('Y', 'A'): 4,  ('Y', 'C'): 4,  ('Y', 'D'): 4,  ('Y', 'E'): 4,  ('Y', 'F'): 1,  ('Y', 'G'): 4,  ('Y', 'H'): 2,  ('Y', 'I'): 4,  ('Y', 'K'): 4,  ('Y', 'L'): 4,  ('Y', 'M'): 4,  ('Y', 'N'): 4,  ('Y', 'P'): 4,  ('Y', 'Q'): 4,  ('Y', 'R'): 4,  ('Y', 'S'): 4,  ('Y', 'T'): 4,  ('Y', 'V'): 4,  ('Y', 'W'): 2,  ('Y', 'Y'): 0}
# tcr_dict_distance_matrix = {f"{x[0]}{x[1]}":d for x,d in tcr_dict_distance_matrix.items()}
tcr_nb_distance_matrix = make_numba_matrix(tcr_dict_distance_matrix)


def dist2kernel(dmat):
    """Convert a distance matrix into a similarity kernel
    for KernelPCA or kernel regression methods.

    Implementation of D2K in MiRKAT, Zhao et al., 2015

    Note that this will return a dype=np.float matrix.

    Parameters
    ----------
    dmat : ndarray shape (n,n)
        Pairwise-distance matrix.

    Returns
    -------
    kernel : ndarray shape (n,n)"""

    n = dmat.shape[0]
    I = np.identity(n)
    """m = I - dot(1,1')/n"""
    m = I - np.ones((n, n))/np.float(n)
    kern = -0.5 * np.linalg.multi_dot((m, dmat**2, m))

    if isinstance(dmat, pd.DataFrame):
        return pd.DataFrame(kern, index=dmat.index, columns=dmat.columns)
    else:
        return kern

def kernel2dist(kern):
    """Convert a similarity kernel into a distance matrix.

    Implementation of K2D in MiRKAT, Zhao et al., 2015

    Also reccommended by:
    E. Halperin, J. Buhler, R. Karp, R. Krauthgamer, B. Westover.
    Detecting protein sequence conservation via metric embeddings.
    Bioinformatics 19, 122-129 (2003).

    d_ij^2 = K_ii + K_jj - 2K_ij

    Note that this will return a dype=np.float matrix. If an integer
    dtype is needed consider multiplying by a factor to preserve
    decimals before rounding and casting as integer dtype
    (e.g. np.int16 for the numba metrics)

    Parameters
    ----------
    kernel : ndarray shape (n,n)

    Returns
    -------
    dmat : ndarray shape (n,n)"""
    dmat = np.sqrt(np.diag(kern)[None,:] + np.diag(kern)[:, None] - 2 * kern)
    return dmat

def dict_from_matrix(mat, alphabet=parasail_aa_alphabet):
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

def seq2vec(seq, alphabet=parasail_aa_alphabet_with_unknown, length=None):
    """
    Convert AA string sequence into numpy vector of integers

    Parameters
    ----------
    seq : string

    alphabet : str
        string of characters usely specifying the 23 AA Parasail 
        uses for its substitution matrix.

    length : None or int
        If an integer, the length of the vector to return,
        with padding (-1) added after last AA of seq if needed

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
    if length is None:
        length = len(seq)
    vec = -1 * np.ones(length, dtype=np.int8)
    for aai in range(length):
        if aai >= len(seq):
            break
        try:
            vec[aai] = alphabet.index(seq[aai])
        except ValueError:
            """Unknown symbols given value for last column/row of matrix"""
            vec[aai] = len(alphabet)
    return vec

def vec2seq(vec, alphabet=parasail_aa_alphabet_with_unknown, unknown='X'):
    """Convert a numpy array of integers back into a AA string sequence.
    (opposite of seq2vec())
    
    Parameters
    ----------
    vec : np.array
    alphabet : str
        String of characters for specifying the integer encoding
        of each symbol, based on its position in the alphabet
    unknown : char
        Symbol used for integer codes that are beyond the length of the alphabet

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
        seq =  ''.join([alphabet[aai] if aai < len(alphabet) else unknown for aai in vec])
    return seq

def seqs2mat(seqs, alphabet=parasail_aa_alphabet, max_len=None):
    """Convert a collection of AA sequences into a
    numpy matrix of integers for fast comparison.

    Parameters
    ----------
    seqs : list 
        List of strings

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
    if max_len is None:
        max_len = np.max([len(s) for s in seqs])
    mat = -1 * np.ones((len(seqs), max_len), dtype=np.int8)
    L = np.zeros(len(seqs), dtype=np.int8)
    for si, s in enumerate(seqs):
        L[si] = len(s)
        for aai in range(max_len):
            if aai >= len(s):
                break
            try:
                mat[si, aai] = alphabet.index(s[aai])
            except ValueError:
                """Unknown symbols given value for last column/row of matrix"""
                mat[si, aai] = len(alphabet)
    return mat, L

def validate_seqs(seqs, alphabet=parasail_aa_alphabet_with_unknown, unknown=None):
    """Check each sequence for an unknown symbol (not in the alphabet)
    and replace it with either the last symbol in the alphabet (unknown = None),
    or the symbol in unknown.

    NOTE: decided not to write this function after all because it will be very slow
    for a large number of sequences and is no needed because all the distance functions
    currently have a way of handling unknown symbols. 

    Parameters
    ----------
    seqs : list of strings
    alphabet : str
        String of characters for specifying the integer encoding
        of each symbol, based on its position in the alphabet
    unknown : char
        Symbol used to replace symbols not in alphabet
        Optionally, None means that the last symbol of the alphabet
        is used to represent unknowns

    Returns
    -------
    seqs : list of strings
        Same as input seqs, except unknown symbols have been replaced"""

    pass


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
