import numpy as np
import itertools
import numba as nb

from .matrices import dict_from_matrix, parasail_aa_alphabet, identity_nb_distance_matrix, tcr_nb_distance_matrix

__all__ = ['nb_hamming_distance',
           'nb_vector_hamming_distance',
           'nb_editdistance',
           'nb_vector_editdistance',
           'nb_tcrdist',
           'nb_vector_tcrdist']

@nb.jit(nopython=True, parallel=False, nogil=True)
def nb_hamming_distance(vec1, vec2, check_lengths=True):
    if check_lengths:
        assert vec1.shape[0] == vec2.shape[0]
        npos = vec1.shape[0]
    else:
        npos = min(vec1.shape[0], vec2.shape[0])

    tot = 0
    for i in range(npos):
        if vec1[i] != vec2[i]:
            tot += 1
    return tot

def nb_vector_hamming_distance(indices, seqs_mat, seqs_L, check_lengths=True):
    """Computes the hamming distance for sequences in seqs_mat indicated by pairs of indices.

    Note: this function is a wrapper of the numba function so that default arguments, and passing of
    keyword arguments is supported.
    
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
    
    Returns
    -------
    dists : np.ndarray, dtype=np.int16
        Vector of distances with length equal to indices.shape[0]"""
    return _nb_vector_hamming_distance(indices, seqs_mat, seqs_L, check_lengths)

@nb.jit(nopython=True, parallel=False, nogil=True)
def _nb_vector_hamming_distance(indices, seqs_mat, seqs_L, check_lengths=True):
    dist = np.zeros(indices.shape[0], dtype=np.int16)
    for ind_i in nb.prange(indices.shape[0]):
        query_i = indices[ind_i, 0]
        seq_i = indices[ind_i, 1]
        q_L = seqs_L[query_i]
        s_L = seqs_L[seq_i]
        if check_lengths:
            assert q_L == s_L
        for i in range(min(q_L, s_L)):
            if seqs_mat[query_i, i] != seqs_mat[seq_i, i]:
                dist[ind_i] += 1
    return dist


@nb.jit(nopython=True, parallel=False, nogil=True)
def nb_editdistance(seq_vec1, seq_vec2, distance_matrix=identity_nb_distance_matrix, gap_penalty=1):
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

@nb.jit(nopython=True, parallel=False, nogil=True)
def nb_tcrdist(seq_vec1, seq_vec2, distance_matrix=tcr_nb_distance_matrix, dist_weight=3, gap_penalty=4, ntrim=3, ctrim=2, fixed_gappos=True):
    """Compute "tcrdist" distance between two TCR CDR3 sequences. Using default weight, gap penalty, ntrim and ctrim is equivalent to the
    original distance published in Dash et al, (2017). By setting ntrim and ctrim to 0 and adjusting the dist_weight, it is also possible
    to compute the CDR1/2 loop distances which can be combined with the CDR3 distance for overall distance. See tcrdist2 package for details.

    NOTE: the same alphabet must be used to encode the sequences as integer vectors and to create the distance matrix.

    Parameters
    ----------
    seq_vec1/2 : np.ndarray [npositions, ]
        Vector of integers that encode AA symbols as an index into the distance_matrix, and associated alphabet.
        Vector can be created using seq2vec or seqs2mat functions which return dtype=np.int8
    distance_matrix : ndarray [alphabet x alphabet]
        Square symetric DISTANCE matrix with zeros along the diagonal. A similarity substitution matrix such as BLOSUM62 cannot be used here
        (see kernel2dist for a converter). Each element_ij contains the distance between the symbols at position i and j in the symbol
        alphabet that was used to create the matrix. The function make_numba_matrix can create this matrix and returns dtype=np.int16
    dist_weight : int
        Weight applied to the mismatch distances before summing with the gap penalties
    gap_penalty : int
        Distance penalty for the difference in the length of the two sequences
    ntrim/ctrim : int
        Positions trimmed off the N-terminus (0) and C-terminus (L-1) ends of the peptide sequence. These symbols will be ignored
        in the distance calculation.
    fixed_gappos : bool
        If True, insert gaps at a fixed position after the cysteine residue statring the CDR3 (typically position 6).
        If False, find the "optimal" position for inserting the gaps to make up the difference in length"""
    q_L = seq_vec1.shape[0]
    s_L = seq_vec2.shape[0]
    if q_L == s_L:
        """No gaps: substitution distance"""
        tmp_dist = 0
        for i in range(ntrim, q_L - ctrim):
            tmp_dist += distance_matrix[seq_vec1[i], seq_vec2[i]]
        return tmp_dist * dist_weight

    short_len = min(q_L, s_L)
    len_diff = abs(q_L - s_L)
    if fixed_gappos:
        """If we are not aligning, use a fixed gap position relative to the start of the CDR3
        that reflects the typically longer and more variable-length contributions to
        the CDR3 from the J than from the V. For a normal-length
        CDR3 this would be after the Cys+5 position (ie, gappos = 6; align 6 rsds on N-terminal side of CDR3).
        Use an earlier gappos if lenshort is less than 11."""
        min_gappos = min(6, 3 + (short_len - 5) // 2)
        max_gappos = min_gappos
    else:
        """The CYS and the first G of the GXG are 'aligned' in the beta sheet
        the alignment seems to continue through roughly CYS+4
        ie it's hard to see how we could have an 'insertion' within that region
        gappos=1 would be a insertion after CYS
        gappos=5 would be a insertion after CYS+4 (5 rsds before the gap)
        the full cdr3 ends at the position before the first G
        so gappos of len(shortseq)-1 would be gap right before the 'G'
        shifting this back by 4 would be analogous to what we do on the other strand, ie len(shortseq)-1-4"""
        min_gappos = 5
        max_gappos = short_len - 1 - 4
        while min_gappos > max_gappos:
            min_gappos -= 1
            max_gappos += 1
    min_dist = -1
    # min_count = -1
    for gappos in range(min_gappos, max_gappos + 1):
        tmp_dist = 0
        # tmp_count = 0
        remainder = short_len - gappos
        for n_i in range(ntrim, gappos):
            """n_i refers to position relative to N term"""
            # print (n_i, shortseq[i], longseq[i], distance_matrix[shortseq[i]+longseq[i]])
            tmp_dist += distance_matrix[seq_vec1[n_i], seq_vec2[n_i]]
            # tmp_count += 1
        #print('sequence_distance_with_gappos1:', gappos, remainder, dist[seq_i])
        for c_i in range(ctrim, remainder):
            """c_i refers to position relative to C term, counting upwards from C term"""
            tmp_dist += distance_matrix[seq_vec1[q_L - 1 - c_i], seq_vec2[s_L - 1 - c_i]]
            # tmp_count += 1
        #print('sequence_distance_with_gappos2:', gappos, remainder, dist[seq_i])
        if tmp_dist < min_dist or min_dist == -1:
            min_dist = tmp_dist
            # min_count = tmp_count
        if min_dist == 0:
            break
    """Note that weight_cdr3_region is not applied to the gap penalty"""
    return min_dist * dist_weight + len_diff * gap_penalty

def nb_vector_tcrdist(indices, seqs_mat, seqs_L, distance_matrix=tcr_nb_distance_matrix, dist_weight=3, gap_penalty=4, ntrim=3, ctrim=2, fixed_gappos=True):
    """Computes the tcrdist distance for sequences in seqs_mat indicated by pairs of indices.

    Note: to use with non-CDR3 sequences set ntrim and ctrim to 0.

    Note: this function is a wrapper of the numba function so that default arguments, and passing of
    keyword arguments is supported.

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
    distance_matrix : np.ndarray [alphabet, alphabet] dtype=int32
        A square distance matrix (NOT a similarity matrix).
        Matrix must match the alphabet that was used to create
        seqs_mat, where each AA is represented by an index into the alphabet.
    dist_weight : int
        Weight applied to the mismatch distances before summing with the gap penalties
    gap_penalty : int
        Distance penalty for the difference in the length of the two sequences
    ntrim/ctrim : int
        Positions trimmed off the N-terminus (0) and C-terminus (L-1) ends of the peptide sequence. These symbols will be ignored
        in the distance calculation.
    fixed_gappos : bool
        If True, insert gaps at a fixed position after the cysteine residue statring the CDR3 (typically position 6).
        If False, find the "optimal" position for inserting the gaps to make up the difference in length

    Returns
    -------
    dists : np.ndarray, dtype=np.int16
        Vector of distances with length equal to indices.shape[0]"""

    return _nb_vector_tcrdist(indices, seqs_mat, seqs_L, distance_matrix, dist_weight, gap_penalty, ntrim, ctrim, fixed_gappos)

@nb.jit(nopython=True, parallel=False, nogil=True)
def _nb_vector_tcrdist(indices, seqs_mat, seqs_L, distance_matrix=tcr_nb_distance_matrix, dist_weight=3, gap_penalty=4, ntrim=3, ctrim=2, fixed_gappos=True):
    """This function works OK on its own. Wrapping it with the above python function was a workaround because
    joblib and multiprocessing seem to have an issue retaining default arguments with numba functions."""
    assert seqs_mat.shape[0] == seqs_L.shape[0]

    dist = np.zeros(indices.shape[0], dtype=np.int16)
    for ind_i in nb.prange(indices.shape[0]):
        query_i = indices[ind_i, 0]
        seq_i = indices[ind_i, 1]
        q_L = seqs_L[query_i]
        s_L = seqs_L[seq_i]
        if q_L == s_L:
            """No gaps: substitution distance"""
            for i in range(ntrim, q_L - ctrim):
                dist[ind_i] += distance_matrix[seqs_mat[query_i, i], seqs_mat[seq_i, i]] * dist_weight
            continue

        short_len = min(q_L, s_L)
        len_diff = abs(q_L - s_L)
        if fixed_gappos:
            min_gappos = min(6, 3 + (short_len - 5) // 2)
            max_gappos = min_gappos
        else:
            min_gappos = 5
            max_gappos = short_len - 1 - 4
            while min_gappos > max_gappos:
                min_gappos -= 1
                max_gappos += 1
        min_dist = -1
        # min_count = -1
        for gappos in range(min_gappos, max_gappos + 1):
            tmp_dist = 0
            # tmp_count = 0
            remainder = short_len - gappos
            for n_i in range(ntrim, gappos):
                """n_i refers to position relative to N term"""
                # print (n_i, shortseq[i], longseq[i], distance_matrix[shortseq[i]+longseq[i]])
                tmp_dist += distance_matrix[seqs_mat[query_i, n_i], seqs_mat[seq_i, n_i]]
                # tmp_count += 1
            #print('sequence_distance_with_gappos1:', gappos, remainder, dist[seq_i])
            for c_i in range(ctrim, remainder):
                """c_i refers to position relative to C term, counting upwards from C term"""
                tmp_dist += distance_matrix[seqs_mat[query_i, q_L - 1 - c_i], seqs_mat[seq_i, s_L - 1 - c_i]]
                # tmp_count += 1
            #print('sequence_distance_with_gappos2:', gappos, remainder, dist[seq_i])
            if tmp_dist < min_dist or min_dist == -1:
                min_dist = tmp_dist
                # min_count = tmp_count
            if min_dist == 0:
                break
        dist[ind_i] = min_dist * dist_weight + len_diff * gap_penalty
    return dist

def nb_vector_editdistance(indices, seqs_mat, seqs_L, distance_matrix=identity_nb_distance_matrix, gap_penalty=1):
    """Computes the Levenshtein edit distance for sequences in seqs_mat indicated
    by pairs of indices.

    Note: this function is a wrapper of the numba function so that default arguments, and passing of
    keyword arguments is supported.

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
    distance_matrix : np.ndarray [alphabet, alphabet] dtype=int32
        A square distance matrix (NOT a similarity matrix).
        Matrix must match the alphabet that was used to create
        seqs_mat, where each AA is represented by an index into the alphabet.
    gap_penalty : int
        Penalty for insertions and deletions in the optimal alignment.

    Returns
    -------
    dists : np.ndarray, dtype=np.int16
        Vector of distances with length equal to indices.shape[0]"""
    #print(indices.shape)
    #print(seqs_mat.shape)
    #print(seqs_L.shape)
    return _nb_vector_editdistance(indices, seqs_mat, seqs_L, distance_matrix, gap_penalty)

@nb.jit(nopython=True, parallel=False, nogil=True)
def _nb_vector_editdistance(indices, seqs_mat, seqs_L, distance_matrix=identity_nb_distance_matrix, gap_penalty=1):
    """This function works OK on its own. Wrapping it with the above python function was a workaround because
    joblib and multiprocessing seem to have an issue retaining default arguments with numba functions."""
    assert seqs_mat.shape[0] == seqs_L.shape[0]
    mx_L = nb.int_(np.max(seqs_L))

    dist = np.zeros(indices.shape[0], dtype=np.int16)
    
    """As long as ldmat is big enough to accomodate the largest sequence
    its OK to only use part of it for the smaller sequences
    NOTE that to create a 2D array it must be created 1D and reshaped"""
    ldmat = np.zeros(mx_L * mx_L, dtype=np.int16).reshape((mx_L, mx_L))
    for ind_i in nb.prange(indices.shape[0]):
        query_i = indices[ind_i, 0]
        seq_i = indices[ind_i, 1]
        
        q_L = seqs_L[query_i]
        s_L = seqs_L[seq_i]
        if q_L == s_L:
            """No gaps: substitution distance
            This will make it differ from a strict edit-distance since
            the optimal edit-distance may insert same number of gaps in both sequences"""
            #tmp_dist = 0
            for i in range(q_L):
                dist[ind_i] += distance_matrix[seqs_mat[query_i, i], seqs_mat[seq_i, i]]
            #dist[ind_i] = tmp_dist
            continue
    
        """Do not need to re-zero each time"""
        # ldmat = np.zeros((q_L, s_L), dtype=np.int16)
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

@nb.jit(nopython=True, nogil=True)
def _nb_hamming_distance(str1, str2):
    assert len(str1) == len(str2)

    tot = 0
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            tot += 1
    return tot

@nb.jit(nopython=True, nogil=True)
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