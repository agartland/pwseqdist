import numpy as np
import numba as nb

from .matrices import dict_from_matrix, parasail_aa_alphabet, identity_nb_distance_matrix, tcr_nb_distance_matrix

__all__ = ['nb_running_editdistance',
            'nb_running_tcrdist']

"""TODO:
 - The way that nndist and neighbors is dynamically expanded when it gets full
 is not compatible with parallelization. The fact that only some of the distance
 computations return a result (d <= R) really doesn't lend itself to simple
 for-loop parallelization. These functions are best run on one CPU, with the tasks
 for multiple query sequences spread to multiple CPUs using multiprocessing,
 or ideally multithreading with shared seqs_mat memory, since numba can release the GIL"""


@nb.jit(nopython=True, parallel=False)
def nb_running_tcrdist(query_i, seqs_mat, seqs_L, radius, density_est=0.05, distance_matrix=tcr_nb_distance_matrix, dist_weight=3, gap_penalty=4, ntrim=3, ctrim=2, fixed_gappos=True):
    """Compute "tcrdist" distance between two TCR CDR3 sequences. Using default weight, gap penalty, ntrim and ctrim is equivalent to the
    original distance published in Dash et al, (2017). By setting ntrim and ctrim to 0 and adjusting the dist_weight, it is also possible
    to compute the CDR1/2 loop distances which can be combined with the CDR3 distance for overall distance. See tcrdist2 package for details.

    NOTE: the same alphabet must be used to encode the sequences as integer vectors and to create the distance matrix.

    Parameters
    ----------
    query_i : int
        Index of seqs_mat for the sequence to be compared to all other seqs in seqs_mat
    seqs_mat : np.ndarray dtype=int16 [nseqs, seq_length]
        Created by pwsd.seqs2mat with padding to accomodate
        sequences of different lengths (-1 padding)
    seqs_L : np.ndarray [nseqs]
        A vector containing the length of each sequence,
        without the padding in seqs_mat
    radius : scalar
        Maximum threshold distance at which a sequence is included in the returned indices.
    density_est : float, [0, 1]
        Estimate of the fraction of seqs that are expected to be within the radius. Used to set an initial
        size for the vector of neighbor indices. Also used to grow the vector in chunks.
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
        If False, find the "optimal" position for inserting the gaps to make up the difference in length

    Returns
    -------
    indices : np.ndarray, dtype=np.uint32
        Positional indices into seqs_mat of neighbors within radius R
    nndists : np.ndarray, dtype=np.uint32
        Distances to query seq of neighbors within radius R"""
    assert seqs_mat.shape[0] == seqs_L.shape[0]

    """Chunk size for allocating array space to hold neighbors: should be a minimum of 100 and a max of seqs_mat.shape[0]"""
    chunk_sz = min(max(int((density_est/2) * seqs_mat.shape[0]) + 1, 100), seqs_mat.shape[0])

    q_L = seqs_L[query_i]
    neighbor_count = 0
    neighbors = np.zeros(chunk_sz, dtype=np.uint32)
    nndists = np.zeros(chunk_sz, dtype=np.uint32)
    for seq_i in range(seqs_mat.shape[0]):
        s_L = seqs_L[seq_i]
        short_len = min(q_L, s_L)
        len_diff = abs(q_L - s_L)
        tot_gap_penalty = len_diff * gap_penalty

        if len_diff == 0:
            """No gaps: substitution distance"""
            tmp_dist = 0
            for i in range(ntrim, q_L - ctrim):
                tmp_dist += distance_matrix[seqs_mat[query_i, i], seqs_mat[seq_i, i]]
                """if tmp_dist > radius:
                    break"""
            if tmp_dist * dist_weight <= radius:
                neighbors[neighbor_count] = seq_i
                nndists[neighbor_count] = tmp_dist * dist_weight
                neighbor_count += 1
                if neighbor_count >= neighbors.shape[0]:
                    neighbors = np.concatenate((neighbors, np.zeros(chunk_sz, dtype=np.uint32)))
                    nndists = np.concatenate((nndists, np.zeros(chunk_sz, dtype=np.uint32)))
            continue
        elif tot_gap_penalty > radius:
            continue
        
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
        for gappos in range(min_gappos, max_gappos + 1):
            tmp_dist = 0
            remainder = short_len - gappos
            for n_i in range(ntrim, gappos):
                """n_i refers to position relative to N term"""
                tmp_dist += distance_matrix[seqs_mat[query_i, n_i], seqs_mat[seq_i, n_i]]
            if tmp_dist * dist_weight + tot_gap_penalty > radius:
                continue
            for c_i in range(ctrim, remainder):
                """c_i refers to position relative to C term, counting upwards from C term"""
                tmp_dist += distance_matrix[seqs_mat[query_i, q_L - 1 - c_i], seqs_mat[seq_i, s_L - 1 - c_i]]
            
            if tmp_dist < min_dist or min_dist == -1:
                min_dist = tmp_dist

        tot_distance = min_dist * dist_weight + tot_gap_penalty
        if tot_distance <= radius:
            neighbors[neighbor_count] = seq_i
            nndists[neighbor_count] = tot_distance
            neighbor_count += 1
            if neighbor_count >= neighbors.shape[0]:
                neighbors = np.concatenate((neighbors, np.zeros(chunk_sz, dtype=np.uint32)))
                nndists = np.concatenate((nndists, np.zeros(chunk_sz, dtype=np.uint32)))
        
    return neighbors[:neighbor_count], nndists[:neighbor_count]


@nb.jit(nopython=True, parallel=False)
def nb_running_editdistance(query_i, seqs_mat, seqs_L, radius, density_est=0.05, distance_matrix=identity_nb_distance_matrix, gap_penalty=1):
    """Computes the Levenshtein edit distance between the query sequence and sequences in seqs_mat.
    Returns a vector of positinal indices of seqs that were within the radius of the query seq and their edit distances.

    Parameters
    ----------
    query_i : int
        Index of seqs_mat for the sequence to be compared to all other seqs in seqs_mat
    seqs_mat : np.ndarray dtype=int16 [nseqs, seq_length]
        Created by pwsd.seqs2mat with padding to accomodate
        sequences of different lengths (-1 padding)
    seqs_L : np.ndarray [nseqs]
        A vector containing the length of each sequence,
        without the padding in seqs_mat
    radius : scalar
        Maximum threshold distance at which a sequence is included in the returned indices.
    density_est : float, [0, 1]
        Estimate of the fraction of seqs that are expected to be within the radius. Used to set an initial
        size for the vector of neighbor indices. Also used to grow the vector in chunks.
    distance_matrix : np.ndarray [alphabet, alphabet] dtype=int32
        A square distance matrix (NOT a similarity matrix).
        Matrix must match the alphabet that was used to create
        seqs_mat, where each AA is represented by an index into the alphabet.
    gap_penalty : int
        Penalty for insertions and deletions in the optimal alignment.

    Returns
    -------
    indices : np.ndarray, dtype=np.uint32
        Positional indices into seqs_mat of neighbors within radius R
    nndists : np.ndarray, dtype=np.uint32
        Distances to query seq of neighbors within radius R"""

    assert seqs_mat.shape[0] == seqs_L.shape[0]
    q_L = seqs_L[query_i]
    mx_L = np.max(seqs_L)

    """Chunk size for allocating array space to hold neighbors: should be a minimum of 100 and a max of seqs_mat.shape[0]"""
    chunk_sz = min(max(int((density_est/2) * seqs_mat.shape[0]) + 1, 100), seqs_mat.shape[0])

    neighbor_count = 0
    neighbors = np.zeros(chunk_sz, dtype=np.uint32)
    nndists = np.zeros(chunk_sz, dtype=np.uint32)

    """As long as ldmat is big enough to accomodate the largest sequence
    its OK to only use part of it for the smaller sequences
    NOTE that to create a 2D array it must be created 1D anfd reshaped"""
    ldmat = np.zeros(q_L * mx_L, dtype=np.int16).reshape((q_L, mx_L))
    for seq_i in range(seqs_mat.shape[0]):
        # query_i = indices[ind_i, 0]
        # seq_i = indices[ind_i, 1]
        
        s_L = seqs_L[seq_i]
        len_diff = abs(q_L - s_L)
        tot_gap_penalty = len_diff * gap_penalty

        if len_diff == 0:
            """No gaps: substitution distance
            This will make it differ from a strict edit-distance since
            the optimal edit-distance may insert same number of gaps in both sequences"""
            tmp_dist = 0
            for i in range(q_L):
                tmp_dist += distance_matrix[seqs_mat[query_i, i], seqs_mat[seq_i, i]]
            if tmp_dist <= radius:
                neighbors[neighbor_count] = seq_i
                nndists[neighbor_count] = tmp_dist
                neighbor_count += 1
                if neighbor_count >= neighbors.shape[0]:
                    neighbors = np.concatenate((neighbors, np.zeros(chunk_sz, dtype=np.uint32)))
                    nndists = np.concatenate((nndists, np.zeros(chunk_sz, dtype=np.uint32)))
            continue
        elif tot_gap_penalty > radius:
            continue
    
        """Do not need to re-zero each time"""
        # ldmat = np.zeros((q_L, s_L), dtype=np.int16)
        for row in range(1, q_L):
            ldmat[row, 0] = row * gap_penalty

        for col in range(1, s_L):
            ldmat[0, col] = col * gap_penalty
            
        BREAK = False
        for col in range(1, s_L):
            for row in range(1, q_L):
                ldmat[row, col] = min(ldmat[row-1, col] + gap_penalty,
                                     ldmat[row, col-1] + gap_penalty,
                                     ldmat[row-1, col-1] + distance_matrix[seqs_mat[query_i, row-1], seqs_mat[seq_i, col-1]]) # substitution
                if ldmat[row, col] > radius:
                    BREAK = True
                    break
            if BREAK:
                break
        if ldmat[row, col] <= radius:
            """Means that the nested loops finished withour BREAKing"""
            neighbors[neighbor_count] = seq_i
            nndists[neighbor_count] = ldmat[row, col]
            neighbor_count += 1
            if neighbor_count >= neighbors.shape[0]:
                neighbors = np.concatenate((neighbors, np.zeros(chunk_sz, dtype=np.uint32)))
                nndists = np.concatenate((nndists, np.zeros(chunk_sz, dtype=np.uint32)))
    return neighbors[:neighbor_count], nndists[:neighbor_count]