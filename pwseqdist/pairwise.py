import multiprocessing
import parmap
import itertools
import numpy as np
import numba as nb
import scipy

from .metrics import compute_many, compute_many_rect
from .numba_tools import *
from .matrices import seqs2mat

__all__ = ['apply_pairwise_rect',
           'apply_pairwise_sparse']

"""TODO:
Currently I pass all the sequences and some set of indices to compute_many. 
Why wouldn't I just send some of the sequences?
The point was to avoid sending all the pairs of sequences and just send
pairs of indices to the workers. So we'd have to be smart about reducing the
total number of sequences that are needed and then sending just those and
translated indices.

"""

def apply_pairwise_rect(seqs1, seqs2, metric, ncpus=1, use_numba=False, uniqify=True, *numba_args, **kwargs):
    """Calculate distance between pairs of sequences in seqs1
    with sequences in seqs2 using metric and kwargs provided to
    metric. Will use multiprocessing Pool if ncpus > 1.

    For efficiency, will only compute metric on unique values in
    seqs1/seqs2. All values are returned, including redundancies.

    Though written to be used for distance calculations,
    it is general enough that it could be used to run
    any arbitrary function on pairs of elements in seqs.

    Parameters
    ----------
    seqs1, seqs2 : lists
        Lists of sequences.
    metric : function
        A distance function of the form
        func(seq1, seq2, **kwargs)
    ncpus : int
        Size of the worker pool to be used by multiprocessing
    use_numba : bool
        Use a numba-compiled outer loop and distance metric.
        For numba, ncpus is ignored because the "outer" loop
        has been compiled with parallel=True.
    uniqify : bool
        Indicates whether only unique sequences should be analyzed.
    **kwargs : keyword arguments
        Additional keyword arguments are supplied to the metric.
        Kwargs are not provided to numba-compiled metrics; use numba_args.
    *numba_args : non-keyword arguments
        These are provided to numba-compiled metrics which do not
        accept kwargs. Use kwargs for non-numba metrics.

    Returns
    -------
    dvec : np.ndarray, length len(seqs1) * len(seqs2)
        Vector form of the pairwise distance rectangle.
    indices : np.ndarray, shape [len(seqs1) * len(seqs2), 2]
        Contains i,j indices on each row where i (j) is an index
        into seqs1 (seqs2) and can be used to recreate a distance rectangle"""

    def _recti2veci(i, j, n2):
        """Translate from rectangle coordinates to vector coordinates"""
        return int(i * len(n2) + j)

    useqs1, seqs1_uind = np.unique(seqs1, return_inverse=True)
    if len(useqs1) == len(seqs1) or not uniqify:
        useqs1 = seqs1
        translate1 = False
    else:
        translate1 = True

    useqs2, seqs2_uind = np.unique(seqs2, return_inverse=True)
    if len(useqs2) == len(seqs2) or not uniqify:
        useqs2 = seqs2
        translate2 = False
    else:
        translate2 = True

    pw_indices = list(itertools.product(range(len(useqs1)), range(len(useqs2))))
    if not use_numba:
        chunk_func = lambda l, n: [l[i:i + n] for i in range(0, len(l), n)]
        chunksz = len(pw_indices)//ncpus
        chunked_indices = chunk_func(pw_indices, chunksz)
        dtype = type(metric(useqs1[0], useqs2[0], **kwargs))
        if ncpus > 1:
            with multiprocessing.Pool(ncpus) as pool:
                try:
                    dists = parmap.map(compute_many_rect,
                                       chunked_indices,
                                       metric,
                                       useqs1,
                                       useqs2,
                                       dtype,
                                       **kwargs,
                                       pm_parallel=True,
                                       pm_pool=pool)
                except ValueError as err:
                    print('pwseqdist.apply_pairwise_rect: error with metric %s and multiprocessing, trying on single core' % metric)
                    dists = parmap.map(compute_many_rect,
                                       chunked_indices,
                                       metric,
                                       useqs1,
                                       useqs2,
                                       dtype,
                                       **kwargs,
                                       pm_parallel=False)
                    print('pwseqdist.apply_pairwise_rect: metric %s could not be spread to multiple processes, ran on single core' % metric)
        else:
            dists = parmap.map(compute_many_rect,
                                   chunked_indices,
                                   metric,
                                   useqs1,
                                   useqs2,
                                   dtype,
                                   **kwargs,
                                   pm_parallel=False)
        
        urect = np.concatenate(dists).reshape((len(useqs1), len(useqs2)))
    else:
        pw_indices = np.array(pw_indices, dtype=np.int64)
        seqs_mat1, seqs_L1 = seqs2mat(useqs1)
        seqs_mat2, seqs_L2 = seqs2mat(useqs2)
        urect = nb_distance_rect(seqs_mat1, seqs_L1, seqs_mat2, seqs_L2, pw_indices, metric, *numba_args).reshape((len(useqs1), len(useqs2)))
    if translate1:
        urect = urect[seqs1_uind, :]
    if translate2:
        urect = urect[:, seqs2_uind]
    return urect


def apply_pairwise_sparse(seqs, pairs, metric, ncpus=1, use_numba=False, *numba_args, **kwargs):
    """Calculate distance between pairs of sequences in seqs using metric and kwargs
    provided to metric. Will only compute distances specified in pairs of indices in pairs.
    Results could be used to create a sparse matrix of pairwise distances.

    Will use multiprocessing Pool if ncpus > 1.

    Though written to be used for distance calculations,
    it is general enough that it could be used to run
    any arbitrary function on pairs of elements in seqs (iterable).

    Parameters
    ----------
    seqs : list or indexable iterable
        List of sequences.
    pairs : iterable
        List or iterable of length 2 tuples/lists, where each length 2 list
        is a pair of integer positional indices into seqs.
    metric : function
        A distance function of the form
        func(seq1, seq2, **kwargs)
    ncpus : int
        Size of the worker pool to be used by multiprocessing
    use_numba : bool
        Use a numba-compiled outer loop and distance metric.
        For numba, ncpus is ignored because the "outer" loop
        has been compiled with parallel=True.
    **kwargs : keyword arguments
        Additional keyword arguments are supplied to the metric.
        Kwargs are not provided to numba-compiled metrics; use numba_args.
    *numba_args : non-keyword arguments
        These are provided to numba-compiled metrics which do not
        accept kwargs. Use kwargs for non-numba metrics.

    Returns
    -------
    dvec : np.ndarray, length len(pairs)
        Vector of distances for each pair of indices in pairs"""
    
    if not use_numba:
        chunk_func = lambda l, n: [l[i:i + n] for i in range(0, len(l), n)]
        chunksz = len(pairs)//ncpus
        chunked_indices = chunk_func(pairs, chunksz)
        dtype = type(metric(seqs[0], seqs[0], **kwargs))

        """compute_many(indices, metric, seqs, dtype, **kwargs)"""

        if ncpus > 1:
            with multiprocessing.Pool(ncpus) as pool:
                try:
                    dists = parmap.map(compute_many,
                                       chunked_indices,
                                       metric,
                                       seqs,
                                       dtype,
                                       **kwargs,
                                       pm_parallel=True,
                                       pm_pool=pool)
                except ValueError as err:
                    print('pwseqdist.apply_pairwise_sparse: error with metric %s and multiprocessing, trying on single core' % metric)
                    dists = parmap.map(compute_many,
                                       chunked_indices,
                                       metric,
                                       seqs,
                                       dtype,
                                       **kwargs,
                                       pm_parallel=False)
                    print('pwseqdist.apply_pairwise_sparse: metric %s could not be spread to multiple processes, ran on single core' % metric)
        else:
            dists = parmap.map(compute_many,
                                   chunked_indices,
                                   metric,
                                   seqs,
                                   dtype,
                                   **kwargs,
                                   pm_parallel=False)
        
        vec = np.concatenate(dists)
    else:
        """nb_distance_vec(seqs_mat, seqs_L, indices, nb_metric, *args)"""
        pw_indices = np.array(pairs, dtype=np.int64)
        seqs_mat, seqs_L = seqs2mat(seqs)
        vec = nb_distance_vec(seqs_mat, seqs_L, pw_indices, metric, *numba_args)

    return urect


'''
Abondoned apply_pairwise_sq because it was slower than apply_pairwise_rect and didn't save
that much memory

@nb.jit(nb.types.intp(nb.types.intp, nb.types.intp), nopython=True)
def nb_comb(N, k):
    """
    Numba jitted function that computes N choose k. Return `0` if the
    outcome exceeds the maximum value of `np.intp` or if N < 0, k < 0,
    or k > N.

    Parameters
    ----------
    N : scalar(int)

    k : scalar(int)

    Returns
    -------
    val : scalar(int)

    """
    # From scipy.special._comb_int_long
    # github.com/scipy/scipy/blob/v1.0.0/scipy/special/_comb.pyx
    
    INTP_MAX = np.iinfo(np.intp).max
    if N < 0 or k < 0 or k > N:
        return 0
    if k == 0:
        return 1
    if k == 1:
        return N
    if N == INTP_MAX:
        val = 0

    M = N + 1
    nterms = min(k, N - k)

    val = 1

    for j in range(1, nterms + 1):
        # Overflow check
        if val > (INTP_MAX // (M - j)):
            val = 0
            break

        val *= M - j
        val //= j
    
    if val != 0:
        return val
    
    M = N + 1
    nterms = min(k, N - k)

    numerator = 1
    denominator = 1
    for j in range(1, nterms + 1):
        numerator *= M - j
        denominator *= j

    val = numerator // denominator
    if val == 0 and nb_comb(N, k) != 0:
        print('comb0_Nk', N, k, '\n\t', numerator, denominator)
        raise ValueError
    return val

def _mati2veci(i, j, n):
    veci = scipy.special.comb(n, 2) - scipy.special.comb(n - i, 2) + (j - i - 1)
    return int(veci)

def apply_pairwise_sq(seqs, metric, ncpus=1, use_numba=False, uniqify=True, *numba_args, **kwargs):
    """Calculate distance between all pairs of seqs using metric
    and kwargs provided to metric. Will use multiprocessing Pool
    if ncpus > 1.

    For efficiency, will only compute metric on unique values in
    seqs. All values are returned, including redundancies.

    Though written to be used for distance calculations,
    it is general enough that it could be used to run
    any arbitrary function on pairs of elements in seqs.

    Parameters
    ----------
    seqs : list
        List of sequences provided to metric in pairs.
    metric : function
        A distance function of the form
        func(seq1, seq2, **kwargs)
    ncpus : int
        Size of the worker pool to be used by multiprocessing
    use_numba : bool
        Use a numba-compiled outer loop and distance metric.
        For numba, ncpus is ignored because the "outer" loop
        has been compiled with parallel=True.
    uniqify : bool
        Indicates whether only unique sequences should be analyzed.
    **kwargs : keyword arguments
        Additional keyword arguments are supplied to the metric.
        Kwargs are not provided to numba-compiled metrics; use numba_args.
    *numba_args : non-keyword arguments
        These are provided to numba-compiled metrics which do not
        accept kwargs. Use kwargs for non-numba metrics.

    Returns
    -------
    dvec : np.ndarray, length n*(n - 1) / 2
        Vector form of the pairwise distance matrix.
        Use scipy.distance.squareform to convert to a square matrix"""
    """Set to false to turn on computation of redundant distances"""
    useqs, seqs_uind = np.unique(seqs, return_inverse=True)
    if len(useqs) == len(seqs) or not uniqify:
        useqs = seqs
        translate = False
    else:
        translate = True

    """itertools.combinations creates the i,j pairs in the same order
    as scipy.distance.pdist/squareform"""
    pw_indices = list(itertools.combinations(range(len(useqs)), 2))

    if not use_numba:
        chunk_func = lambda l, n: [l[i:i + n] for i in range(0, len(l), n)]
        chunksz = max(len(pw_indices) // ncpus, 1)
        chunked_indices = chunk_func(pw_indices, chunksz)
        dtype = type(metric(useqs[0], useqs[0], **kwargs))

        if ncpus > 1:
            with multiprocessing.Pool(ncpus) as pool:
                try:
                    dists = parmap.map(compute_many,
                                       chunked_indices,
                                       metric,
                                       useqs,
                                       dtype,
                                       **kwargs,
                                       pm_parallel=True,
                                       pm_pool=pool)
                except ValueError as err:
                    print('pwseqdist.apply_pairwise_sq: error with metric %s and multiprocessing, trying on single core' % metric)
                    dists = parmap.map(compute_many,
                                       chunked_indices,
                                       metric,
                                       useqs,
                                       dtype,
                                       **kwargs,
                                       pm_parallel=False)
                    print('pwseqdist.apply_pairwise_sq: metric %s could not be spread to multiple processes, ran on single core' % metric)
        else:
            #print(chunked_indices)
            #print(useqs)
            dists = parmap.map(compute_many,
                                   chunked_indices,
                                   metric,
                                   useqs,
                                   dtype,
                                   **kwargs,
                                   pm_parallel=False)
        """Get the vector form of the useqs"""
        """n = len(useqs)
        uvec = np.zeros(scipy.special.comb(n, 2))
        for ichunk, dchunk in zip(chunked_indices, dists):
            for (i,j), d in zip(ichunk, dchunk):
                uvec[_mati2veci(i, j, n)] = d"""
        uvec = np.concatenate(dists) # this may be more memory intensive, but should be fine
        #print(uvec)
    else:
        pw_indices = np.array(pw_indices, dtype=np.int64)
        seqs_mat, seqs_L = seqs2mat(useqs)
        uvec = nb_distance_vec(seqs_mat, seqs_L, pw_indices, metric, *numba_args)
    
    """Create translation dict from vector_i to mat_ij coordinates for the
    distance matrix of unique seqs (unneccessary, but may be useful later)"""
    # uveci2umati = {veci:(i, j) for veci, (i, j) in enumerate(itertools.combinations(range(len(useqs)), 2))}
    
    if translate:
        """Then translate the vector form of the useqs to the vector
        form of the seqs"""
        indices = nb_explode(len(useqs),
                             np.array([ij for ij in itertools.combinations(range(len(seqs)), 2)], dtype=np.int64),
                             seqs_uind)
        vout = uvec[indices]
        vout[indices == -1] = 0

        """vout = np.zeros(int(scipy.special.comb(len(seqs), 2)))
        for veci, (i,j), in enumerate(itertools.combinations(range(len(seqs)), 2)):
            #ui = useqs.index(seqs[i])
            #uj = useqs.index(seqs[j])
            ui = seqs_uind[i]
            uj = seqs_uind[j]
            if ui == uj:
                vout[veci] = 0
            else:
                if uj < ui:
                    uj, ui = ui, uj
                # vout[veci] = uvec[_mati2veci(ui, uj, len(useqs))]
                vout[veci] = uvec[nb_mati2veci(ui, uj, len(useqs))]
                #assert vout[veci] == uvec[nb_mati2veci(ui, uj, len(useqs))]
        
        assert np.all(vout == vout2)"""
    else:
        vout = uvec
    return vout

@nb.jit(nopython=True)
def nb_mati2veci(i, j, n):
    veci = nb_comb(n, 2) - nb_comb(n - i, 2) + (j - i - 1)
    return veci

@nb.jit(nopython=True)
def nb_explode(n_uvec, ijs, seqs_uind):
    indices = np.zeros(ijs.shape[0], dtype=np.int64)
    for veci in range(ijs.shape[0]):
        ui = seqs_uind[ijs[veci, 0]]
        uj = seqs_uind[ijs[veci, 1]]
        if ui == uj:
            indices[veci] = -1
        else:
            if uj < ui:
                uj, ui = ui, uj
            uveci = nb_comb(n_uvec, 2) - nb_comb(n_uvec - ui, 2) + (uj - ui - 1)
            indices[veci] = uveci
    return indices
'''