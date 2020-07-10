import itertools
import numpy as np
import numba as nb
import scipy
import scipy.spatial.distance
import multiprocessing
import parmap
# from joblib import Parallel, delayed
# import concurrent.futures

from .metrics import compute_many
from .numba_tools import nb_distance_vec
from .matrices import seqs2mat

__all__ = ['apply_pairwise_rect',
           'apply_pairwise_sparse']

"""TODO:
Currently I pass all the sequences and some set of indices to compute_many. 
Why wouldn't I just send some of the sequences?
The point was to avoid sending all the pairs of sequences and just send
pairs of indices to the workers. So we'd have to be smart about reducing the
total number of sequences that are needed and then sending just those and
translated indices."""


def apply_pairwise_rect(metric, seqs1, seqs2=None, ncpus=1, use_numba=False, uniqify=True, numba_args=(), **kwargs):
    """Calculate distance between pairs of sequences in seqs1
    with sequences in seqs2 using metric and kwargs provided to
    metric.

    Can provide a number compiled metric to increase speed. Note that the numba metric should accept a different
    set of inputs (see metric parameter below, e.g. pwsd.metrics.nb_vector_editdistance)

    Will use multiprocessing Pool if ncpus > 1. With numba metrics, multiprocessing will probably not lead to reduced
    wall time because there is overhead compiling the numba metric and typically the metrics are fast enough
    that it would only help with a large number of sequences (so large that pairwise distances would probably 
    not fit in memory)

    For efficiency, will only compute metric on unique values in
    seqs1/seqs2. All values are returned, including redundancies.

    Though written to be used for distance calculations,
    it is general enough that it could be used to run
    any arbitrary function on pairs of elements in seqs.

    Parameters
    ----------
    metric : function
        A distance function of the form:
            func(seq1, seq2, **kwargs)
        If use_numba is True then metric must operate on a vector of pairwise indices of the form:
            func(seqs_mat, seqs_L, pw_indices, *numba_args)
    seqs1, seqs2 : lists
        Lists of sequences. seqs2 is optional and if it is None all pairwise distances are computed for seqs1
        and a square matrix is returned
    ncpus : int
        Size of the worker pool to be used by multiprocessing
    use_numba : bool
        Use a numba-compiled metric
        For numba, ncpus is ignored because the loop
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

    useqs1, seqs1_uind = np.unique(seqs1, return_inverse=True)
    if len(useqs1) == len(seqs1) or not uniqify:
        useqs1 = seqs1
        translate1 = False
    else:
        translate1 = True

    if not seqs2 is None:
        useqs2, seqs2_uind = np.unique(seqs2, return_inverse=True)
        if len(useqs2) == len(seqs2) or not uniqify:
            useqs2 = seqs2
            translate2 = False
        else:
            translate2 = True

        if not uniqify or (not translate1 and not translate2):
            useqs = useqs1 + useqs2
        else:
            useqs = np.concatenate((useqs1, useqs2))

        pw_indices = list(itertools.product(range(len(useqs1)), range(len(useqs1), len(useqs2) + len(useqs1))))
    else:
        useqs = useqs1
        pw_indices = list(itertools.combinations(range(len(useqs)), 2))

    chunk_func = lambda l, n: [l[i:i + n] for i in range(0, len(l), n)]
    chunksz = len(pw_indices) // ncpus
    """Chunked indices is a list of lists of indices"""
    chunked_indices = chunk_func(pw_indices, chunksz)
    
    if not use_numba:
        dtype = type(metric(useqs[0], useqs[0], **kwargs))
        if ncpus > 1:
            try:
                """Was not able to get joblib to provide any speedup over 1 CPU, though did not test thoroughly.
                multiprocessing.Pool works OK and provides speedup over 1 CPU"""
                # dists = Parallel(n_jobs=ncpus)(delayed(compute_many)(pw_i, metric, useqs, dtype, **kwargs) for pw_i in chunked_indices)
                with multiprocessing.Pool(ncpus) as pool:
                    dists = parmap.map(compute_many,
                                       chunked_indices,
                                       metric,
                                       useqs,
                                       dtype,
                                       **kwargs,
                                       pm_parallel=True,
                                       pm_pool=pool)
                urect = np.concatenate(dists)
            except ValueError as err:
                print('pwseqdist.apply_pairwise_rect: error with metric %s and multiprocessing, trying on single core' % metric)
                urect = compute_many(pw_indices, metric, useqs, dtype, **kwargs)
                print('pwseqdist.apply_pairwise_rect: metric %s could not be spread to multiple processes, ran on single core' % metric)
        else:
            urect = compute_many(pw_indices, metric, useqs, dtype, **kwargs)

    else:
        if ncpus > 1:
            """Now a list of the chunked [chunksz x 2] arrays"""
            chunked_indices = [np.array(i, dtype=np.int64) for i in chunked_indices]
            seqs_mat, seqs_L = seqs2mat(useqs)
            
            # , prefer='threads', require='sharedmem'
            # dists = Parallel(n_jobs=ncpus)(delayed(metric)(pw_i, seqs_mat, seqs_L, *numba_args) for pw_i in chunked_indices)
            with multiprocessing.Pool(ncpus) as pool:
                dists = parmap.map(metric,
                                   chunked_indices,
                                   seqs_mat,
                                   seqs_L,
                                   *numba_args,
                                   pm_parallel=True,
                                   pm_pool=pool)
            urect = np.concatenate(dists)
        else:
            pw_indices = np.array(pw_indices, dtype=np.int64)
            seqs_mat, seqs_L = seqs2mat(useqs)

            """Not neccessary because metric should be pre-jitted. This allowed for changing parallel
            programatically, but this ended up not being helpful for speed"""
            # nb_metric = nb.jit(metric, nopython=True, parallel=ncpus > 1, nogil=True)
            
            """Second one here requires passing the standard metric, while the first requires
            passing the "vector" metric. Speed is quite comparable"""
            urect = metric(pw_indices, seqs_mat, seqs_L, *numba_args)
            # urect = nb_distance_vec(seqs_mat, seqs_L, pw_indices, metric, *numba_args)

    if seqs2 is None:
        urect = scipy.spatial.distance.squareform(urect)
        if translate1:
            urect = urect[seqs1_uind, :][:, seqs1_uind]
    else:
        urect = urect.reshape((len(useqs1), len(useqs2)))
    
        if translate1:
            urect = urect[seqs1_uind, :]
        if translate2:
            urect = urect[:, seqs2_uind]
    return urect

def apply_pairwise_sparse(metric, seqs, pairs, ncpus=1, use_numba=False, numba_args=(), **kwargs):
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
        A distance function of the form:
            func(seq1, seq2, **kwargs)
        If use_numba is True then metric must operate on a vector of pairwise indices of the form:
            func(seqs_mat, seqs_L, pw_indices, *numba_args)
    ncpus : int
        Size of the worker pool to be used by multiprocessing
    use_numba : bool
        Use a numba-compiled metric
        For numba, ncpus is ignored
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
    
    chunk_func = lambda l, n: [l[i:i + n] for i in range(0, len(l), n)]
    chunksz = len(pairs)//ncpus
    chunked_indices = chunk_func(pairs, chunksz)
    if not use_numba:
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
                    vec = np.concatenate(dists)
                except ValueError as err:
                    print('pwseqdist.apply_pairwise_sparse: error with metric %s and multiprocessing, trying on single core' % metric)
                    urect = compute_many(pairs, metric, seqs, dtype, **kwargs)
                    print('pwseqdist.apply_pairwise_sparse: metric %s could not be spread to multiple processes, ran on single core' % metric)
        else:
            vec = compute_many(pairs, metric, seqs, dtype, **kwargs)
        
    else:
        if ncpus > 1:
            """Now a list of the chunked [chunksz x 2] arrays"""
            chunked_indices = [np.array(i, dtype=np.int64) for i in chunked_indices]
            seqs_mat, seqs_L = seqs2mat(seqs)
            
            # dists = Parallel(n_jobs=ncpus)(delayed(metric)(pw_i, seqs_mat, seqs_L, *numba_args) for pw_i in chunked_indices)
            with multiprocessing.Pool(ncpus) as pool:
                dists = parmap.map(metric,
                                   chunked_indices,
                                   seqs_mat,
                                   seqs_L,
                                   *numba_args,
                                   pm_parallel=True,
                                   pm_pool=pool)
            vec = np.concatenate(dists)
        else:
            pw_indices = np.array(pairs, dtype=np.int64)
            seqs_mat, seqs_L = seqs2mat(seqs)

            """Not neccessary because metric should be pre-jitted. This allowed for changing parallel
            programatically, but this ended up not being helpful for speed"""
            # nb_metric = nb.jit(metric, nopython=True, parallel=ncpus > 1, nogil=True)
            
            """Second one here requires passing the standard metric, while the first requires
            passing the "vector" metric. Speed is quite comparable"""
            vec = metric(pw_indices, seqs_mat, seqs_L, *numba_args)
            # urect = nb_distance_vec(seqs_mat, seqs_L, pw_indices, metric, *numba_args)
    return vec


def _apply_pairwise_sq(seqs, metric, ncpus=1, use_numba=False, uniqify=True, numba_args=(), **kwargs):
    """Calculate distance between all pairs of seqs using metric
    and kwargs provided to metric. Will use multiprocessing Pool
    if ncpus > 1.

    WORKING BUT DEPRECATED IN FAVOR OF apply_pairwise_rect TAKING seqs2=None AS AN OPTION

    For efficiency, will only compute metric on unique values in
    seqs. All values are returned, including redundancies.

    Assumes that distance between identical seqs is 0.

    Though written to be used for distance calculations,
    it is general enough that it could be used to run
    any arbitrary function on pairs of elements in seqs.

    Parameters
    ----------
    seqs : list
        List of sequences provided to metric in pairs.
    metric : function
        A distance function of the form:
            func(seq1, seq2, **kwargs)
        If use_numba is True then metric must operate on a vector of pairwise indices of the form:
            func(seqs_mat, seqs_L, pw_indices, *numba_args)
    ncpus : int
        Size of the worker pool to be used by multiprocessing. If ncpus > 1 and use_numba=True, will
        use all CPUs available to numba
    use_numba : bool
        Use a numba-compiled metric
        For numba, ncpus is ignored because the loop
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
    dmat : np.ndarray, [n, n]
        Square pairwise distance matrix."""
    
    useqs, seqs_uind = np.unique(seqs, return_inverse=True)
    if len(useqs) == len(seqs) or not uniqify:
        useqs = seqs
        translate = False
    else:
        translate = True

    """itertools.combinations creates the i,j pairs in the same order
    as scipy.spatial.distance.pdist/squareform"""
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
            dists = parmap.map(compute_many,
                                   chunked_indices,
                                   metric,
                                   useqs,
                                   dtype,
                                   **kwargs,
                                   pm_parallel=False)

        uvec = np.concatenate(dists) # this may be more memory intensive, but should be fine
    else:
        pw_indices = np.array(pw_indices, dtype=np.int64)
        seqs_mat, seqs_L = seqs2mat(useqs)
        """Not neccessary because metric should be pre-jitted. This allowed for changing parallel
        programatically, but this ended up not being helpful for speed"""
        # nb_metric = nb.jit(metric, nopython=True, parallel=ncpus > 1, nogil=True)
        
        """Second one here requires passing the standard metric, while the first requires
        passing the "vector" metric. Speed is quite comparable"""
        uvec = metric(seqs_mat, seqs_L, pw_indices, *numba_args)
        # uvec = nb_distance_vec(seqs_mat, seqs_L, pw_indices, metric, *numba_args)
    
    umat = scipy.spatial.distance.squareform(uvec)
    if translate:
        vout = umat[seqs_uind, :][:, seqs_uind]
    else:
        vout = umat
    return vout