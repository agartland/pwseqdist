import itertools
import numpy as np
import numba as nb
import scipy
import scipy.spatial.distance
import multiprocessing
import parmap
import numpy as np
# from joblib import Parallel, delayed
# import concurrent.futures

from .metrics import compute_many
# from .numba_tools import nb_distance_vec
from .matrices import seqs2mat, parasail_aa_alphabet

__all__ = ['apply_pairwise_rect',
           'apply_pairwise_sparse',
           'apply_running_rect']

"""TODO:
Currently I pass all the sequences and some set of indices to compute_many. 
Why wouldn't I just send some of the sequences?
The point was to avoid sending all the pairs of sequences and just send
pairs of indices to the workers. So we'd have to be smart about reducing the
total number of sequences that are needed and then sending just those and
translated indices."""


def apply_pairwise_rect(metric, seqs1, *args, seqs2=None, ncpus=1, use_numba=False, uniqify=True, reexpand=True, alphabet=parasail_aa_alphabet, **kwargs):
    """Calculate distance between pairs of sequences in seqs1
    with sequences in seqs2 using metric and kwargs provided to
    metric.

    When seqs2=None, a square matrix of pairwise distances is computed among all seqs in seqs1.
    However, it is assumed that the diagonal (ie dist(seq_a, seqs_a)) is always zero and that
    the metric is symetric. If this may not be true, provide the same set of seqs as
    seqs1 and seqs2 to get the fully computed square pairwise matrix.

    Can provide a numba compiled metric to increase speed. Note that the numba metric should accept a different
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
    uniqify : bool
        Indicates whether only unique sequences should be analyzed.
    *args, **kwargs : additional arguments
        Additional positional/keyword arguments supplied to the metric.

    Returns
    -------
    dmat : np.ndarray, length len(seqs1) * len(seqs2)
        Matrix of the pairwise distance rectangle."""

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

        useqs = np.concatenate((useqs1, useqs2))

        pw_indices = list(itertools.product(range(len(useqs1)), range(len(useqs1), len(useqs2) + len(useqs1))))
    else:
        useqs = useqs1
        if len(useqs) == 1:
            """Only one unique sequence (this is only a problem when seqs2=None)"""
            urect = np.zeros((1, 1))
            if reexpand:
                if translate1:
                    urect = urect[seqs1_uind, :][:, seqs1_uind]
                return urect
            else:
                return urect, seqs1_uind, seqs1_uind
        else:
            pw_indices = list(itertools.combinations(range(len(useqs)), 2))

    chunk_func = lambda l, n: [l[i:i + n] for i in range(0, len(l), n)]
    chunksz = max(len(pw_indices) // ncpus, 1)
    """Chunked indices is a list of lists of indices"""
    chunked_indices = chunk_func(pw_indices, chunksz)
    
    if not use_numba:
        dtype = type(metric(useqs[0], useqs[0], **kwargs))
        if ncpus > 1 and len(useqs) > 10:
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
                                       *args,
                                       **kwargs,
                                       pm_parallel=True,
                                       pm_pool=pool)
                urect = np.concatenate(dists)
            except ValueError as err:
                print('pwseqdist.apply_pairwise_rect: error with metric %s and multiprocessing, trying on single core' % metric)
                urect = compute_many(pw_indices, metric, useqs, dtype, *args, **kwargs)
                print('pwseqdist.apply_pairwise_rect: metric %s could not be spread to multiple processes, ran on single core' % metric)
        else:
            urect = compute_many(pw_indices, metric, useqs, dtype, *args, **kwargs)

    else:
        seqs_mat, seqs_L = seqs2mat(useqs, alphabet=alphabet)

        if ncpus > 1 and len(useqs) > 10:
            """Now a list of the chunked [chunksz x 2] arrays"""
            chunked_indices = [np.array(i, dtype=np.int64) for i in chunked_indices]
            
            # , prefer='threads', require='sharedmem'
            # dists = Parallel(n_jobs=ncpus)(delayed(metric)(pw_i, seqs_mat, seqs_L, *numba_args) for pw_i in chunked_indices)
            with multiprocessing.Pool(ncpus) as pool:
                dists = parmap.map(metric,
                                   chunked_indices,
                                   seqs_mat,
                                   seqs_L,
                                   *args,
                                   **kwargs,
                                   pm_parallel=True,
                                   pm_pool=pool)
            urect = np.concatenate(dists)
        else:
            pw_indices = np.array(pw_indices, dtype=np.int64)
            
            """Not neccessary because metric should be pre-jitted. This allowed for changing parallel
            programatically, but this ended up not being helpful for speed"""
            # nb_metric = nb.jit(metric, nopython=True, parallel=ncpus > 1, nogil=True)
            
            """Second one here requires passing the standard metric, while the first requires
            passing the "vector" metric. Speed is quite comparable"""
            urect = metric(pw_indices, seqs_mat, seqs_L, *args, **kwargs)
            # urect = nb_distance_vec(seqs_mat, seqs_L, pw_indices, metric, *numba_args)
    if reexpand:
        if seqs2 is None:
            urect = scipy.spatial.distance.squareform(urect, force='tomatrix')
            if translate1:
                urect = urect[seqs1_uind, :][:, seqs1_uind]
        else:
            urect = urect.reshape((len(useqs1), len(useqs2)))
        
            if translate1:
                urect = urect[seqs1_uind, :]
            if translate2:
                urect = urect[:, seqs2_uind]
        return urect
    else:
        if seqs2 is None:
            urect = scipy.spatial.distance.squareform(urect, force='tomatrix')
            if not translate1:
                seqs1_uind = np.arange(urect.shape[0])
                seqs2_uind = np.arange(urect.shape[0])
            else:
                seqs2_uind = seqs1_uind
        else:
            urect = urect.reshape((len(useqs1), len(useqs2)))
        
            if not translate1:
                seqs1_uind = np.arange(urect.shape[0])
            if not translate2:
                seqs2_uind = np.arange(urect.shape[1])
        """Return the unexpanded pw matrix and indices to expand along axis=0 and axis=1"""
        return urect, seqs1_uind, seqs2_uind 

def apply_pairwise_sparse(metric, seqs, pairs, *args, ncpus=1, use_numba=False, alphabet=parasail_aa_alphabet, **kwargs):
    """Calculate distance between pairs of sequences in seqs using metric and kwargs
    provided to metric. Will only compute distances specified in pairs of indices in pairs.
    Results could be used to create a sparse matrix of pairwise distances.

    Disadvantage here is that there is no attempt to avoid redundant distance calculations.

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
    *args, **kwargs : additional arguments
        Additional positional/keyword arguments supplied to the metric.

    Returns
    -------
    dvec : np.ndarray, length len(pairs)
        Vector of distances for each pair of indices in pairs"""
    assert np.max(np.array(pairs)) <= len(seqs), "Indices specified in pairs cannot exceed length of seqs"
    assert np.min(np.array(pairs)) >= 0, "Indices specified in pairs cannot be less than 0"
          
    chunk_func = lambda l, n: [l[i:i + n] for i in range(0, len(l), n)]
    chunksz = max(len(pairs) // ncpus, 1)
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
                                       *args,
                                       **kwargs,
                                       pm_parallel=True,
                                       pm_pool=pool)
                    vec = np.concatenate(dists)
                except ValueError as err:
                    print('pwseqdist.apply_pairwise_sparse: error with metric %s and multiprocessing, trying on single core' % metric)
                    urect = compute_many(pairs, metric, seqs, dtype, *args, **kwargs)
                    print('pwseqdist.apply_pairwise_sparse: metric %s could not be spread to multiple processes, ran on single core' % metric)
        else:
            vec = compute_many(pairs, metric, seqs, dtype, *args, **kwargs)
        
    else:
        if ncpus > 1:
            """Now a list of the chunked [chunksz x 2] arrays"""
            chunked_indices = [np.array(i, dtype=np.int64) for i in chunked_indices]
            seqs_mat, seqs_L = seqs2mat(seqs, alphabet=alphabet)
            
            # dists = Parallel(n_jobs=ncpus)(delayed(metric)(pw_i, seqs_mat, seqs_L, *numba_args) for pw_i in chunked_indices)
            with multiprocessing.Pool(ncpus) as pool:
                dists = parmap.map(metric,
                                   chunked_indices,
                                   seqs_mat,
                                   seqs_L,
                                   *args,
                                   **kwargs,
                                   pm_parallel=True,
                                   pm_pool=pool)
            vec = np.concatenate(dists)
        else:
            pw_indices = np.array(pairs, dtype=np.int64)
            seqs_mat, seqs_L = seqs2mat(seqs, alphabet=alphabet)

            """Not neccessary because metric should be pre-jitted. This allowed for changing parallel
            programatically, but this ended up not being helpful for speed"""
            # nb_metric = nb.jit(metric, nopython=True, parallel=ncpus > 1, nogil=True)
            
            """Second one here requires passing the standard metric, while the first requires
            passing the "vector" metric. Speed is quite comparable"""
            vec = metric(pw_indices, seqs_mat, seqs_L, *args, **kwargs)
            # urect = nb_distance_vec(seqs_mat, seqs_L, pw_indices, metric, *numba_args)
    return vec

def apply_running_rect(metric, seqs1, seqs2, radius, density_est, *args, ncpus=1, uniqify=True, alphabet=parasail_aa_alphabet, **kwargs):
    """Compute distances between seqs in seqs1 and seqs in seqs2 but only return
    the indices into seqs2 for each seqs1 when the distance < radius.

    Parameters
    ----------
    metric : function
        A "running" distance function, e.g. pwsd.running.nb_running_tcrdist
    seqs1, seqs2 : lists
        Lists of sequences with seqs1 ideally being shorter
    ncpus : int
        Size of the worker pool to be used by multiprocessing
    uniqify : bool
        Indicates whether only unique sequences in seqs1 should be analyzed.
    *args, **kwargs : additional arguments for the metric
        
    Returns
    -------
    res : list of (indices, dvec) np.ndarrays of equal length
        indices : np.ndarray
            List (i) of length seqs1 with indices (j) into seqs2 with D(seqs1_i|seqs2_j) < radius
        dvec : np.ndarray
            Vector of distances with radius < R"""

    useqs1, seqs1_uind = np.unique(seqs1, return_inverse=True)
    if len(useqs1) == len(seqs1) or not uniqify:
        useqs1 = seqs1
        translate1 = False
    else:
        translate1 = True

    # nb_running_x(query_i, seqs_mat, seqs_L, radius, density_est=0.05, *args)
    """This is wasteful. Should create a seqsmat slice of only the rect that is needed
    each time."""
    seqs_mat, seqs_L = seqs2mat(useqs1 + seqs2, alphabet=alphabet)
    n1 = len(useqs1)

    query_indices = range(n1)
    if ncpus > 1:
        with multiprocessing.Pool(ncpus) as pool:
            res = parmap.map(metric,
                               query_indices,
                               seqs_mat,
                               seqs_L,
                               radius,
                               density_est,
                               *args,
                               **kwargs,
                               pm_parallel=True,
                               pm_pool=pool)
        
    else:
        res = [metric(query_i, seqs_mat, seqs_L, radius, density_est, **kwargs) for query_i in query_indices]
        
    """Subtract off the n1 seqs from useqs1"""
    if translate1:
        """Re-expand seqs1"""
        res = [(res[res_i][0][res[res_i][0] >= n1] - n1, res[res_i][1][res[res_i][0] >= n1]) for resi in seqs1_uind]
    else:
        res = [(ind[ind >= n1] - n1, d[ind >= n1]) for ind, d in res]

    return res


def _apply_pairwise_sq(seqs, metric, ncpus=1, use_numba=False, uniqify=True, numba_args=(), alphabet=parasail_aa_alphabet, **kwargs):
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
        seqs_mat, seqs_L = seqs2mat(useqs, alphabet=alphabet)
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
