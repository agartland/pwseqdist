import pytest 

def test_README_example1():
    import numpy as np

    import pwseqdist as pwsd
    import multiprocessing

    peptides = ['CACADLGAYPDKLIF','CACDALLAYTDKLIF',
                'CACDAVGDTLDKLIF','CACDDVTEVEGDKLIF',
                'CACDFISPSNWGIQSGRNTDKLIF','CACDPVLGDTRLTDKLIF']

    dmat = pwsd.apply_pairwise_rect(seqs1  = peptides, 
                                    metric = pwsd.metrics.nw_hamming_metric, 
                                    ncpus  = multiprocessing.cpu_count()   )

    exp = np.array(
       [[ 0,  4,  6,  7, 15, 8],
       [ 4,  0,  5,  7, 14,  7],
       [ 6,  5,  0,  6, 14,  4],
       [ 7,  7,  6,  0, 14,  8],
       [15, 14, 14, 14,  0, 11],
       [ 8,  7,  4,  8, 11,  0]])

    assert np.all(dmat == exp)