import sys
import unittest
import pytest
import numpy as np
from scipy.spatial.distance import squareform
import parasail
import timeit
import pandas as pd
import operator
import numba as nb
from os.path import join as opj

import editdistance
import stringdist

from fg_shared import *

sys.path.append(opj(_git, 'pwseqdist'))
import pwseqdist as pwsd

mixed_seqs = ['CACADLGAYPDKLIF',
             'CACDALLAYTDKLIF',
             'CACDAVGDTLDKLIF',
             'CACDDVTEVEGDKLIF',
             'CACDFISPSNWGIQSGRNTDKLIF',
             'CACDILLGDTADKLIF',
             'CACDIVLSGGLDTRQMFF',
             'CACDLLLRQSSTDKLIF',
             'CACDNLSETTDKLIF',
             'CACDPLGTDKLIF',
             'CACDPMGGSGGLSWDTRQMFF',
             'CACDPVLGDTRLTDKLIF',
             'CACDPVQGYSGQNRAYTDKLIF',
             'CACDSILGDTLYTDKLIF',
             'CACDSLTSHTGGFGPDKLIF',
             'CACDSTGDLSSWDTRQMFF',
             'CACDSVESRNVLGDPTTDKLIF',
             'CACDSVLSRDLGDSELIF',
             'CACDTAAGGYASSWDTRQMFF',
             'CACDTAPHGGRTWDTRQMFF',
             'CACDTGGYVNWDTRQMFF',
             'CACDTGRLLGDTADTRQMFF',
             'CACDTIRGFSSWDTRQMFF',
             'CACDTIVAPALDKLIF',
             'CACDTLFLGEDTPTDKLIF',
             'CACDTLGDLSLTAQLFF',
             'CACDTLGDPPHTDKLIF',
             'CACDTLGDYTQSDKLIF',
             'CACDTLGGYPWDTRQMFF',
             'CACDTLGKTDKLIF',
             'CACDTLPLKTGGPLYTDKLIF',
             'CACDTLRLGDPLNTDKLIF',
             'CACDTVALGDTESSWDTRQMFF',
             'CACDTVGAVLGDPKGTDKLIF',
             'CACDTVGDGPDTDKLIF',
             'CACDTVGDTADKLIF',
             'CACDTVGDTHSWDTRQMFF',
             'CACDTVGGSTDKLIF',
             'CACDTVGIPPDKLIF',
             'CACDTVGYGEGDTDKLIF',
             'CACDTVISSNRRGGDKLIF',
             'CACDTVPPGDTGTDKLIF',
             'CACDTVRFTGGYENTDKLIF',
             'CACDYVLGAEDKLIF',
             'CACEGILKSEPLGIDKLIF',
             'CACEMLGHPPGDKLIF',
             'CACVSLDLSYTDKLIF',
             'CALGEIAFRSRTGGPPYTDKLIF',
             'CALGTAYFLRDPGADKLIF',
             'CAVKVPLTSSPREGPTVLHDKLIF']

seqs = [s[:10] for s in mixed_seqs]

def test_nb_pw_sq_hamming():
    dvec = pwsd.apply_pairwise_sq(mixed_seqs, pwsd.metrics.nb_editdistance, use_numba=True)
    assert dvec.shape[0] == (len(mixed_seqs)**2 - len(mixed_seqs)) / 2

def test_nb_pw_rect():
    drect = pwsd.apply_pairwise_rect(mixed_seqs[:2], mixed_seqs, pwsd.metrics.nb_editdistance, use_numba=True)
    assert drect.shape[0] == 2
    assert drect.shape[1] == len(mixed_seqs)



def benchmark_metrics():
    number = 10
    longseqs = mixed_seqs * 10000 # 50K sequences, 50 unique
    seqs_mat, seqs_L = pwsd.matrices.seqs2mat(longseqs)
    indices = np.array([(0, i) for i in range(seqs_mat.shape[0])], dtype=np.int)

    res = timeit.repeat(stmt='pwsd.matrices.seqs2mat(longseqs)', repeat=4, number=10, globals=globals())
    print(f'Conversion of {len(longseqs)} sequences with seqs2mat(): {1e3 * np.min(res)/10:1.1f} ms')

    nb_metrics = [('edit', 'pwsd.metrics.nb_editdistance', []),
                   ('tcrdist', 'nb_tcrdist_distance', ['tcr_nb_distance_matrix', '3', '4', '3', '2', 'True']),
                   ('tcrdist (optimal gappos)', 'nb_tcrdist_distance', ['tcr_nb_distance_matrix', '3', '4', '3', '2', 'False']),
                   ('hamming', 'nb_hamming_distance', ['False'])]

    for metric_name, metric_str, kwargs in nb_metrics:
        stmt = f'pwsd.numba_tools.nb_distance_vec(seqs_mat, seqs_L, indices, {metric_str}{", " if kwargs else ""}{", ".join(kwargs)})'
        #stmt = f'nb_distance_vec(seqs_mat, seqs_L, indices, {metric_str}{", " if kwargs else ""}{", ".join(kwargs)})'
        res = timeit.repeat(stmt=stmt, setup=stmt, repeat=4, number=number, globals=globals())
        print(f'Numba, individual {metric_name} distances: {1e3 * np.min(res)/number:1.1f} ms')

    nb_vec_metrics = [('edit', 'nb_vector_editdistance', []),
                       ('tcrdist', 'nb_vector_tcrdist_distance', ['tcr_nb_distance_matrix', '3', '4', '3', '2', 'True']),
                       ('tcrdist (optimal gappos)', 'nb_vector_tcrdist_distance', ['tcr_nb_distance_matrix', '3', '4', '3', '2', 'False'])]
    for metric_name, metric_str, kwargs in nb_vec_metrics:
        # nb_vector_editdistance(0, seqs_mat, seqs_L)
        stmt = f'{metric_str}(0, seqs_mat, seqs_L{", " if kwargs else ""}{", ".join(kwargs)})'
        res = timeit.repeat(stmt=stmt, setup=stmt, repeat=4, number=number, globals=globals())
        print(f'Numba, vectorized {metric_name} distances: {1e3 * np.min(res)/number:1.1f} ms')