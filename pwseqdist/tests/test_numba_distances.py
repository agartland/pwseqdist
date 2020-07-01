import sys
import unittest
import pytest
import numpy as np
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

    res = timeit.repeat(stmt='pwsd.matrices.seqs2mat(longseqs)', repeat=4, number=10, globals=locals())
    print(f'Conversion of {len(longseqs)} sequences with seqs2mat(): {1e3 * np.min(res)/10:1.1f} ms')

    nb_metrics = [('edit', 'pwsd.metrics.nb_editdistance', []),
                   ('tcrdist', 'pwsd.metrics.nb_tcrdist_distance', ['pwsd.matrices.tcr_nb_distance_matrix', '3', '4', '3', '2', 'True']),
                   ('tcrdist (optimal gappos)', 'pwsd.metrics.nb_tcrdist_distance', ['pwsd.matrices.tcr_nb_distance_matrix', '3', '4', '3', '2', 'False']),
                   ('hamming', 'pwsd.metrics.nb_hamming_distance', ['False'])]

    for metric_name, metric_str, kwargs in nb_metrics:
        stmt = f'pwsd.numba_tools.nb_distance_vec(seqs_mat, seqs_L, indices, {metric_str}{", " if kwargs else ""}{", ".join(kwargs)})'
        #stmt = f'nb_distance_vec(seqs_mat, seqs_L, indices, {metric_str}{", " if kwargs else ""}{", ".join(kwargs)})'
        res = timeit.repeat(stmt=stmt, setup=stmt, repeat=4, number=number, globals=locals())
        print(f'Numba, individual {metric_name} distances: {1e3 * np.min(res)/number:1.1f} ms')

    nb_vec_metrics = [('edit', 'pwsd.metrics.nb_vector_editdistance', []),
                       ('tcrdist', 'pwsd.metrics.nb_vector_tcrdist_distance', ['tcr_nb_distance_matrix', '3', '4', '3', '2', 'True']),
                       ('tcrdist (optimal gappos)', 'pwsd.metrics.nb_vector_tcrdist_distance', ['tcr_nb_distance_matrix', '3', '4', '3', '2', 'False'])]
    for metric_name, metric_str, kwargs in nb_vec_metrics:
        # nb_vector_editdistance(0, seqs_mat, seqs_L)
        stmt = f'{metric_str}(0, seqs_mat, seqs_L{", " if kwargs else ""}{", ".join(kwargs)})'
        res = timeit.repeat(stmt=stmt, setup=stmt, repeat=4, number=number, globals=locals())
        print(f'Numba, vectorized {metric_name} distances: {1e3 * np.min(res)/number:1.1f} ms')

def benchmark_pairwise_rect():
    longseqs = mixed_seqs * 10 # 500 sequences, 50 unique, 250,000 squared or 2,500 squared-unique

    metrics = [('Numba edit', 'pwsd.metrics.nb_editdistance', [], True),
                   ('Numba tcrdist', 'pwsd.metrics.nb_tcrdist_distance', ['pwsd.matrices.tcr_nb_distance_matrix', '3', '4', '3', '2', 'True'], True),
                   ('Numba tcrdist (optimal gappos)', 'pwsd.metrics.nb_tcrdist_distance', ['pwsd.matrices.tcr_nb_distance_matrix', '3', '4', '3', '2', 'False'], True),
                   ('Numba hamming', 'pwsd.metrics.nb_hamming_distance', ['False'], True),
                   ('String hamming', 'pwsd.metrics.hamming_distance', [], False),
                   ('C NW', 'pwsd.metrics.nw_metric', [], False),
                   ('Cython stringdist', 'stringdist.levenshtein', [], False),
                   ('Cython editdistance', 'editdistance.eval', [], False)]
    
    pw_functions = [('rect', 'pwsd.apply_pairwise_rect(seqs1=longseqs, seqs2=longseqs, ncpus={ncpus}, metric={metric_str}, uniqify={uniqify}, use_numba={use_numba}, numba_args=({kwargs_str}))'),
                    ('square', 'pwsd.apply_pairwise_sq(seqs=longseqs, ncpus={ncpus}, metric={metric_str}, uniqify={uniqify}, use_numba={use_numba}, numba_args=({kwargs_str}))')]
    
    for ncpus in [1, 2]:
        for uniq_str, uniqify, number in [(f'{len(mixed_seqs)**2} unique', True, 10), (f'all {len(longseqs)**2}', False, 5)]:
            for pw_name, pw_str in pw_functions:
                for metric_name, metric_stmt, kwargs, use_numba in metrics:
                    stmt = pw_str.format(ncpus=ncpus,
                                         metric_str=metric_stmt,
                                         uniqify=uniqify,
                                         use_numba=use_numba,
                                         kwargs_str=f'{", ".join(kwargs)}{"," if kwargs else ""}')
                    res = timeit.repeat(stmt=stmt, setup=stmt, repeat=3, number=number, globals=locals())
                    print(f'PWSD pairwise {pw_name} ({uniq_str}), ncpus={ncpus}, {metric_name} distances: {1e3 * np.min(res)/number:1.1f} ms')
'''
seqs = ['CASSARGF','CASSAARGF'] * 40
#%timeit -r 1 apply_pairwise_sq(seqs=seqs , ncpus=1, metric=pwsd.metrics.nw_hamming_metric)
%timeit -r 1 apply_pairwise_rect(seqs1=seqs, seqs2=seqs, ncpus=1, metric=pwsd.metrics.nw_hamming_metric)

seqs = ['CASSARGF','CASSAARGF'] * 400
#%timeit -r 1 apply_pairwise_sq(seqs=seqs, ncpus=1, metric=pwsd.metrics.nw_hamming_metric)
%timeit -r 1 apply_pairwise_rect(seqs1=seqs, seqs2=seqs, ncpus=1, metric=pwsd.metrics.nw_hamming_metric)


%prun apply_pairwise_sq(seqs=seqs, ncpus=1, metric=pwsd.metrics.nw_hamming_metric)
%prun apply_pairwise_rect(seqs1=seqs, seqs2=seqs, ncpus=1, metric=pwsd.metrics.nw_hamming_metric)
'''
if __name__ == '__main__':
    benchmark_metrics()
    benchmark_pairwise_rect()