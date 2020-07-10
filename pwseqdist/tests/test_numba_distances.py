import numpy as np
import timeit
import pandas as pd
import numba as nb

import editdistance
import stringdist

# from fg_shared import *
# import sys
# from os.path import join as opj
# sys.path.append(opj(_git, 'pwseqdist'))
import pwseqdist as pwsd

mixed_seqs = ['CACADLGAYPDKLIF',
             'CACADLGARPDKLIF',
             'CACADLGAAYPDKLIF',
             'CACADLGAYPDRLIF',
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

def test_nb_pw_rect():
    drect = pwsd.apply_pairwise_rect(pwsd.metrics.nb_vector_editdistance, seqs1=mixed_seqs[:2], seqs2=mixed_seqs, use_numba=True)
    assert drect.shape[0] == 2
    assert drect.shape[1] == len(mixed_seqs)

def test_benchmark_running(rapid_test=True):
    if rapid_test:
        longseqs = mixed_seqs * 20000 # 1M sequences, 50 unique
        number=1
        repeat=1
    else:
        longseqs = mixed_seqs * 5000000 # 250M sequences, 50 unique
        number=2
        repeat=2

    seqs_mat, seqs_L = pwsd.matrices.seqs2mat(longseqs)
    indices = np.array([(0, i) for i in range(seqs_mat.shape[0])], dtype=np.int)

    namespace = globals()
    namespace.update(locals())
    #res = timeit.repeat(stmt='pwsd.matrices.seqs2mat(longseqs)', repeat=repeat, number=repeat, globals=namespace)
    #print(f'Conversion of {len(longseqs)} sequences with seqs2mat(): {1e3 * np.min(res)/1:1.1f} ms')

    nb_metrics = [('edit', 'pwsd.running.nb_running_editdistance', ['3', '0.05'], 'pwsd.metrics.nb_vector_editdistance'),
                   ('tcrdist', 'pwsd.running.nb_running_tcrdist', ['30', '0.05', 'pwsd.matrices.tcr_nb_distance_matrix', '3', '4', '3', '2', 'True'], 'pwsd.metrics.nb_vector_tcrdist'),
                   ('tcrdist (optimal gappos)', 'pwsd.running.nb_running_tcrdist', ['50', '0.05', 'pwsd.matrices.tcr_nb_distance_matrix', '3', '4', '3', '2', 'False'], 'pwsd.metrics.nb_vector_tcrdist')]

    print(f'\nBenchmarks on {len(longseqs)//1000000}M sequences:')
    for metric_name, metric_str, kwargs, vec_metric in nb_metrics:
        # nb_running_editdistance(0, seqs_mat, seqs_L, radius, est_density)
        # nb_metric = eval(f'nb.jit({metric_str}, nopython=True, parallel={par}, nogil=True)')
        # namespace.update({'nb_metric':nb_metric})
        stmt = f'{metric_str}(0, seqs_mat, seqs_L{", " if kwargs else ""}{", ".join(kwargs)})'
        res = timeit.repeat(stmt=stmt, setup=stmt, repeat=repeat, number=number, globals=namespace)
        print(f'Numba running, {metric_name} distances:\t{1e3 * np.min(res)/number:1.1f} ms')
        
        ind, d = eval(stmt)
        #nb_metric = eval(f'nb.jit({vec_metric}, nopython=True, parallel=False, nogil=True)')
        #namespace.update({'nb_metric':nb_metric})
        stmt = f'{vec_metric}(indices, seqs_mat, seqs_L{", " if len(kwargs) > 2 else ""}{", ".join(kwargs[2:])})'
        #print(stmt)
        d2 = eval(stmt)
        ind2 = np.nonzero(d2 <= int(kwargs[0]))[0]
        d2 = d2[ind2]
        if np.all(d == d2):
            print(f'\tConfirmed {len(d)//1000}K neighbors.')
        else:
            print(f'\tFound {len(d)//1000}K neighbors with running; expected {len(d2)//1000}K.')
            print(ind[:10], d[:10])
            print(ind2[:10], d2[:10])
            raise
    print(f'\nBenchmarks on 2 x {len(longseqs)//1000000}M sequences:')
    # apply_running_rect(metric, seqs1, seqs2, radius, density_est, *args, ncpus=1, uniqify=True, **kwargs)
    for metric_name, metric_str, kwargs, vec_metric in nb_metrics:
        stmt = f'pwsd.apply_running_rect({metric_str}, mixed_seqs[:2], longseqs{", " if kwargs else ""}{", ".join(kwargs)})'
        res = timeit.repeat(stmt=stmt, setup=stmt, repeat=repeat, number=number, globals=namespace)
        print(f'PWSD numba running, {metric_name} distances:\t{1e3 * np.min(res)/number:1.1f} ms')


def test_benchmark_metrics(rapid_test=True):
    if rapid_test:
        longseqs = mixed_seqs * 20000 # 1M sequences, 50 unique
        number=1
        repeat=1
    else:
        longseqs = mixed_seqs * 500000 # 25M sequences, 50 unique
        number=3
        repeat=3

    seqs_mat, seqs_L = pwsd.matrices.seqs2mat(longseqs)
    indices = np.array([(0, i) for i in range(seqs_mat.shape[0])], dtype=np.int)

    namespace = globals()
    namespace.update(locals())
    res = timeit.repeat(stmt='pwsd.matrices.seqs2mat(longseqs)', repeat=repeat, number=number, globals=namespace)
    print(f'Conversion of {len(longseqs)} sequences with seqs2mat(): {1e3 * np.min(res)/1:1.1f} ms')

    nb_metrics = [('edit', 'pwsd.metrics.nb_editdistance', []),
                   ('tcrdist', 'pwsd.metrics.nb_tcrdist', ['pwsd.matrices.tcr_nb_distance_matrix', '3', '4', '3', '2', 'True']),
                   ('tcrdist (optimal gappos)', 'pwsd.metrics.nb_tcrdist', ['pwsd.matrices.tcr_nb_distance_matrix', '3', '4', '3', '2', 'False']),
                   ('hamming', 'pwsd.metrics.nb_hamming_distance', ['False'])]

    print(f'\nBenchmarks on {len(longseqs)//1000000}M sequences (50 unique):')
    for metric_name, metric_str, kwargs in nb_metrics:
        stmt = f'pwsd.numba_tools.nb_distance_vec(indices, seqs_mat, seqs_L, {metric_str}{", " if kwargs else ""}{", ".join(kwargs)})'
        #stmt = f'nb_distance_vec(seqs_mat, seqs_L, indices, {metric_str}{", " if kwargs else ""}{", ".join(kwargs)})'
        res = timeit.repeat(stmt=stmt, setup=stmt, repeat=repeat, number=number, globals=namespace)
        print(f'Numba, individual {metric_name} distances:\t{1e3 * np.min(res)/number:1.1f} ms')

    nb_vec_metrics = [('edit', 'pwsd.metrics.nb_vector_editdistance', {}),
                      ('hamming', 'pwsd.metrics.nb_vector_hamming_distance', {'check_lengths':'False'}),
                       ('tcrdist', 'pwsd.metrics.nb_vector_tcrdist', {'fixed_gappos':'True'}),
                       ('tcrdist (optimal gappos)', 'pwsd.metrics.nb_vector_tcrdist', {'fixed_gappos':'False'})]
    for metric_name, metric_str, kwargs in nb_vec_metrics:
        # nb_metric = eval(f'nb.jit({metric_str}, nopython=True, parallel={par}, nogil=True)')
        # namespace.update({'nb_metric':nb_metric})
        stmt = f'{metric_str}(indices, seqs_mat, seqs_L{get_kwargs_str(kwargs)})'
        res = timeit.repeat(stmt=stmt, setup=stmt, repeat=repeat, number=number, globals=namespace)
        print(f'Numba vectorized, {metric_name} distances:\t{1e3 * np.min(res)/number:1.1f} ms')

def test_benchmark_pairwise_rect(rapid_test=True):
    if rapid_test:
        longseqs = mixed_seqs * 10 # 500 sequences, 50 unique, 250,000 squared or 2,500 squared-unique
        repeat = 1
        number = 1
    else:
        longseqs = mixed_seqs * 50 # 2500 sequences, 50 unique, 6,250,000 squared or 2,500 squared-unique
        # longseqs = mixed_seqs * 500 # 25000 sequences, 50 unique, 625,000,000 squared or 2,500 squared-unique
        repeat = 2
        number = 2

    metrics = [('Numba edit', 'pwsd.metrics.nb_vector_editdistance', {'distance_matrix':'pwsd.matrices.identity_nb_distance_matrix'}, True),
                   ('Numba tcrdist', 'pwsd.metrics.nb_vector_tcrdist', {'fixed_gappos':'True'}, True),
                   ('Numba tcrdist (optimal gappos)', 'pwsd.metrics.nb_vector_tcrdist', {'fixed_gappos':'False'}, True),
                   ('Numba hamming', 'pwsd.metrics.nb_vector_hamming_distance', {'check_lengths':'False'}, True),
                   ('String hamming', 'pwsd.metrics.hamming_distance', {}, False),
                   ('C Needleman-Wunsch', 'pwsd.metrics.nw_metric', {}, False),
                   ('Cython stringdist', 'stringdist.levenshtein', {}, False),
                   ('Cython editdistance', 'editdistance.eval', {}, False)]

    # metrics = [metrics[1]]
    # metrics = [metrics[0]]
    
    """DEPRECATED calls to pwsd.apply_pairwise_X"""
    """pw_functions = [('rect', 'pwsd.apply_pairwise_rect(seqs1=longseqs, seqs2=longseqs, ncpus={ncpus}, metric={metric_str}, uniqify={uniqify}, use_numba={use_numba}, numba_args=({kwargs_str}))'),
                    ('square', 'pwsd.apply_pairwise_sq(seqs=longseqs, ncpus={ncpus}, metric={metric_str}, uniqify={uniqify}, use_numba={use_numba}, numba_args=({kwargs_str}))')]"""

    pw_str = 'pwsd.apply_pairwise_rect(seqs1=longseqs, seqs2=longseqs, ncpus={ncpus}, metric={metric_str}, uniqify={uniqify}, use_numba={use_numba}{kwargs_str})'
    pw_name = 'rect'

    namespace = globals()
    namespace.update(locals())
    for ncpus in [2, 1, 4]:
        for uniq_str, uniqify, number in [(f'{len(mixed_seqs)**2} unique', True, 2), (f'all {len(longseqs)**2}', False, 2)]:
            for metric_name, metric_stmt, kwargs, use_numba in metrics:
                stmt = pw_str.format(ncpus=ncpus,
                                     metric_str=metric_stmt,
                                     uniqify=uniqify,
                                     use_numba=use_numba,
                                     kwargs_str=get_kwargs_str(kwargs))
                res = timeit.repeat(stmt=stmt, setup=stmt, repeat=repeat, number=number, globals=namespace)
                print(f'PWSD pairwise {pw_name} ({uniq_str}), ncpus={ncpus}, {metric_name} distances:\t{1e3 * np.min(res)/number:1.1f} ms')

def get_kwargs_str(kwargs):
    kwargs_str = ''
    for k,v in kwargs.items():
        kwargs_str = kwargs_str + ', ' + f'{k}={v}'
    return kwargs_str

if __name__ == '__main__':
    #test_nb_pw_rect()
    #test_benchmark_running(rapid_test=True)
    #test_benchmark_metrics(rapid_test=True)
    #test_benchmark_pairwise_rect(rapid_test=True)
    pass
    #unittest.main()