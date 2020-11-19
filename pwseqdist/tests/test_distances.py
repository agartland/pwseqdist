"""
python -m unittest pwseqdist/tests/test_distances.py
"""
import sys
import unittest
import numpy as np
from scipy.spatial.distance import squareform
import parasail
import itertools
# import pytest
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

class TestUnknownSymbols(unittest.TestCase):
    def test_unknowns_metric(self):
        seqs_dot = ['CACADLGAYPDKLIF',
                 'CACDALLAYTDKLIF',
                 'CACDAV...LDKLIF',
                 'CACDDVTEVEGDKLIF',
                 'CACDFISPSNWGIQSGRNTDKLIF']*100
        drect_dots2 = pwsd.apply_pairwise_rect(pwsd.metrics.nb_vector_tcrdist, seqs1=seqs_dot, seqs2=None, use_numba=True, ncpus=2, uniqify=False)
        drect_dots1 = pwsd.apply_pairwise_rect(pwsd.metrics.nb_vector_tcrdist, seqs1=seqs_dot, seqs2=None, use_numba=True, ncpus=1, uniqify=False)

        #print()
        #print(drect_dots1)
        #print(drect_dots2)
        self.assertTrue((drect_dots1 == drect_dots2).all())

class TestDistances(unittest.TestCase):

    def test_one_unique_seq(self):
        seqs_dot = ['CACADLGAYPDKLIF']*100
        drect_dots2 = pwsd.apply_pairwise_rect(pwsd.metrics.nb_vector_tcrdist, seqs1=seqs_dot, seqs2=None, use_numba=True, ncpus=1, uniqify=False)
        drect_dots1 = pwsd.apply_pairwise_rect(pwsd.metrics.nb_vector_tcrdist, seqs1=seqs_dot, seqs2=None, use_numba=True, ncpus=1, uniqify=True)
        self.assertTrue((drect_dots1 == drect_dots2).all())

        drect_u, uind_i, uind_j = pwsd.apply_pairwise_rect(pwsd.metrics.nb_vector_tcrdist, seqs1=seqs_dot, seqs2=None, use_numba=True, ncpus=1, uniqify=True, reexpand=False)
        drect_dots3 = drect_u[uind_i, :][:, uind_j]
        self.assertTrue((drect_dots1 == drect_dots3).all())
    
    def test_haming_metric(self):
        self.assertTrue(pwsd.metrics.hamming_distance(seqs[0], seqs[1]) == 4)
        self.assertTrue(pwsd.metrics.hamming_distance(seqs[0], seqs[0]) == 0)

    def test_subst(self):
        subst_dict = pwsd.matrices.dict_from_matrix(parasail.blosum62)
        for s1, s2 in zip(seqs[-10:], seqs[:10]):
            str_d = pwsd.metrics.str_subst_metric(s1, s2, subst_dict, as_similarity=False, na_penalty=None)

    def test_nw_metric(self):
        subst_dict = pwsd.matrices.dict_from_matrix(parasail.blosum62)
        nw_d = pwsd.metrics.nw_metric(mixed_seqs[0], mixed_seqs[1], matrix='blosum62', open=3, extend=3)
        
        for s1, s2 in zip(seqs[-10:], seqs[:10]):
            nw_d = pwsd.metrics.nw_metric(s1, s2, matrix='blosum62', open=30, extend=30)
            str_d = pwsd.metrics.str_subst_metric(s1, s2, subst_dict, as_similarity=False, na_penalty=None)
            self.assertTrue(nw_d == str_d)

    def test_nw_hamming_metric(self):
        subst_dict = pwsd.matrices.dict_from_matrix(parasail.blosum62)
        nw_d = pwsd.metrics.nw_hamming_metric(mixed_seqs[0], mixed_seqs[1], matrix='blosum62', open=3, extend=3)
        
        for s1, s2 in zip(seqs[-10:], seqs[:10]):
            nw_d = pwsd.metrics.nw_hamming_metric(s1, s2, matrix='blosum62', open=30, extend=30)
            str_d = pwsd.metrics.hamming_distance(s1, s2)
            # print('%s\t%s\t%1.0f\t%1.0f' % (s1, s2, str_d, nw_d))
            self.assertTrue(nw_d == str_d)

class TestApply(unittest.TestCase):
    def test_pw_sq(self):
        dmat = pwsd.apply_pairwise_rect(seqs1=seqs[:10], seqs2=None, metric=pwsd.metrics.hamming_distance, ncpus=1)
        dmat2 = pwsd.apply_pairwise_rect(seqs1=seqs[:10], seqs2=seqs[:10], metric=pwsd.metrics.hamming_distance, ncpus=1)
        
        self.assertTrue(dmat.shape[0] == 10 and dmat.shape[1] == 10)
        self.assertTrue(dmat2.shape[0] == 10 and dmat2.shape[1] == 10)
        self.assertTrue(np.all(dmat == dmat2))
    def test_pw_sq_subst(self):
        subst_dict = pwsd.matrices.dict_from_matrix(parasail.blosum62)
        dmat = pwsd.apply_pairwise_rect(seqs1=seqs[:10], metric=pwsd.metrics.str_subst_metric,  subst_dict=subst_dict, ncpus=1)
        self.assertTrue(dmat.shape[0] == 10 and dmat.shape[1] == 10)

    def test_pw_sq_nonuniq(self):
        dmat = pwsd.apply_pairwise_rect(seqs1=seqs[:10], metric=pwsd.metrics.hamming_distance, ncpus=1)
        dmat2 = pwsd.apply_pairwise_rect(seqs1=seqs[:10] + seqs[:10], metric=pwsd.metrics.hamming_distance, ncpus=1)
        self.assertTrue(np.all(dmat2[:10, :][:, :10] == dmat))
    def test_pw_sq_nonuniq_tcrdist(self):
        tmp = ['PNSSL', 'KEKRN', 'KEKRN', 'PNASF', 'PNASF', 'PNASF', 'EKKES', 'EKKER', 'IRTEH']
        res = np.array([[0, 5, 5, 2, 2, 2, 5, 5, 5,],
                         [5, 0, 0, 5, 5, 5, 4, 4, 5,],
                         [5, 0, 0, 5, 5, 5, 4, 4, 5,],
                         [2, 5, 5, 0, 0, 0, 5, 5, 5,],
                         [2, 5, 5, 0, 0, 0, 5, 5, 5,],
                         [2, 5, 5, 0, 0, 0, 5, 5, 5,],
                         [5, 4, 4, 5, 5, 5, 0, 1, 4,],
                         [5, 4, 4, 5, 5, 5, 1, 0, 4,],
                         [5, 5, 5, 5, 5, 5, 4, 4, 0,]])
        dmat = pwsd.apply_pairwise_rect(seqs1=tmp, metric=pwsd.metrics.nw_hamming_metric, ncpus=1)
        self.assertTrue(np.all(dmat == res))

    def test_pw_rect_nonuniq_tcrdist(self):
        tmp = ['PNSSL', 'KEKRN', 'KEKRN', 'PNASF', 'PNASF', 'PNASF', 'EKKES', 'EKKER', 'IRTEH']
        res = np.array([[0, 5, 5, 2, 2, 2, 5, 5, 5,],
                         [5, 0, 0, 5, 5, 5, 4, 4, 5,],
                         [5, 0, 0, 5, 5, 5, 4, 4, 5,],
                         [2, 5, 5, 0, 0, 0, 5, 5, 5,],
                         [2, 5, 5, 0, 0, 0, 5, 5, 5,],
                         [2, 5, 5, 0, 0, 0, 5, 5, 5,],
                         [5, 4, 4, 5, 5, 5, 0, 1, 4,],
                         [5, 4, 4, 5, 5, 5, 1, 0, 4,],
                         [5, 5, 5, 5, 5, 5, 4, 4, 0,]])
        drect = pwsd.apply_pairwise_rect(seqs1=tmp, seqs2=tmp, metric=pwsd.metrics.nw_hamming_metric, ncpus=1)
        self.assertTrue(np.all(drect == res))

    def test_multiprocessing(self):
        dmat = pwsd.apply_pairwise_rect(seqs1=seqs[:10], metric=pwsd.metrics.hamming_distance, ncpus=1)
        dmat_multi = pwsd.apply_pairwise_rect(seqs1=seqs[:10], metric=pwsd.metrics.hamming_distance, ncpus=2)
        self.assertTrue(np.all(dmat == dmat_multi))

    def test_multiprocessing_sparse(self):
        dmat = pwsd.apply_pairwise_rect(seqs1=seqs[:10], metric=pwsd.metrics.hamming_distance, ncpus=1)
        dmat_multi = pwsd.apply_pairwise_rect(seqs1=seqs[:10], metric=pwsd.metrics.hamming_distance, ncpus=2)

        ind_s = list(itertools.product(range(10), range(10)))
        dmat_s = pwsd.apply_pairwise_sparse(seqs=seqs[:10], pairs=ind_s, metric=pwsd.metrics.hamming_distance, ncpus=1)
        dmat_multi_s = pwsd.apply_pairwise_sparse(seqs=seqs[:10], pairs=ind_s, metric=pwsd.metrics.hamming_distance, ncpus=2)
        self.assertTrue(np.all(dmat == dmat_multi))
        self.assertTrue(np.all(dmat_s == dmat_multi_s))
        self.assertTrue(np.all(dmat == dmat_s.reshape((10, 10))))

    def test_multiprocessing_parasail(self):
        dmat = pwsd.apply_pairwise_rect(seqs1=mixed_seqs[:20], metric=pwsd.metrics.nw_metric, matrix='blosum62', ncpus=1)
        dmat_multi = pwsd.apply_pairwise_rect(seqs1=mixed_seqs[:20], metric=pwsd.metrics.nw_metric, matrix='blosum62', ncpus=2)
        self.assertTrue(np.all(dmat == dmat_multi))

    def test_pw_rect(self):
        drect = pwsd.apply_pairwise_rect(seqs1=seqs[:10], seqs2=seqs[:20], metric=pwsd.metrics.hamming_distance, ncpus=1)
        self.assertTrue(drect.shape == (10, 20))

    def test_multiprocessing_more(self):
        dmat_multi = pwsd.apply_pairwise_rect(seqs1=mixed_seqs, metric=pwsd.metrics.nw_metric, matrix='blosum62', ncpus=2)
        dmat = pwsd.apply_pairwise_rect(seqs1=mixed_seqs, metric=pwsd.metrics.nw_metric, matrix='blosum62', ncpus=1)
        self.assertTrue(np.all(dmat == dmat_multi)) 


if __name__ == '__main__':
    unittest.main()
