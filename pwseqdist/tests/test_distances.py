"""
python -m unittest pwseqdist/tests/test_distances.py
"""
import sys
import unittest
import numpy as np
from scipy.spatial.distance import squareform
import parasail

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

class TestDistances(unittest.TestCase):
    
    def test_haming_metric(self):
        self.assertTrue(pwsd.metrics.hamming_distance(seqs[0], seqs[1]) == 4)
        self.assertTrue(pwsd.metrics.hamming_distance(seqs[0], seqs[0]) == 0)

    def test_pw_sq(self):
        dvec = pwsd.apply_pairwise_sq(seqs[:10], pwsd.metrics.hamming_distance, ncpus=1)
        dmat = squareform(dvec)
        self.assertTrue(dmat.shape[0] == 10 and dmat.shape[1] == 10)

    def test_pw_sq_nonuniq(self):
        dvec = pwsd.apply_pairwise_sq(seqs[:10], pwsd.metrics.hamming_distance, ncpus=1)
        dmat = squareform(dvec)

        dvec2 = pwsd.apply_pairwise_sq(seqs[:10] + seqs[:10], pwsd.metrics.hamming_distance, ncpus=1)
        dmat2 = squareform(dvec2)

        self.assertTrue(np.all(dmat2[:10, :][:, :10] == dmat))

    def test_multiprocessing(self):
        dvec = pwsd.apply_pairwise_sq(seqs[:10], pwsd.metrics.hamming_distance, ncpus=1)
        dvec_multi = pwsd.apply_pairwise_sq(seqs[:10], pwsd.metrics.hamming_distance, ncpus=2)
        self.assertTrue(np.all(dvec == dvec_multi))

    def test_pw_rect(self):
        indices, dvec = pwsd.apply_pairwise_rect(seqs[:10], seqs[:20], pwsd.metrics.hamming_distance, ncpus=1)
        self.assertTrue(dvec.shape[0] == 200)

    def test_subst(self):
        subst_dict = pwsd.matrices.dict_from_matrix(parasail.blosum62)
        for s1, s2 in zip(seqs[-10:], seqs[:10]):
            str_d = pwsd.metrics.str_subst_metric(s1, s2, subst_dict, as_similarity=False, na_penalty=None)
            np_d = pwsd.metrics.np_subst_metric(pwsd.matrices.seq2vec(s1),
                                                pwsd.matrices.seq2vec(s2),
                                                parasail.blosum62.matrix, as_similarity=False)
            # print('%s\t%s\t%1.0f\t%1.0f' % (s1, s2, str_d, np_d))
            self.assertTrue(str_d == np_d)

    def test_nw_metric(self):
        subst_dict = pwsd.matrices.dict_from_matrix(parasail.blosum62)
        nw_d = pwsd.metrics.nw_metric(mixed_seqs[0], mixed_seqs[1], matrix=parasail.blosum62, open=3, extend=3)
        
        for s1, s2 in zip(seqs[-10:], seqs[:10]):
            nw_d = pwsd.metrics.nw_metric(s1, s2, matrix=parasail.blosum62, open=30, extend=30)
            str_d = pwsd.metrics.str_subst_metric(s1, s2, subst_dict, as_similarity=False, na_penalty=None)
            self.assertTrue(nw_d == str_d)

    def test_nw_hamming_metric(self):
        subst_dict = pwsd.matrices.dict_from_matrix(parasail.blosum62)
        nw_d = pwsd.metrics.nw_hamming_metric(mixed_seqs[0], mixed_seqs[1], matrix=parasail.blosum62, open=3, extend=3)
        
        for s1, s2 in zip(seqs[-10:], seqs[:10]):
            nw_d = pwsd.metrics.nw_hamming_metric(s1, s2, matrix=parasail.blosum62, open=30, extend=30)
            str_d = pwsd.metrics.hamming_distance(s1, s2)
            # print('%s\t%s\t%1.0f\t%1.0f' % (s1, s2, str_d, nw_d))
            self.assertTrue(nw_d == str_d)

    def test_multiprocessing_more(self):
        dvec_multi = pwsd.apply_pairwise_sq(mixed_seqs, pwsd.metrics.nw_metric, ncpus=2)
        dvec = pwsd.apply_pairwise_sq(mixed_seqs, pwsd.metrics.nw_metric, ncpus=1)
        self.assertTrue(np.all(dvec == dvec_multi)) 

if __name__ == '__main__':
    unittest.main()
