import sys
import unittest
import numpy as np
from scipy.spatial.distance import squareform
import parasail
import pytest
import pwseqdist as pwsd

try:
    import numba
    # print('pwseqdist: Successfully imported numba version %s' % (numba.__version__))
    NB_SUCCESS = True
except ImportError:
    NB_SUCCESS = False
    print('pwseqdist: Could not import numba')

try:
    numba.typed.List()
    numba_typed_list_missing = False
except AttributeError:
    numba_typed_list_missing = True



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

#class TestNumba(unittest.TestCase):
@pytest.mark.skipif(numba_typed_list_missing and NB_SUCCESS, reason = "numba.typed.List is an experimental feature that may be missing")
def test_nb_pw_sq_hamming():
    dvec = pwsd.apply_pairwise_sq(seqs[:10], pwsd.metrics.hamming_distance, ncpus=1)
    dvec_nb = pwsd.numba_tools.nb_pairwise_sq(seqs[:10], pwsd.numba_tools.nb_hamming_distance)
    assert(np.all(dvec == dvec_nb))

@pytest.mark.skipif(numba_typed_list_missing and NB_SUCCESS, reason = "numba.typed.List is an experimental feature that may be missing")
def test_nb_pw_sq():
    subst_dict = pwsd.matrices.dict_from_matrix(parasail.blosum62)
    dvec = pwsd.apply_pairwise_sq(seqs[:10], pwsd.metrics.str_subst_metric, subst_dict=subst_dict, ncpus=1)

    subst_dict = pwsd.numba_tools.nb_dict_from_matrix(parasail.blosum62)
    dvec_nb = pwsd.numba_tools.nb_pairwise_sq(seqs[:10], pwsd.numba_tools.nb_subst_metric, subst_dict)
    assert(np.all(dvec == dvec_nb))
