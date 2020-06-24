import pytest
import parasail 
import pwseqdist as pw
import numpy as np


peptides = ['CACADLGAYPDKLIF','CACADLGAYPDKLIF','CACDALLAYTDKLIF',
            'CACDAVGDTLDKLIF','CACDDVTEVEGDKLIF',
            'CACDFISPSNWGIQSGRNTDKLIF','CACDPVLGDTRLTDKLIF']

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

def test_nb_dict_from_matrix():
	nb_dict = pw.numba_tools.nb_dict_from_matrix(parasail.blosum62)
	assert nb_dict['P|V'] == -2
	assert nb_dict['A|A'] == 4

def test_seq2mat():
	s2m = pw.matrices.seqs2mat(peptides)
	assert isinstance(s2m, tuple)
	assert isinstance(s2m[0], np.ndarray)

def test_distance_vec_w_nb_hamming_distance():
	s2m = pw.matrices.seqs2mat(peptides)
	dv = pw.numba_tools.nb_distance_vec(seqs_mat = s2m[0],\
		seqs_L=s2m[1],\
		indices=np.array([[0,1],[0,2],[0,3]]),\
		nb_metric=pw.numba_tools.nb_hamming_distance)
	assert isinstance(dv,np.ndarray )

def test_distance_vec_w_nb_editdistance():
	s2m = pw.matrices.seqs2mat(peptides)
	dv = pw.numba_tools.nb_distance_vec(seqs_mat = s2m[0],\
		seqs_L=s2m[1],\
		indices=np.array([[0,1],[0,2],[0,3]]),\
		nb_metric=pw.numba_tools.nb_editdistance)

def test_nb_pw_rect():
    drect = pw.apply_pairwise_rect(mixed_seqs[:2], mixed_seqs, pw.metrics.nb_editdistance, use_numba=True)
    assert drect.shape[0] == 2
    assert drect.shape[1] == len(mixed_seqs)


def test_nb_pw_sq():
	dvec = pw.apply_pairwise_sq(mixed_seqs, pw.metrics.nb_editdistance, use_numba=True)
	assert dvec.shape[0] == (len(mixed_seqs)**2 - len(mixed_seqs)) / 2

# def test_nb_pairwise_sq():
# 	x=pw.numba_tools._nb_pairwise_sq(seqs = peptides,\
# 	nb_metric = pw.numba_tools.nb_hamming_distance)
# 	print(x)


