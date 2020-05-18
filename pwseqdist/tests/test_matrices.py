import pwseqdist as pwsd
import parasail
import numpy as np
import itertools
import pytest

def alphabet_oder():
	assert pwsd.matrices.parasail_aa_alphabet == 'ARNDCQEGHILKMFPSTWYVBZX'

def test_dict_from_matrix():

	d = pwsd.matrices.dict_from_matrix(parasail.blosum62)
	assert isinstance(d, dict)

def test_dict_from_matrix_against_external_blossum_reference():
	# See truth values: https://www.ncbi.nlm.nih.gov/books/NBK6831/figure/A551/
	# ftp://ftp.ncbi.nlm.nih.gov/blast/matrices
	d = pwsd.matrices.dict_from_matrix(parasail.blosum62)
	assert isinstance(d, dict)
	assert d[('X', 'V')] == -1
	assert d[('A', 'A')] == 4
	assert d[('N', 'N')] == 6
	assert d[('W', 'W')] == 11
	assert d[('D', 'W')] == -4
	assert d[('W', 'D')] == -4
	# No diagnol entries less than 4
	assert np.all([d[(i, i)] >=4 for i in  pwsd.matrices.parasail_aa_alphabet if i != "X"] )
	# No off diagnol entries greater than 4
	assert np.all([d[(i[0], i[1])] <=4 for i in \
	 itertools.product(pwsd.matrices.parasail_aa_alphabet, pwsd.matrices.parasail_aa_alphabet) if i[0] !=i[1]] )

def test_seq2veq():
	assert(isinstance(pwsd.matrices.seq2vec("CAT"), np.ndarray))

def test_seq2veq():
	assert np.all(pwsd.matrices.seq2vec("ARNDCQEGHILKMFPSTWYVBZX") == \
		np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,17, 18, 19, 20, 21, 22]))

def test_vec2seq():
    assert pwsd.matrices.vec2seq([ 4,  0, 16]) == 'CAT'

def test_vec2seq_out_of_range_alph24():
	with pytest.raises(ValueError) as ex:
		pwsd.matrices.vec2seq([ 4,  0, 24], alphabet = pwsd.matrices.parasail_aa_alphabet_with_unknown )
	print(ex.value)
	assert str(ex.value) == 'vex2seq only works with integers 0 to 23 corresponding with ARNDCQEGHILKMFPSTWYVBZX*'

def test_vec2seq_out_of_range_alph23():
	with pytest.raises(ValueError) as ex:
		pwsd.matrices.vec2seq([ 4,  0, 23], alphabet = pwsd.matrices.parasail_aa_alphabet )
	print(ex.value)
	assert str(ex.value) == 'vex2seq only works with integers 0 to 22 corresponding with ARNDCQEGHILKMFPSTWYVBZX'


def test_seqs2mat():
	assert np.all(pwsd.matrices.seqs2mat(["CAT","HAT"]) == np.array([[ 4,  0, 16],[ 8,  0, 16]]))
    
def test_mat2seqs():
	assert np.all(pwsd.matrices.mat2seqs(np.array([[ 4,  0, 16],[ 8,  0, 16]]) ) == ['CAT', 'HAT'])
