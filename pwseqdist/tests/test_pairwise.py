import pwseqdist as pwsd
import parasail
import numpy as np
import itertools
import pytest

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


def test_nw_cpu1_no_kwargs():
	r = pwsd.pairwise.apply_pairwise_sq(seqs   = mixed_seqs, 
										metric = pwsd.metrics.nw_metric,
										ncpus = 1)
	assert isinstance(r, np.ndarray)

expectation = pwsd.pairwise.apply_pairwise_sq(seqs   = mixed_seqs, 
										metric = pwsd.metrics.nw_metric,
										ncpus = 1)
testspace = [
	(mixed_seqs, pwsd.metrics.nw_metric, 1,  expectation, None),
	(mixed_seqs, pwsd.metrics.nw_metric, 2,  expectation, None),
	(mixed_seqs, pwsd.metrics.nw_metric, 4,  expectation, None),
	(mixed_seqs, pwsd.metrics.nw_metric, 8,  expectation, None),
	(mixed_seqs, pwsd.metrics.nw_metric, 32, expectation, None),
	(mixed_seqs, pwsd.metrics.hamming_distance, 1, expectation, AssertionError)]

@pytest.mark.parametrize("s,d,cpus,expected,expected_error", testspace)
def test_nw_multcpu_no_kwargs(s,d,cpus,expected,expected_error):
	try:
		r = pwsd.pairwise.apply_pairwise_sq(seqs   = s, 
											metric = d,
											ncpus = cpus)
		assert np.all(r == expected)
	except: 
		with pytest.raises(expected_error):
			r = pwsd.pairwise.apply_pairwise_sq(seqs   = s, 
												metric = d,
												ncpus = cpus)





def test_nw_hamming_cpu1_no_kwargs():
	r = pwsd.pairwise.apply_pairwise_sq(seqs   = mixed_seqs, 
										metric = pwsd.metrics.nw_metric,
										ncpus = 1)
	assert isinstance(r, np.ndarray)


expectation = pwsd.pairwise.apply_pairwise_sq(seqs   = mixed_seqs, 
											  metric = pwsd.metrics.nw_hamming_metric,
											  ncpus = 1)
testspace = [
	(mixed_seqs, pwsd.metrics.nw_hamming_metric, 1,  expectation, None),
	(mixed_seqs, pwsd.metrics.nw_hamming_metric, 2,  expectation, None),
	(mixed_seqs, pwsd.metrics.nw_hamming_metric, 4,  expectation, None),
	(mixed_seqs, pwsd.metrics.nw_hamming_metric, 8,  expectation, None),
	(mixed_seqs, pwsd.metrics.nw_hamming_metric, 32, expectation, None)]
@pytest.mark.parametrize("s,d,cpus,expected,expected_error", testspace)
def test_first_nw_hamming_metric_multcpu_no_kwargs(s,d,cpus,expected,expected_error):
	"""
	Tests that output is the same for a range of cpus
	"""
	try:
		r = pwsd.pairwise.apply_pairwise_sq(seqs   = s, 
											metric = d,
											ncpus = cpus)
		assert np.all(r == expected)
	except: 
		with pytest.raises(expected_error):
			r = pwsd.pairwise.apply_pairwise_sq(seqs   = s, 
												metric = d,
												ncpus = cpus)

