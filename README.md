# pwseqdist

[![Build Status](https://travis-ci.com/kmayerb/pwseqdist.svg?branch=master)](https://travis-ci.com/kmayerb/pwseqdist)

A small package that efficiently computes distances between protein sequences.
Can accommodate similarity matrices, sequences of different lengths and custom
metrics.

## Install

```
pip install pwseqdist
```

## Example

```python
import pwseqdist as pw
import multiprocessing
from scipy.spatial.distance import squareform

peptides = ['CACADLGAYPDKLIF','CACDALLAYTDKLIF',
            'CACDAVGDTLDKLIF','CACDDVTEVEGDKLIF',
            'CACDFISPSNWGIQSGRNTDKLIF','CACDPVLGDTRLTDKLIF']

dvec = pw.apply_pairwise_sq(seqs = peptides, 
							metric = pw.metrics.nw_hamming_metric, 
							ncpus  = multiprocessing.cpu_count() )

dmat = squareform(dvec).astype(int)
dmat
```

```
array([[ 0,  4,  6,  7, 15,  8],
       [ 4,  0,  5,  7, 14,  7],
       [ 6,  5,  0,  6, 14,  4],
       [ 7,  7,  6,  0, 14,  8],
       [15, 14, 14, 14,  0, 11],
       [ 8,  7,  4,  8, 11,  0]])
```
