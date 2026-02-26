import ctypes as ct
import numpy as np
import numpy.typing as npt

from scloudplus_params import SCloudPlusParams, scloudplus128, scloudplus192, scloudplus256, ScpParams
from pke import *

from clib import C_FFI, funD, argT, Scref, rnd_uint8
from jlib import *

import pytest
testrep = 10


# Unit tests

#
# pke.c
#

@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_correctness_plain(n,param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    alpha = np.array(list(rndgen.bytes(32)), dtype=np.uint8)
    m = np.array(list(rndgen.bytes(params.l//8)), dtype=np.uint8)
    r = np.array(list(rndgen.bytes(32)), dtype=np.uint8)
    pk, sk = keyGen_plain(alpha,param_l)
    ctxt = enc_plain(pk,m,r,param_l)
    mm = dec_plain(sk,ctxt,param_l)
    assert np.array_equal(m,mm), 'correctness error\n'+str(m)+'\n'+str(mm)
  print("%d random tests for correctness (plain) [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_correctness(n,param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    alpha = np.array(list(rndgen.bytes(32)), dtype=np.uint8)
    m = np.array(list(rndgen.bytes(params.l//8)), dtype=np.uint8)
    r = np.array(list(rndgen.bytes(32)), dtype=np.uint8)
    pk, sk = keyGen(alpha,param_l)
    ctxt = enc(pk,m,r,param_l)
    mm = dec(sk,ctxt,param_l)
    assert np.array_equal(m,mm), 'correctness error\n'+str(m)+'\n'+str(mm)
  print("%d random tests for correctness [scloudplus%d]: OK!" % (n,params.l))

