import ctypes as ct
import numpy as np
import numpy.typing as npt

from scloudplus_params import SCloudPlusParams, scloudplus128, scloudplus192, scloudplus256, ScpParams

from sample import *

from clib import rnd_uint8
from jlib import Sjref

import pytest
testrep = 10


#
# scloudplus_keccak.jinc
#

@pytest.mark.parametrize("n", [testrep])
def test_j_F(n):
  for _ in range(n):
    alpha = np.array(rnd_uint8(32))
    (cr,) = Sjref[128].run('F', (alpha,))
    seed, r1, r2 = F(alpha)
    pr = np.array(list(seed)+list(r1)+list(r2), dtype=np.uint8)
    assert np.array_equal(cr,pr), 'J vs P\n'+str(v)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for j_F: OK!" % n)

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_j_shake256_eta1(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    alpha = np.array(rnd_uint8(32))
    (cr,) = Sjref[param_l].run('shake256_eta1', (alpha,))
    pr = shake256_eta1(alpha, param_l)
    assert np.array_equal(cr,pr), 'J vs P\n'+str(alpha)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for j_shake256_eta1 [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_j_shake256_eta2(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    alpha = np.array(rnd_uint8(32))
    (cr,) = Sjref[param_l].run('shake256_eta2', (alpha,))
    pr = shake256_eta2(alpha, param_l)
    assert np.array_equal(cr,pr), 'J vs P\n'+str(alpha)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for j_shake256_eta2 [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_j_XOF_rejblocks(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    alpha = np.array(rnd_uint8(32))
    (st,) = Sjref[param_l].run('XOF_init', (alpha,))
    (cr,_) = Sjref[param_l].run('XOF_rejblocks', (st,))
    pst = XOF_init(alpha)
    pr = XOF_rejblocks(pst, param_l)
    assert np.array_equal(cr,pr), 'J vs P\n'+str(alpha)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for j_XOF_rejblocks [scloudplus%d]: OK!" % (n,params.l))

