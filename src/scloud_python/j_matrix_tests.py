import ctypes as ct
import numpy as np
import numpy.typing as npt

from scloudplus_params import SCloudPlusParams, scloudplus128, scloudplus192, scloudplus256, ScpParams

from matrix import *

from clib import rnd_u16matrix
from jlib import Sjref

import pytest
testrep = 2


#
# matrix.jinc
#

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_j_AxStxE(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    A = rnd_u16matrix((params.m,params.n))
    S = rnd_u16matrix((params.nbar,params.n))
    E = rnd_u16matrix((params.m,params.nbar))
    (cr,) = Sjref[param_l].run('AxStxE', (A,S,E))
    pr = np.dot(A,S.transpose()) + E
    assert np.array_equal(cr,pr), 'J vs P\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for j_AxStxE [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_j_SxAxE(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    S = rnd_u16matrix((params.mbar,params.m))
    A = rnd_u16matrix((params.m,params.n))
    E = rnd_u16matrix((params.mbar,params.n))
    (cr,) = Sjref[param_l].run('SxAxE', (S,A,E))
    pr = np.dot(S,A) + E
    assert np.array_equal(cr,pr), 'J vs P\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for j_SxAxE [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_j_SxBxExM(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    S = rnd_u16matrix((params.mbar,params.m))
    B = rnd_u16matrix((params.m,params.nbar))
    E = rnd_u16matrix((params.mbar,params.nbar))
    M = rnd_u16matrix((params.mbar,params.nbar))
    (cr,) = Sjref[param_l].run('SxBxExM', (S,B,E,M))
    pr = np.dot(S,B) + E + M
    assert np.array_equal(cr,pr), 'J vs P\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for j_SxBxExM [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_j_C2xC1xS(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    C2 = rnd_u16matrix((params.mbar,params.nbar))
    C1 = rnd_u16matrix((params.mbar,params.n))
    S  = rnd_u16matrix((params.nbar,params.n))
    (cr,) = Sjref[param_l].run('C2xC1xS', (C2,C1,S))
    pr = C2 - np.dot(C1,S.transpose())
    assert np.array_equal(cr,pr), 'J vs P\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for j_C2xC1xS [scloudplus%d]: OK!" % (n,params.l))

