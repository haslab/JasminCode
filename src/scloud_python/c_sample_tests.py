import ctypes as ct
import numpy as np
import numpy.typing as npt

from scloudplus_params import SCloudPlusParams, scloudplus128, scloudplus192, scloudplus256, ScpParams
from sample import *

from clib import C_FFI, funD, argT, Scref, rnd_uint8
from jlib import *

import pytest
testrep = 10


# Unit tests

#
# sample.c
#

@pytest.mark.parametrize("n", [testrep])
def test_F(n):
  for _ in range(n):
    alpha = np.array(rnd_uint8(32))
    (cr,) = Scref[128].run('F', (80,alpha,32))
    seed, r1, r2 = F(alpha)
    pr = np.array(list(seed)+list(r1)+list(r2), dtype=np.uint8)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(v)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for F: OK!" % n)

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_sampleeta1(n,param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    r2 = np.array(rnd_uint8(32))
    (cr,) = Scref[params.l].run('sampleeta1', (r2,))
    pr = scloudplus_sampleeta1(r2,param_l)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(r2)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for sampleeta1 [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_sampleeta2(n,param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    r2 = np.array(rnd_uint8(32))
    (cr1,cr2,) = Scref[params.l].run('sampleeta2', (r2,))
    pr1,pr2 = scloudplus_sampleeta2(r2,param_l)
    assert np.array_equal(cr1,pr1), 'C vs P (1)\n'+str(r2)+'\n'+str(cr1)+"\n"+str(pr1)
    assert np.array_equal(cr2,pr2), 'C vs P (2)\n'+str(r2)+'\n'+str(cr2)+"\n"+str(pr2)
  print("%d random tests for sampleeta2 [scloudplus%d]: OK!" % (n,params.l))

#
## defs. auxiliares para workaround do problema apontado em 'test_readu8ton'
def mptr(ty, sz):
    return np.ctypeslib.ndpointer(dtype=ty, ndim=1, shape=(sz,), flags='CONTIGUOUS,ALIGNED,WRITEABLE')
def cptr(ty, sz):
    return np.ctypeslib.ndpointer(dtype=ty, ndim=1, shape=(sz,), flags='CONTIGUOUS,ALIGNED')

def clib_readu8ton(params):
  f = Scref[params.l].clib['readu8ton']
  f.restype = None
  f.argtypes = [ cptr(np.uint8, params.rejblocks*136), ct.c_int, mptr(np.uint16, params.mnout), ct.POINTER(ct.c_int)]
  return f


def c_readu8ton(buf: npt.NDArray[np.uint8], param_l: int = 128) -> tuple[npt.NDArray[np.uint16], np.uint16]:
  params = ScpParams[param_l]
  np.require(buf, np.uint8, ['C'])
  assert buf.shape==((params.rejblocks*136,)), "typing error: wrong size in buf on readu8ton"
  out = np.array([0]*params.mnout, dtype=np.uint16)
  outlen = ct.c_int(0)
  np.require(out, np.uint16, ['C','W'])
  clib_readu8ton(params)(buf, params.rejblocks*136, out, ct.pointer(outlen))
  return out, outlen.value

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_readu8ton(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    buf = np.array(rnd_uint8(params.rejblocks*136))
#    buf = np.ones((params.rejblocks*136),np.uint8)
    # não percebo pq é que não está a actualizar o apontador...!!!
    #outlen = ct.c_int(0)
    #(cr,) = Scref[params.l].run('readu8ton', (buf, params.mnin, ct.pointer(outlen)))
    #crn = outlen.value
    cr, crn = c_readu8ton(buf,param_l)
    pr, prn = rej_upto(buf, params.n, param_l)
    assert crn==prn, "C vs P: wrong acceptance rate -- "+str(crn)+' vs '+str(prn)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(buf)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for readu8ton [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_readu8tom(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    buf = np.array(rnd_uint8(params.rejblocks*136))
    outlen = ct.c_int(0)
    (cr,) = Scref[params.l].run('readu8tom', (buf, params.rejblocks*136, ct.pointer(outlen)))
    crn = outlen.value
    pr, prn = rej_upto(buf, params.m, param_l)
    #assert crn==prn, "C vs P: wrong acceptance rate -- "+str(crn)+' vs '+str(prn)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(buf)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for readu8tom [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_samplepsi(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    r1 = np.array(rnd_uint8(32))
    (cr,) = Scref[params.l].run('samplepsi', (r1,))
    pr = scloud_samplepsiT(r1, param_l)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(r1)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for samplepsi [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_samplephi(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    r1 = np.array(rnd_uint8(32))
    (cr,) = Scref[params.l].run('samplephi', (r1,))
    pr = scloud_samplephi(r1, param_l)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(r1)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for samplephi [scloudplus%d]: OK!" % (n,params.l))

