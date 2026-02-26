import ctypes as ct
import numpy as np
import numpy.typing as npt

from scloudplus_params import SCloudPlusParams, scloudplus128, scloudplus192, scloudplus256, ScpParams
from sample import *

from clib import rnd_uint8, rnd_uint64
from jlib import Sjref

import pytest
testrep = 10


# Unit tests

#
# sample.jinc
#

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,256])
def test_j_sampleeta1(n,param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    r2 = np.array(rnd_uint8(32))
    (cr,) = Sjref[params.l].run('sampleeta1', (r2,))
    pr = scloudplus_sampleeta1(r2,param_l)
    assert np.array_equal(cr,pr), 'J vs P\n'+str(r2)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for j_sampleeta1 [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_j_sampleeta2(n,param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    r2 = np.array(rnd_uint8(32))
    (cr1,cr2,) = Sjref[params.l].run('sampleeta2', (r2,))
    pr1,pr2 = scloudplus_sampleeta2(r2,param_l)
    assert np.array_equal(cr1,pr1), 'J vs P (1)\n'+str(r2)+'\n'+str(cr1)+"\n"+str(pr1)
    assert np.array_equal(cr2,pr2), 'J vs P (2)\n'+str(r2)+'\n'+str(cr2)+"\n"+str(pr2)
  print("%d random tests for j_sampleeta2 [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_j_readu8ton(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    buf = np.array(rnd_uint8(params.rejblocks*136))
    (crn, cr) = Sjref[params.l].run('readu8ton', (buf,))
    pr, prn = rej_upto(buf, params.n, param_l)
    assert crn==prn, "CJvs P: wrong acceptance rate (upto_n) -- "+str(crn)+' vs '+str(prn)
    assert np.array_equal(cr[:crn],pr[:prn]), 'J vs P\n'+str(buf)+'\n'+str(cr)+"\n"+str(pr)
    #print("outlen=",crn,"out=", [ int(x) for x in (list(cr))[:crn]])
  print("%d random tests for j_readu8ton [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_j_readu8tom(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    buf = np.array(rnd_uint8(params.rejblocks*136))
    (crn, cr) = Sjref[params.l].run('readu8tom', (buf,))
    pr, prn = rej_upto(buf, params.m, param_l)
    assert crn==prn, "J vs P: wrong acceptance rate (upto_m) -- "+str(crn)+' vs '+str(prn)
    assert np.array_equal(cr[:crn],pr[:prn]), 'J vs P\n'+str(buf)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for j_readu8tom [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_j_bm_set(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    bm_in = np.array(rnd_uint64(params.bm_size))
    bmn_in = np.array(rnd_uint64(params.bm_size))
    idx = rndgen.integers(params.n,dtype=np.uint64)
    bmn_flag = rndgen.integers(2,dtype=np.uint64)
    cbm = bm_in.copy()
    cbmn = bmn_in.copy()
    (cr,cbm,cbmn) = Sjref[params.l].run('bm_set', (cbm,cbmn,idx,bmn_flag))
    pbm = bm_in.copy()
    pbmn = bmn_in.copy()
    (pbm,pbmn,pr) = bm_set(pbm, pbmn, idx, bmn_flag!=0, param_l)
    assert cr==pr, 'J vs P(r)\n'+str(bm_in)+'\n'+str(cr)+"\n"+str(pr)
    assert np.array_equal(cbm,pbm), 'J vs P(bm)\n'+str(bm_in)+'\n'+str(cbm)+"\n"+str(pbm)
    assert np.array_equal(cbmn,pbmn), 'J vs P(bmn)\n'+"bmn= "+str(bmn_in)+'\n'+"bm= "+str(bm_in)+'\n'+str(bmn_flag)+'\n'+str(idx)+'\n'+str(cbmn)+"\n"+str(pbmn)
  print("%d random tests for j_bm_set [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_j_bm_dump_n(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    bm = np.array(rnd_uint64(params.bm_size))
    bmn = np.array(rnd_uint64(params.bm_size))
    (cr,) = Sjref[params.l].run('bm_dump_n', (bm,bmn))
    pr = bm_dump_row(bm, bmn, params.n, param_l)
    assert np.array_equal(cr,pr), 'J vs P\n'+str(bm)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for j_bm_dump_n [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_j_samplepsi(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    r1 = np.array(rnd_uint8(32))
    (cr,) = Sjref[params.l].run('samplepsi', (r1,))
    pr = scloud_samplepsiT(r1, param_l, "Jasmin")
    assert np.array_equal(cr,pr), 'J vs P\n'+str(list(r1))+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for j_samplepsi [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_j_samplephi(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    r1 = np.array(rnd_uint8(32))
    (cr,) = Sjref[params.l].run('samplephi', (r1,))
    pr = scloud_samplephi(r1, param_l, "Jasmin")
    assert np.array_equal(cr,pr), 'J vs P\n'+str(list(r1))+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for j_samplephi [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_j_genMat(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    seedA = np.array(rnd_uint8(16))
    (cr,) = Sjref[params.l].run('genMat', (seedA,))
    pr = genMat(seedA, param_l)
    assert np.array_equal(cr,pr), 'J vs P\n'+str(list(seedA))+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for j_genMat [scloudplus%d]: OK!" % (n,params.l))


