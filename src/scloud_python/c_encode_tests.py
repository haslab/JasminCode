import ctypes as ct
import numpy as np
import numpy.typing as npt

from scloudplus_params import SCloudPlusParams, scloudplus128, scloudplus192, scloudplus256, ScpParams

from encode import *

from clib import C_FFI, funD, argT, Scref, rnd_uint8, rnd_u16matrix, rnd_complex128

import pytest
testrep = 10


@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_compute_v(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    m = np.array(rnd_uint8(params.mu//8))
    (cr,) = Scref[params.l].run('compute_v', (m,))
    pr = compute_v(m, param_l)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(m)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for compute_v [scloudplus%d]: OK!" % (n,params.l))


@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_compute_w(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    v = np.array(rnd_complex128(16), dtype=np.complex128)
    (cr,) = Scref[params.l].run('compute_w', (v,))
    pr = compute_w(v, param_l)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(v)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for compute_w [scloudplus%d]: OK!" % (n,params.l))


@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_reduce_w(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    v = np.array(rnd_complex128(16), dtype=np.complex128)
    (cr,) = Scref[params.l].run('reduce_w', (v,))
    pr = reduce_w(v, param_l)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(v)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for reduce_w [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_recover_m(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    v = np.array(rnd_complex128(16), dtype=np.complex128)
    (cr,) = Scref[params.l].run('recover_m', (v,))
    pr = recover_m(v, param_l)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(v)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for recover_m [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_recover_v(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    v = np.array(rnd_complex128(16), dtype=np.complex128)
    (cr,) = Scref[params.l].run('recover_v', (v,))
    pr = recover_v(v, param_l)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(v)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for recover_v [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_bddbwn(n,param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    v = np.array(rnd_complex128(16), dtype=np.complex128)
    (cr,) = Scref[params.l].run('bddbw32', (v,32))
    pr = bddbwn(v, 32)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(v)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for bddbwn [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_msgencode(n,param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    m = np.array(rnd_uint8(params.subm*params.mu//8))
    (cr,) = Scref[params.l].run('msgencode', (m,))
    pr = scloudplus_msgencode(m,param_l)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(v)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for msgencode [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_msgdecode(n,param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    v = np.array(rnd_u16matrix((params.mbar,params.nbar)))
    (cr,) = Scref[params.l].run('msgdecode', (v,))
    pr = scloudplus_msgdecode(v,param_l)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(v)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for msgdecode [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_packpk(n,param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    B = np.array(rnd_u16matrix((params.m,params.nbar)))
    (cr,) = Scref[params.l].run('packpk', (B,))
    pr = scloudplus_packpk(B,param_l)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(B)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for packpk [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_unpackpk(n,param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    pk = np.array(rnd_uint8(params.m*params.nbar*12//8))
    (cr,) = Scref[params.l].run('unpackpk', (pk,))
    pr = scloudplus_unpackpk(pk,param_l)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(pk)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for unpackpk [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_packsk(n,param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    ST = np.array(rnd_u16matrix((params.nbar,params.n)))
    (cr,) = Scref[params.l].run('packsk', (ST,))
    pr = scloudplus_packsk(ST,param_l)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(ST)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for packsk [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_unpacksk(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    sk = np.array(rnd_uint8(params.pke_sk))
    (cr,) = Scref[params.l].run('unpacksk', (sk,))
    pr = scloudplus_unpacksk(sk,param_l)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(sk)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for unpacksk [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_compressc1(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    C1 = np.array(rnd_u16matrix((params.mbar,params.n)))
    (cr,) = Scref[params.l].run('compressc1', (C1.copy(),))
    pr = scloudplus_compressc1(C1,param_l)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(C1)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for compressc1 [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_decompressc1(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    C1bar = np.array(rnd_u16matrix((params.mbar,params.n)))
    (cr,) = Scref[params.l].run('decompressc1', (C1bar.copy(),))
    pr = scloudplus_decompressc1(C1bar,param_l)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(C1bar)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for decompressc1 [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_compressc2(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    C2 = np.array(rnd_u16matrix((params.mbar,params.nbar)))
    (cr,) = Scref[params.l].run('compressc2', (C2.copy(),))
    pr = scloudplus_compressc2(C2,param_l)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(C2)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for compressc2 [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_decompressc2(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    C2bar = np.array(rnd_u16matrix((params.mbar,params.nbar)))
    (cr,) = Scref[params.l].run('decompressc2', (C2bar.copy(),))
    pr = scloudplus_decompressc2(C2bar,param_l)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(C2bar)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for decompressc2 [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_packc1(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    C1bar = np.array(rnd_u16matrix((params.mbar,params.n)))
    for i in range(params.mbar*params.n):
      C1bar.flat[i] = C1bar.flat[i] & ((2**params.logq1)-1)
    (cr,) = Scref[params.l].run('packc1', (C1bar,))
    pr = scloudplus_packc1(C1bar,param_l)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(C1bar)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for packc1 [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_unpackc1(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    c1 = np.array(rnd_uint8(params.c1))
    (cr,) = Scref[params.l].run('unpackc1', (c1,))
    pr = scloudplus_unpackc1(c1,param_l)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(c1)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for unpackc1 [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_packc2(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    C2bar = np.array(rnd_u16matrix((params.mbar,params.nbar)))
    for i in range(params.mbar*params.nbar):
      C2bar.flat[i] = C2bar.flat[i] & ((2**params.logq2)-1)
    (cr,) = Scref[params.l].run('packc2', (C2bar,))
    pr = scloudplus_packc2(C2bar,param_l)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(C2bar)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for packc2 [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_unpackc2(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    c2 = np.array(rnd_uint8(params.c2))
    (cr,) = Scref[params.l].run('unpackc2', (c2,))
    pr = scloudplus_unpackc2(c2,param_l)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(c2)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for unpackc2 [scloudplus%d]: OK!" % (n,params.l))

