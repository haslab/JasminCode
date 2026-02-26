import ctypes as ct
import numpy as np
import numpy.typing as npt

from scloudplus_params import SCloudPlusParams, scloudplus128, scloudplus192, scloudplus256, ScpParams

from pke import *

from clib import rnd_uint8, rnd_u16matrix
from jlib import Sjref

import pytest
testrep = 4


#
# pke.jinc
#

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_j_keygen_plain(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    alpha = np.array(rnd_uint8(32))
    (cB,cseed,cS) = Sjref[param_l].run('keygen_plain', (alpha,))
    (pB,pseed), pS = keyGen_plain(alpha, param_l, strategy="Jasmin")
    assert np.array_equal(cB,pB), 'J vs P (B)\n'+str(alpha)+"\n"+str(cB)+"\n"+str(pB)
    assert np.array_equal(cseed,pseed), 'J vs P (seedA)\n'+str(alpha)+"\n"+str(cseed)+"\n"+str(pseed)
    assert np.array_equal(cS,pS), 'J vs P (sk)\n'+str(cS)+"\n"+str(pS)
  print("%d random tests for j_keyGen_plain [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_j_keygen(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    alpha = np.array(rnd_uint8(32))
    (cpk,csk) = Sjref[param_l].run('keygen', (alpha,))
    ppk, psk = keyGen(alpha, param_l, strategy="Jasmin")
    assert np.array_equal(cpk,ppk), 'J vs P (pk)\n'+str(cpk)+"\n"+str(ppk)
    assert np.array_equal(csk,psk), 'J vs P (sk)\n'+str(csk)+"\n"+str(psk)
  print("%d random tests for j_keyGen [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_j_enc_plain(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    B = np.array(rnd_u16matrix((params.m,params.nbar)))
    B = np.ones((params.m,params.nbar), dtype=np.uint16)
    seedA = np.array(rnd_uint8(16))
    seedA = np.ones((16,), dtype=np.uint8)
    msg = np.array(rnd_uint8(params.l//8))
    msg = np.ones((params.l//8,), dtype=np.uint8)
    coins = np.array(rnd_uint8(32))
    coins = np.ones((32,), dtype=np.uint8)
    (cr1,cr2) = Sjref[param_l].run('enc_derand_plain', (B, seedA, msg, coins))
    (cr1,cr2) = Sjref[param_l].run('enc_derand_plain', (B, seedA, msg, coins))
    pr1,pr2 = enc_plain((B,seedA), msg, coins, param_l, strategy="Jasmin")
    assert np.array_equal(cr1,pr1), 'J vs P(c1)\n'+str(cr1)+"\n"+str(pr1)
    assert np.array_equal(cr2,pr2), 'J vs P(c2)\n'+str(cr2)+"\n"+str(pr2)
  print("%d random tests for j_enc_derand_plain [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [0])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_j_enc(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    msg = np.array(rnd_uint8(params.l//8))
    coins = np.array(rnd_uint8(32))
    pk = np.array(rnd_uint8(params.pk))
    (cr,) = Sjref[param_l].run('enc_derand', (pk, msg, coins))
    pr = enc(pk, msg, coins, param_l, strategy="Jasmin")
    assert np.array_equal(cr,pr), 'J vs P\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for j_enc_derand [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_j_dec(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    sk = np.array(rnd_uint8(params.pke_sk))
    ctxt = np.array(rnd_uint8(params.c1+params.c2))
    (cr,) = Sjref[param_l].run('dec', (sk,ctxt))
    pr = dec(sk, ctxt, param_l, strategy="Jasmin")
    assert np.array_equal(cr,pr), 'J vs P\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for j_dec [scloudplus%d]: OK!" % (n,params.l))

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_j_encdec(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    msg = np.array(rnd_uint8(params.l//8))
    coins = np.array(rnd_uint8(32))
    alpha = np.array(rnd_uint8(32))
    (cpk,csk) = Sjref[param_l].run('keygen', (alpha,))
    (ctxt,) = Sjref[param_l].run('enc_derand', (cpk, msg, coins))
    (cr,) = Sjref[param_l].run('dec', (csk, ctxt))
    assert np.array_equal(cr,msg), 'J vs P\n'+str(cr)+"\n"+str(msg)
  print("%d random tests for correctness of keygen/enc/dec [scloudplus%d]: OK!" % (n,params.l))
