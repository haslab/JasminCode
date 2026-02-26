import numpy as np
import numpy.typing as npt

from scloudplus_params import *
from encode import *
from sample import *

def keyGen_plain(alpha: npt.NDArray[np.uint8], param_l: int = 128, strategy: str = "C"):
  """ SCloud KeyGen without pack/unpack """
  params = ScpParams[param_l]
  assert alpha.shape==(32,), "type error: wrong shape on alpha (keyGen) "+str(alpha.shape)
  seedA, r1, r2 = F(alpha)
  A = genMat(seedA, param_l)
  S = scloud_samplepsiT(r1, param_l, strategy)
  E = scloudplus_sampleeta1(r2, param_l)
  B = np.dot(A, S.transpose()) + E
  assert B.shape==(params.m,params.nbar), "type error: wrong shape on B (keyGen) "+str(B.shape)
  assert seedA.shape==(16,), "type error: wrong shape on seedA (keyGen) "+str(seedA.shape)
  assert S.shape==(params.nbar,params.n), "type error: wrong shape on S (keyGen) "+str(S.shape)
  return (B,seedA), S

def enc_plain(Pk: tuple[npt.NDArray[np.uint16],npt.NDArray[np.uint8]], msg: npt.NDArray[np.uint8], coins: npt.NDArray[np.uint8], param_l: int = 128, strategy: str = "C") -> tuple[npt.NDArray[np.uint16],npt.NDArray[np.uint16]]:
  """ SCloud Enc without pack/unpack """
  params = ScpParams[param_l]
  (B,seedA) = Pk
  assert B.shape==(params.m,params.nbar), "type error: wrong shape on B (enc_plain) "+str(B.shape)
  assert seedA.shape==(16,), "type error: wrong shape on seedA (enc_plain) "+str(seedA.shape)
  assert msg.shape==(params.l//8,), "type error: wrong shape on msg (enc_plain) "+str(msg.shape)
  assert coins.shape==(32,), "type error: wrong shape on coins (enc_plain) "+str(coins.shape)
  A = genMat(seedA, param_l)
  _, r1, r2 = F(coins)
  S1 = scloud_samplephi(r1, param_l, strategy)
  E1, E2 = scloudplus_sampleeta2(r2, param_l)
  M = scloudplus_msgencode(msg, param_l)
  C1 = np.dot(S1,A) + E1
  C2 = np.dot(S1,B) + E2 + M
  C1bar = scloudplus_compressc1(C1,param_l)
  C2bar = scloudplus_compressc2(C2,param_l)
  assert C1bar.shape==(params.mbar,params.n), "type error: wrong shape on C1bar (enc_plain) "+str(C1bar.shape)
  assert C2bar.shape==(params.mbar,params.nbar), "type error: wrong shape on C2bar (enc_plain) "+str(C2bar.shape)
  return C1bar, C2bar

def dec_plain(Sk: npt.NDArray[np.uint16], Ctxt: tuple[npt.NDArray[np.uint16],npt.NDArray[np.uint16]], param_l: int = 128) -> npt.NDArray[np.uint8]:
  """ SCloud Dec without pack/unpack """
  params = ScpParams[param_l]
  assert Sk.shape==(params.nbar,params.n), "type error: wrong shape on Sk (dec_plain) "+str(Sk.shape)
  C1bar, C2bar = Ctxt
  assert C1bar.shape==(params.mbar,params.n), "type error: wrong shape on C1bar (dec_plain) "+str(C1bar.shape)
  assert C2bar.shape==(params.mbar,params.nbar), "type error: wrong shape on C2bar (dec_plain) "+str(C2bar.shape)
  C1 = scloudplus_decompressc1(C1bar, param_l)
  C2 = scloudplus_decompressc2(C2bar, param_l)
  D = C2 - np.dot(C1,Sk.transpose())
  m = scloudplus_msgdecode(D, param_l)
  assert m.shape==(params.l//8,), "type error: wrong shape on msg (dec_plain) "+str(m.shape)
  return m

def keyGen(alpha: npt.NDArray[np.uint8], param_l: int = 128, strategy: str = "C") -> tuple[npt.NDArray[np.uint8],npt.NDArray[np.uint8]]:
  """ SCloud KeyGen including pack/unpack """
  params = ScpParams[param_l]
  assert alpha.shape==(32,), "type error: wrong shape on seedA (keygen) "+str(alpha.shape)
  (B,seedA), S = keyGen_plain(alpha, param_l, strategy)
  pkB = scloudplus_packpk(B, param_l)
  pk = np.array(list(pkB) + list(seedA), dtype=np.uint8)
  sk = scloudplus_packsk(S, param_l)
  assert pk.shape==(params.pk,), "type error: wrong shape on pk (keygen) "+str(pk.shape)
  assert sk.shape==(params.pke_sk,), "type error: wrong shape on sk (keygen) "+str(sk.shape)
  return pk, sk

def enc(pk: npt.NDArray[np.uint8], msg: npt.NDArray[np.uint8], coins: npt.NDArray[np.uint8], param_l: int = 128, strategy: str = "C") -> npt.NDArray[np.uint8]:
  params = ScpParams[param_l]
  assert pk.shape==(params.pk,), "type error: wrong shape on pk (enc) "+str(pk.shape)
  assert msg.shape==(params.l//8,), "type error: wrong shape on msg (enc) "+str(msg.shape)
  assert coins.shape==(32,), "type error: wrong shape on coins (enc) "+str(coins.shape)
  pkB = np.array(pk[:params.pk-16],dtype=np.uint8)
  seedA = np.array(pk[params.pk-16:], dtype=np.uint8)
  Pk = (scloudplus_unpackpk(pkB,param_l), seedA)
  C1bar, C2bar = enc_plain(Pk, msg, coins, param_l, strategy)
  if strategy=="C":
    c1 = scloudplus_packc1(C1bar, param_l)
    c2 = scloudplus_packc2(C2bar, param_l)
  else:
    c1 = scloudplus_packc1_paper(C1bar, param_l)
    c2 = scloudplus_packc2_paper(C2bar, param_l)
  ctxt = np.array(list(c1)+list(c2), dtype=np.uint8)
  assert ctxt.shape==(params.c1+params.c2,), "type error: wrong shape on ctxt (enc) "+str(ctxt.shape)
  return ctxt

def dec(sk: npt.NDArray[np.uint8], ctxt: npt.NDArray[np.uint8], param_l: int = 128, strategy: str = "C") -> npt.NDArray[np.uint8]:
  """ SCloud Dec without pack/unpack """
  params = ScpParams[param_l]
  assert sk.shape==(params.pke_sk,), "type error: wrong shape on sk (dec) "+str(sk.shape)
  assert ctxt.shape==(params.c1+params.c2,), "type error: wrong shape on ctxt (dec) "+str(ctxt.shape)
  c1 = np.array(ctxt[:params.c1],dtype=np.uint8)
  c2 = np.array(ctxt[params.c1:],dtype=np.uint8)
  if strategy=="C":
    C1bar = scloudplus_unpackc1(c1, param_l)
    C2bar = scloudplus_unpackc2(c2, param_l)
  else:
    C1bar = scloudplus_unpackc1_paper(c1, param_l)
    C2bar = scloudplus_unpackc2_paper(c2, param_l)
  SkT = scloudplus_unpacksk(sk, param_l)
  m = dec_plain(SkT, (C1bar,C2bar), param_l)
  assert m.shape==(params.l//8,), "type error: wrong shape on msg (dec) "+str(m.shape)
  return m
