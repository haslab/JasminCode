import numpy as np
import numpy.typing as npt

from clib import rnd_uint8, rnd_u16matrix
import pytest
testrep = 2

from scloudplus_params import *

def AxStxE( A: npt.NDArray[np.uint16]
          , S: npt.NDArray[np.uint16]
          , E: npt.NDArray[np.uint16]
          , param_l: int = 128
          ) -> npt.NDArray[np.uint16]:
  """ B = A*S + E """
  params = ScpParams[param_l]
  B = np.zeros((params.m,params.nbar), dtype=np.uint16)
  assert B.shape==(params.m,params.nbar), "type error: wrong shape on B (AxStxE) "+str(B.shape)
  assert A.shape==(params.m,params.n), "type error: wrong shape on A (AxStxE) "+str(A.shape)
  assert S.shape==(params.nbar,params.n), "type error: wrong shape on S (AxStxE) "+str(S.shape)
  assert E.shape==(params.m,params.nbar), "type error: wrong shape on E (AxStxE) "+str(E.shape)
  o = 0 # E idx
  i = 0 # A idx
  while ( i < params.m*params.n ):
    j = 0 # S idx
    while ( j < params.nbar*params.n ):
      s = np.uint32(E.flat[o])
      k = 0
      while ( k < params.n ):
        kk = k
        kk += i
        x = np.uint32(A.flat[kk])
        kk = k
        kk += j
        y = np.uint32(S.flat[kk])
        x = x*y
        s += x
        k += 1
      B.flat[o] = np.uint16(s)
      o += 1
      j += params.n
    i += params.n
  return B

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_AxStxE(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    A = np.array(rnd_u16matrix((params.m,params.n)))
    S = np.array(rnd_u16matrix((params.nbar,params.n)))
    E = np.array(rnd_u16matrix((params.m,params.nbar)))
    B = AxStxE(A,S,E,param_l)
    Bref = np.dot(A,S.transpose()) + E
    assert np.array_equal(B,Bref), 'AxStxE failed\nA=\n'+str(A)+"\nS=\n"+str(S)+"\nE=\n"+str(E)+"\nB=\n"+str(B)+"\nBref=\n"+str(Bref)
  print("%d random tests for j_enc_derand_plain [scloudplus%d]: OK!" % (n,params.l))

def SxAxE( S: npt.NDArray[np.uint16]
         , A: npt.NDArray[np.uint16]
         , E1: npt.NDArray[np.uint16]
         , param_l: int = 128
         ) -> npt.NDArray[np.uint16] :
  """ C1 = S*A + E1 """ 
  params = ScpParams[param_l]
  C1 = np.zeros((params.mbar,params.n), dtype=np.uint16)
  assert C1.shape==(params.mbar,params.n), "type error: wrong shape on C1 (SxAxE) "+str(C1.shape)
  assert S.shape==(params.mbar,params.m), "type error: wrong shape on S (SxAxE) "+str(S.shape)
  assert A.shape==(params.m,params.n), "type error: wrong shape on A (SxAxE) "+str(A.shape)
  assert E1.shape==(params.mbar,params.n), "type error: wrong shape on E1 (SxAxE) "+str(E1.shape)

  o = 0 # C1 idx
  i = 0 # S idx
  while ( i < params.mbar*params.m ):
    k = 0
    while ( k < params.n ):
      kk = k
      kk += o
      t = E1.flat[kk]
      C1.flat[kk] = t
      k += 1
    j = 0 # B idx
    jj = 0
    while ( j < params.m ): # j <- [0..n[, jj <- [0,n..m*n[
      kk = j
      kk += i
      x = np.uint32(S.flat[kk])
      k = 0
      while ( k < params.n ):
        kk = k
        kk += jj
        y = np.uint32(A.flat[kk])
        y = y*x
        kk = k
        kk += o
        C1.flat[kk] += np.uint16(y)
        k += 1
      j += 1
      jj += params.n
    o += params.n
    i += params.m
  return C1

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_SxAxE(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    S = np.array(rnd_u16matrix((params.mbar,params.m)))
    A = np.array(rnd_u16matrix((params.m,params.n)))
    E1 = np.array(rnd_u16matrix((params.mbar,params.n)))
    C1 = SxAxE(S,A,E1,param_l)
    C1ref = np.dot(S,A) + E1
    assert np.array_equal(C1,C1ref), 'SxAxE failed\nS=\n'+str(S)+"\nA=\n"+str(A)+"\nE1=\n"+str(E1)+"\nC1=\n"+str(C1)+"\nC1ref=\n"+str(C1ref)
  print("%d random tests for SxAxE [scloudplus%d]: OK!" % (n,params.l))

def SxBxExM( S: npt.NDArray[np.uint16]
           , A: npt.NDArray[np.uint16]
           , B: npt.NDArray[np.uint16]
           , E2: npt.NDArray[np.uint16]
           , M: npt.NDArray[np.uint16]
           , param_l: int = 128
           ) -> npt.NDArray[np.uint16] :
  """ C2 = S*B + E2 + M """
  params = ScpParams[param_l]
  C2 = np.zeros((params.mbar,params.nbar), dtype=np.uint16)
  assert C2.shape==(params.mbar,params.nbar), "type error: wrong shape on C2 (SxBxExM) "+str(C2.shape)
  assert S.shape==(params.mbar,params.m), "type error: wrong shape on S (SxBxExM) "+str(S.shape)
  assert A.shape==(params.m,params.n), "type error: wrong shape on A (SxBxExM) "+str(A.shape)
  assert B.shape==(params.m,params.nbar), "type error: wrong shape on B (SxBxExM) "+str(B.shape)
  assert E2.shape==(params.mbar,params.nbar), "type error: wrong shape on E2 (SxBxExM) "+str(E2.shape)
  assert M.shape==(params.mbar,params.nbar), "type error: wrong shape on M (SxBxExM) "+str(M.shape)
  o = 0 # C2 idx
  i = 0 # S idx
  while ( i < params.mbar*params.m ):
    k = 0
    while ( k < params.nbar ):
      kk = k
      kk += o
      t = E2.flat[kk]
      C2.flat[kk] = t
      t = M.flat[kk]
      C2.flat[kk] += t
      k += 1
    j = 0 # B idx
    jj = 0
    while ( j < params.m ): # j <- [0..n[, jj <- [0,n..m*n[
      kk = j
      kk += i
      x = np.uint32(S.flat[kk])
      k = 0
      while ( k < params.nbar ):
        kk = k
        kk += jj
        y = np.uint32(B.flat[kk])
        y = y*x
        kk = k
        kk += o
        C2.flat[kk] += np.uint16(y)
        k += 1
      j += 1
      jj += params.nbar
    o += params.nbar
    i += params.m
  return C2

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_SxBxExM(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    S = np.array(rnd_u16matrix((params.mbar,params.m)))
    A = np.array(rnd_u16matrix((params.m,params.n)))
    B = np.array(rnd_u16matrix((params.m,params.nbar)))
    E2 = np.array(rnd_u16matrix((params.mbar,params.nbar)))
    M = np.array(rnd_u16matrix((params.mbar,params.nbar)))
    C2 = SxBxExM(S,A,B,E2,M,param_l)
    C2ref = np.dot(S,B) + E2 + M
    assert np.array_equal(C2,C2ref), 'SxBxExM failed\nS=\n'+str(S)+"\nA=\n"+str(A)+"\nB=\n"+str(B)+"\nE2=\n"+str(E2)+"\nM=\n"+str(M)+"\nC2=\n"+str(C2)+"\nC2ref=\n"+str(C2ref)
  print("%d random tests for SxBxExM [scloudplus%d]: OK!" % (n,params.l))

def C2xC1xS( C2: npt.NDArray[np.uint16]
           , C1: npt.NDArray[np.uint16]
           , S: npt.NDArray[np.uint16]
           , param_l: int = 128
           ) -> npt.NDArray[np.uint16] :
  """ D = C2 - C1 * S """
  params = ScpParams[param_l]
  D = np.zeros((params.mbar,params.nbar), dtype=np.uint16)
  assert D.shape==(params.mbar,params.nbar), "type error: wrong shape on D (C2xC1xS) "+str(D.shape)
  assert C2.shape==(params.mbar,params.nbar), "type error: wrong shape on C2 (C2xC1xS) "+str(C2.shape)
  assert C1.shape==(params.mbar,params.n), "type error: wrong shape on C1 (C2xC1xS) "+str(C1.shape)
  assert S.shape==(params.nbar,params.n), "type error: wrong shape on S (C2xC1xS) "+str(S.shape)
  o = 0 # D idx
  i = 0 # C2 idx
  while ( i < params.mbar*params.n ):
    j = 0 # S idx
    while ( j < params.nbar*params.n ):
      s = np.uint32(C2.flat[o])
      k = 0
      while ( k < params.n ):
        kk = k
        kk += i
        x = np.uint32(C1.flat[kk])
        kk = k
        kk += j
        y = np.uint32(S.flat[kk])
        x = x*y
        s -= x
        k += 1
      D.flat[o] = np.uint16(s)
      o += 1
      j += params.n
    i += params.n
  return D

@pytest.mark.parametrize("n", [testrep])
@pytest.mark.parametrize("param_l", [128,192,256])
def test_C2xC1xS(n, param_l):
  params = ScpParams[param_l]
  for _ in range(n):
    C2 = np.array(rnd_u16matrix((params.mbar,params.nbar)))
    C1 = np.array(rnd_u16matrix((params.mbar,params.n)))
    S  = np.array(rnd_u16matrix((params.nbar,params.n)))
    D = C2xC1xS(C2,C1,S,param_l)
    Dref = C2 - np.dot(C1,S.transpose())
    assert np.array_equal(D,Dref), 'C2xC1xS failed\nC2=\n'+str(C2)+"\nC1=\n"+str(C1)+"\nS=\n"+str(S)+"\nD=\n"+str(D)+"\nDref=\n"+str(Dref)
  print("%d random tests for C2xC1xS [scloudplus%d]: OK!" % (n,params.l))
  