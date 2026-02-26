import sys
import numpy as np
import numpy.typing as npt
import ctypes as ct
import struct

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from jlib import Sjref

import pytest
testrep = 10
#todo: https://pypi.org/project/pytest-html-plus/ (https://marketplace.visualstudio.com/items?itemName=reporterplus.pytest-html-plus-vscode)
#or https://pypi.org/project/pytest-md-report/

from scloudplus_params import *

from clib import Scref, rndgen, rnd_complex128, rnd_uint16

#
# F
#

def F(alpha: npt.NDArray[np.uint8], workaround: bool = False) -> tuple[ npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
  assert alpha.shape==(32,), "typing error: wrong alpha size"
  digest_len = 16+32+32
  if workaround: # use Jasmin implementation to avoid the cryptography package...
    (buf,) = Sjref[128].run('F', (alpha,))
    seedA = np.array(list(buf[:16]), dtype=np.uint8)
    r1 = np.array(list(buf[16:16+32]), dtype=np.uint8)
    r2 = np.array(list(buf[16+32:]), dtype=np.uint8)
  else: # use cryptography
    shake_ctx = hashes.Hash(hashes.SHAKE256(digest_len))
    shake_ctx.update(alpha.tobytes())
    buf = shake_ctx.finalize()
    seedA = np.array(list(buf[:16]), dtype=np.uint8)
    r1 = np.array(list(buf[16:16+32]), dtype=np.uint8)
    r2 = np.array(list(buf[16+32:]), dtype=np.uint8)
  assert seedA.shape==(16,), "typing error: wrong seedA size"
  assert r1.shape==(32,), "typing error: wrong r1 size"
  assert r2.shape==(32,), "typing error: wrong r2 size"
  return seedA, r1, r2


def shake256_eta1(seed: npt.NDArray[np.uint8], param_l: int = 128, workaround: bool = False) -> npt.NDArray[np.uint8]:
  params = ScpParams[param_l]
  assert seed.shape==(32,), "typing error: wrong seed size in sampleeta1"
  hashlen = params.m * params.nbar * 2 * params.eta1 // 8
  if workaround: # use Jasmin implementation to avoid the cryptography package...
    (tmp,) = Sjref[param_l].run('shake256_eta1', (seed,))
  else: # use cryptography
    shake_ctx = hashes.Hash(hashes.SHAKE256(hashlen))
    shake_ctx.update(seed.tobytes())
    tmp_bs = shake_ctx.finalize()
    tmp = np.array(list(tmp_bs), dtype=np.uint8)
  assert tmp.shape==(hashlen,), "typing error: wrong buf size in shake256_eta1"
  return tmp

def shake256_eta2(seed: npt.NDArray[np.uint8], param_l: int = 128, workaround: bool = False) -> npt.NDArray[np.uint8]:
  params = ScpParams[param_l]
  assert seed.shape==(32,), "typing error: wrong seed size in sampleeta2"
  hashlen1 = (params.mbar * params.n * 2 * params.eta2) // 8
  hashlen2 = (params.mbar * params.nbar * 2 * params.eta2 + 7) // 8
  if workaround: # use Jasmin implementation to avoid the cryptography package...
    (tmp,) = Sjref[param_l].run('shake256_eta2', (seed,))
  else: # use cryptography
    shake_ctx = hashes.Hash(hashes.SHAKE256(hashlen1+hashlen2))
    shake_ctx.update(seed.tobytes())
    tmp_bs = shake_ctx.finalize()
    tmp = np.array(list(tmp_bs), dtype=np.uint8)
  assert tmp.shape==(hashlen1+hashlen2,), "typing error: wrong buf size in shake256_eta2"
  return tmp

def XOF_init(seed: npt.NDArray[np.uint8], param_l: int = 128, workaround: bool = False):
  if workaround:
    (st,) = Sjref[param_l].run('XOF_init', (seed,))
  else:
    st = hashes.XOFHash(hashes.SHAKE256(digest_size=sys.maxsize))
    st.update(seed.tobytes())
  return st

def XOF_rejblocks(st, param_l: int = 128, workaround: bool = False):
  params = ScpParams[param_l]
  if workaround:
    (buf,_) = Sjref[param_l].run('XOF_rejblocks', (st,))
  else:
    buf = np.array(list(st.squeeze(params.rejblocks*136)), dtype=np.uint8)
  return buf

#
# sampleeta1, sampleeta2
#

def cbd1(b: np.uint8) -> npt.NDArray[np.uint16]:
  out = np.array([0]*4, dtype=np.uint16)
  for j in range(4):
    b0 = b & 0x1
    b1 = (b >> 1) & 0x1
    out[j] = np.uint16((int(b0)-int(b1))%2**16)
    b = b >> 2;
  assert out.shape==(4,), "typing error: wrong cbd1 out size"
  return out

def cbd2(bin: np.uint8) -> npt.NDArray[np.uint16]:
  b = np.uint8(bin & 0x55)
  b += (bin >> 1) & 0x55
  out = np.zeros((2,), dtype=np.uint16)
  out[0] = np.uint16((int(b & 0x03) - int((b>>2) & 0x03))%2**16)
  out[1] = np.uint16((int((b >> 4) & 0x03) - int((b >> 6) & 0x03))%2**16)
  assert out.shape==(2,), "typing error: wrong cbd2 out size"
  return out

def cbd3(bin: np.uint32) -> npt.NDArray[np.uint16]:
  b = np.uint32(bin & 0x00249249)
  b += (bin >> 1) & 0x00249249
  b += (bin >> 2) & 0x00249249
  out = np.zeros((4,), dtype=np.uint16)
  for i in range(4):
    out[i] = np.uint16((int((b >> (6*i)) & 0x07) - int((b >> (6*i+3)) & 0x07))%2**16)
  assert out.shape==(4,), "typing error: wrong cbd3 out size"
  return out

def cbd7(bin: np.uint64) -> npt.NDArray[np.uint16]:
  b = np.uint64(0)
  b = bin & 0x2040810204081
  b += (bin >> 1) & 0x2040810204081
  b += (bin >> 2) & 0x2040810204081
  b += (bin >> 3) & 0x2040810204081
  b += (bin >> 4) & 0x2040810204081
  b += (bin >> 5) & 0x2040810204081
  b += (bin >> 6) & 0x2040810204081
  out = np.zeros((4,), dtype=np.uint16)
  for i in range(4):
    out[i] = np.uint16((int((b >> (14*i)) & 0x7F) - int((b >> (14*i+7)) & 0x7F))%2**16)
  assert out.shape==(4,), "typing error: wrong cbd7 out size"
  return out

def u32_from_3u8(bs: npt.NDArray[np.uint8]) -> np.uint32:
  assert bs.shape==(3,), "typing error: wrong number of bytes in u32_from_3u8 "+str(bs.shape)
  return np.uint32(int.from_bytes(bs,'little'))

def u64_from_7u8(bs: npt.NDArray[np.uint8]) -> np.uint64:
  assert bs.shape==(7,), "typing error: wrong number of bytes in u64_from_7u8"
  return np.uint64(int.from_bytes(bs,'little'))

def scloudplus_sampleeta1(seed: npt.NDArray[np.uint8], param_l: int = 128) -> npt.NDArray[np.uint16]:
  params = ScpParams[param_l]
  assert seed.shape==(32,), "typing error: wrong seed size in sampleeta1"
  matE = np.zeros((params.m,params.nbar), dtype=np.uint16)
  tmp = shake256_eta1(seed, param_l, workaround=True)
  tmp_idx = 0
  if params.eta1 == 2:
    for i in range(0, params.m*params.nbar, 2):
      buf = cbd2(tmp[tmp_idx])
      matE.flat[i:i+2] = buf[:]
      tmp_idx += 1
  elif params.eta1 == 3:
    for i in range(0, params.m*params.nbar, 4):
      buf = cbd3(u32_from_3u8(tmp[tmp_idx: tmp_idx+3]))
      matE.flat[i:i+4] = buf[:]
      tmp_idx += 3
  elif params.eta1 == 7:
    for i in range(0, params.m*params.nbar, 4):
      buf = cbd7(u64_from_7u8(tmp[tmp_idx: tmp_idx+7]))
      matE.flat[i:i+4] = buf[:]
      tmp_idx += 7
  else:
    assert False, "wrong params.eta1"
  assert matE.shape==(params.m,params.nbar), "typing error: wrong matE size in sampleeta1"
  return matE

def scloudplus_sampleeta2(seed: npt.NDArray[np.uint8], param_l: int = 128) -> tuple[npt.NDArray[np.uint16],npt.NDArray[np.uint16]]:
  params = ScpParams[param_l]
  assert seed.shape==(32,), "typing error: wrong seed size in sampleeta2"
  matE1 = np.zeros((params.mbar,params.n), dtype=np.uint16)
  matE2 = np.zeros((params.mbar,params.nbar), dtype=np.uint16)
  tmp = shake256_eta2(seed, param_l, workaround=True)
  tmp_idx1 = 0
  tmp_idx2 = (params.mbar * params.n * 2 * params.eta2) // 8
  if params.eta2 == 1:
    mat_idx = 0
    assert params.mbar*params.n%4 == 0, "params.mbar*params.n%4 != 0"
    for i in range(0, params.mbar*params.n, 4):
      buf = cbd1(tmp[tmp_idx1])
      matE1.flat[mat_idx:mat_idx+4] = buf[:]
      tmp_idx1 += 1
      mat_idx += 4
    mat_idx = 0
    assert params.mbar*params.nbar%4 == 0, "params.mbar*params.nbar%4 != 0"
    for i in range(0, params.mbar*params.nbar, 4):
      buf = cbd1(tmp[tmp_idx2])
      matE2.flat[mat_idx:mat_idx+4] = buf[:]
      tmp_idx2 += 1
      mat_idx += 4
  elif params.eta2 == 2:
    mat_idx = 0
    assert (params.mbar*params.n)%2==0, "params.mbar*params.n%2 != 0"
    for i in range(0, params.mbar*params.n, 2):
      buf = cbd2(tmp[tmp_idx1])
      matE1.flat[mat_idx:mat_idx+2] = buf[:]
      tmp_idx1 += 1
      mat_idx += 2
    mat_idx = 0
    assert (params.mbar*params.nbar)%2==0, "params.mbar*params.nbar%2 != 0"
    for i in range(0, params.mbar*params.nbar, 2):
      buf = cbd2(tmp[tmp_idx2])
      matE2.flat[mat_idx:mat_idx+2] = buf[:]
      tmp_idx2 += 1
      mat_idx += 2
  elif params.eta2 == 7:
    mat_idx = 0
    assert (params.mbar*params.n)%4==0, "params.mbar*params.n%4 != 0"
    for i in range(0, params.mbar*params.n, 4):
      buf = cbd7(u64_from_7u8(tmp[tmp_idx1: tmp_idx1+7]))
      matE1.flat[mat_idx:mat_idx+4] = buf[:]
      tmp_idx1 += 7
      mat_idx += 4
    mat_idx = 0
    assert (params.mbar*params.nbar)%4==0, "params.mbar*params.nbar%4 != 0"
    for i in range(0, params.mbar*params.nbar, 4):
      buf = cbd7(u64_from_7u8(tmp[tmp_idx2: tmp_idx2+7]))
      matE2.flat[mat_idx:mat_idx+4] = buf[:]
      tmp_idx2 += 7
      mat_idx += 4
  else:
    assert False, "wrong params.eta2"
  assert matE1.shape==(params.mbar,params.n), "typing error: wrong matE1 size in sampleeta2"
  assert matE2.shape==(params.mbar,params.nbar), "typing error: wrong matE2 size in sampleeta2"
  return matE1, matE2


#
# samplepsi, samplephi
#
import math
def rej_choices(n: int):
  for i in range(1,16):
    nbits = int(n**i).bit_length()
    bitsper = nbits/i
    rr = 100*((2**nbits)-n**i)/(2**nbits)
    ar = 1-rr
    fullbytes = math.lcm(8,nbits)//8
    fullblocks = math.lcm(136,fullbytes)//136
    print("pack %2d: nbits=%3d, bits/s=%.2f, rr=%.2f, bytes=%d, blocks=%d" % (i,nbits,bitsper,rr,fullbytes,fullblocks))


def rej_upto(buf: npt.NDArray[np.uint8], n: int, param_l: int = 128) -> tuple[npt.NDArray[np.uint16], int]:
  params = ScpParams[param_l]
  assert n==params.n or n==params.m, "wrong bound in rejection sampling"
  buflen = params.rejblocks*136 #680
  assert buf.shape==(buflen,), "typing error: wrong buffer size in rej_upto_n"
  out = np.array([0]*params.mnout, dtype=np.uint16)
  outlen = 0
  if params.l == 128:
    for i in range(0,buflen//7*7,7): # buflen == 97*7 + 1 == 679 + 1
      tmp32 = np.uint32(int.from_bytes(buf[i:i+4],'little')) & 0xFFFFFFF
      if tmp32 < n ** 3:
        out[outlen] = tmp32 % n
        tmp32 = tmp32 // n
        out[outlen+1] = tmp32 % n
        tmp32 = tmp32 // n
        out[outlen+2] = tmp32 % n
        outlen += 3
      tmp32 = (np.uint32(int.from_bytes(buf[i+3:i+7],'little')) >> 4) & 0xFFFFFFF
      if tmp32 < n ** 3:
        out[outlen] = tmp32 % n
        tmp32 = tmp32 // n
        out[outlen+1] = tmp32 % n
        tmp32 = tmp32 // n
        out[outlen+2] = tmp32 % n
        outlen += 3
  elif params.l == 192:
    for i in range(0,params.rejblocks*136,5):
      tmp64 = np.uint64(int.from_bytes(buf[i:i+5],'little'))
      tmp16 = np.uint16(tmp64 & 0x3FF)
      tmp64 = tmp64 >> 10
      if tmp16 < n:
        out[outlen] = tmp16
        outlen += 1
      tmp16 = np.uint16(tmp64 & 0x3FF)
      tmp64 = tmp64 >> 10
      if tmp16 < n:
        out[outlen] = tmp16
        outlen += 1
      tmp16 = np.uint16(tmp64 & 0x3FF)
      tmp64 = tmp64 >> 10
      if tmp16 < n:
        out[outlen] = tmp16
        outlen += 1
      tmp16 = np.uint16(tmp64 & 0x3FF)
      tmp64 = tmp64 >> 10
      assert tmp64==0, "expected to be zero!"
      if tmp16 < n:
        out[outlen] = tmp16
        outlen += 1
  elif params.l == 256:
    for i in range(0,buflen//51*51, 51): 
      tmp64 = (np.uint64(int.from_bytes(buf[i+0 :i+8 ],'little'))      )& 0x7FFFFFFFFFFFF
      if tmp64 < n**5:
        out[outlen+0] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+1] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+2] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+3] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+4] = tmp64 % n
        outlen += 5
      tmp64 = (np.uint64(int.from_bytes(buf[i+6 :i+14],'little')) >> 3) & 0x7FFFFFFFFFFFF
      if tmp64 < n**5:
        out[outlen+0] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+1] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+2] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+3] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+4] = tmp64 % n
        outlen += 5
      tmp64 = (np.uint64(int.from_bytes(buf[i+12:i+20],'little')) >> 6) & 0x7FFFFFFFFFFFF
      if tmp64 < n**5:
        out[outlen+0] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+1] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+2] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+3] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+4] = tmp64 % n
        outlen += 5
      tmp64 = (np.uint64(int.from_bytes(buf[i+19:i+27],'little')) >> 1) & 0x7FFFFFFFFFFFF
      if tmp64 < n**5:
        out[outlen+0] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+1] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+2] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+3] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+4] = tmp64 % n
        outlen += 5
      tmp64 = (np.uint64(int.from_bytes(buf[i+25:i+33],'little')) >> 4) & 0x7FFFFFFFFFFFF
      if tmp64 < n**5:
        out[outlen+0] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+1] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+2] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+3] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+4] = tmp64 % n
        outlen += 5
      tmp64 = (np.uint64(int.from_bytes(buf[i+31:i+39],'little')) >> 7) & 0x7FFFFFFFFFFFF
      if tmp64 < n**5:
        out[outlen+0] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+1] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+2] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+3] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+4] = tmp64 % n
        outlen += 5
      tmp64 = (np.uint64(int.from_bytes(buf[i+38:i+46],'little')) >> 2) & 0x7FFFFFFFFFFFF
      if tmp64 < n**5:
        out[outlen+0] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+1] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+2] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+3] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+4] = tmp64 % n
        outlen += 5
      tmp64 = (np.uint64(int.from_bytes(buf[i+44:i+52],'little')) >> 5) & 0x7FFFFFFFFFFFF
      if tmp64 < n**5:
        out[outlen+0] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+1] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+2] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+3] = tmp64 % n
        tmp64 = tmp64 // n
        out[outlen+4] = tmp64 % n
        outlen += 5
  else:
    assert False, "wrong params.l in rej_upto"
  return out, outlen

def bm_set(bm: npt.NDArray[np.uint64], bmn: npt.NDArray[np.uint64], idx: int, flag: bool, param_l: int = 128) -> tuple[npt.NDArray[np.uint64],npt.NDArray[np.uint64], int]:
  params = ScpParams[param_l]
  assert bm.shape==(params.bm_size,), "type error: wrong shape on bm (bm_set) "+str(bm.shape)
  assert bmn.shape==(params.bm_size,), "type error: wrong shape on bmn (bm_set) "+str(bmn.shape)
  assert idx < 64*params.bm_size, "wrong bm idx! (bm_set):"+str(idx)
  idx_i = idx // 64
  idx_mask = 1 << (idx % 64)
  x = bm[idx_i]
  if (x & idx_mask)!=0:
    r = 0
    neg_mask = 0
  else:
    r = 1
    neg_mask = idx_mask
  bm[idx_i] = x | idx_mask
  if flag:
    bmn[idx_i] = bmn[idx_i] | neg_mask
  return bm, bmn, r


def bm_dump_row(bm: npt.NDArray[np.uint64], bmn: npt.NDArray[np.uint64], rlen: int, param_l: int = 128) -> npt.NDArray[np.uint16]:
  params = ScpParams[param_l]
  assert bm.shape==(params.bm_size,), "type error: wrong shape on bm (dump_row) "+str(bm.shape)
  assert bmn.shape==(params.bm_size,), "type error: wrong shape on bmn (dump_row) "+str(bmn.shape)
  assert rlen <= 64*params.bm_size, "wrong bm size! (dump_row):"+str(rlen)
  S = np.array([0]*rlen, dtype=np.uint64)
  for i in range(rlen//64):
    bmi = bm[i]
    bmni = bmn[i]
    mask = np.uint64(1)
    for j in range(64):
      if bmni & mask != 0:
        x = np.uint16(2**16-1)
      else:
        x = np.uint16(1)
      if bmi & mask != 0:
        S[i*64+j] = x
      mask = mask << 1
  i = rlen//64
  bmi = bm[i]
  bmni = bmn[i]
  mask = 1
  for j in range(rlen%64):
    if bmni & mask != 0:
      x = np.uint16(2**16-1)
    else:
      x = np.uint16(1)
    if bmi & mask != 0:
      S[i*64+j] = x
    mask = mask << 1
  return S

def scloud_samplepsiT(seed: npt.NDArray[np.uint8], param_l: int = 128, strategy: string = "C") -> npt.NDArray[np.uint16]:
  """ Remark: generates matrix in transposed form... """
  params = ScpParams[param_l]
  assert seed.shape==(32,), "type error: wrong shape on seed (samplephi) "+str(seed.shape)
  buflen = params.rejblocks*136
  shake_ctxt = XOF_init(seed, workaround=True)
  buf = XOF_rejblocks(shake_ctxt, param_l, workaround=True)
  out, outlen = rej_upto(buf, params.n, param_l)
  matS = np.zeros((params.nbar,params.n), dtype=np.uint16)
  k = 0
  for i in range(params.nbar):
    j = 0
    while j < 2*params.h1:
      while k==outlen:
        buf = XOF_rejblocks(shake_ctxt, param_l, workaround=True)
        out, outlen = rej_upto(buf, params.n, param_l)
        k = 0
      location = out[k]
      condition = matS.flat[i*params.n+location]==0
      mask = np.uint16(-int(condition) % 2**16)
      if strategy=="C":
        matS.flat[i*params.n+location] = (matS.flat[i*params.n+location] & ~mask) | ((1-2*(j%2)) & mask)
      else:
        matS.flat[i*params.n+location] = (matS.flat[i*params.n+location] & ~mask) | ((1 if j<params.h1 else mask) & mask)
      j += condition
      k += 1
  assert matS.shape==(params.nbar,params.n), "type error: wrong shape on matS (samplepsi) "+str(matS.shape)
  return matS

def scloud_samplepsi(seed: npt.NDArray[np.uint8], param_l: int = 128) -> npt.NDArray[np.uint16]:
  S = scloud_samplepsiT(seed,param_l)
  return S.transpose()

def scloud_samplephi(seed: npt.NDArray[np.uint8], param_l: int = 128, strategy: string = "C") -> npt.NDArray[np.uint16]:
  """ Remark: generates matrix in transposed form... """
  params = ScpParams[param_l]
  assert seed.shape==(32,), "type error: wrong shape on seed (samplephi) "+str(seed.shape)
  buflen = params.rejblocks*136
  shake_ctxt = XOF_init(seed, workaround=True)
  buf = XOF_rejblocks(shake_ctxt, param_l, workaround=True)
  out, outlen = rej_upto(buf, params.m, param_l)
  matS = np.zeros((params.mbar,params.m), dtype=np.uint16)
  k = 0
  for i in range(params.mbar):
    j = 0
    while j < 2*params.h2:
      while k==outlen:
        buf = XOF_rejblocks(shake_ctxt, param_l, workaround=True)
        out, outlen = rej_upto(buf, params.m, param_l)
        k = 0
      location = out[k]
      condition = matS.flat[i*params.m+location]==0
      mask = np.uint16(-int(condition) % 2**16)
      if strategy=="C":
        matS.flat[i*params.m+location] = (matS.flat[i*params.m+location] & ~mask) | ((1-2*(j%2)) & mask)
      else:
        matS.flat[i*params.m+location] = (matS.flat[i*params.m+location] & ~mask) | ((1 if j<params.h2 else mask) & mask)
      j += condition
      k += 1
  assert matS.shape==(params.mbar,params.m), "type error: wrong shape on matS (samplephi) "+str(matS.shape)
  return matS

#
# genMat
#

def genMat_paper(seedA: npt.NDArray[np.uint8], param_l: int = 128) -> npt.NDArray[np.uint16]:
  params = ScpParams[param_l]
  assert seedA.shape==(16,), "typing error: wrong seedA size"
  A = np.zeros(shape=(params.m,params.n), dtype=np.uint16)
  cipher_ctx = Cipher(algorithms.AES(seedA.tobytes()), modes.ECB())
  encryptor_ctx = cipher_ctx.encryptor()
  assert params.m%8==0, "8 does not divide parms.m!!!"
  for i in range(0,params.m,8):
    for j in range(0,params.n):
      b = bytearray(16)
      struct.pack_into('<H', b, 0, i)
      struct.pack_into('<H', b, 2, j)
      print("IN =",b)
      buf = encryptor_ctx.update(b)
      print("OUT=",buf)
      for k in range(8):
        A[i+k][j] = np.uint16(struct.unpack_from('<H', buf, 2*k)[0])
  assert A.shape==(params.m,params.n), "typing error: wrong A size"
  return A

def genMat(seedA: npt.NDArray[np.uint8], param_l: int = 128) -> npt.NDArray[np.uint16]:
  params = ScpParams[param_l]
  assert seedA.shape==(16,), "typing error: wrong seedA size"
  A = np.zeros(shape=(params.m,params.n), dtype=np.uint16)
  cipher_ctx = Cipher(algorithms.AES(seedA.tobytes()), modes.ECB())
  encryptor_ctx = cipher_ctx.encryptor()
  assert params.n%8==0, "8 does not divide params.n!!!"
  for i in range(params.m):
    rowlen = params.n//8 # (75/112/140)
    for j in range(rowlen):
      b = bytearray(16)
      struct.pack_into('<I', b, 0, i*rowlen+j)
      buf = encryptor_ctx.update(b)
      for k in range(8):
        A[i][j*8+k] = np.uint16(struct.unpack_from('<H', buf, 2*k)[0])
  assert A.shape==(params.m,params.n), "typing error: wrong A size"
  return A

