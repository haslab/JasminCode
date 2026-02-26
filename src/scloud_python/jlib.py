import ctypes as ct
import numpy as np
import numpy.typing as npt

from scloudplus_params import SCloudPlusParams, scloudplus128, scloudplus192, scloudplus256, ScpParams

from clib import C_FFI, funD, argT, rndgen, rnd_complex128, rnd_uint16, rnd_uint64

import pytest
testrep = 10


#
# Jasmin SCloudPlus reference implementation (https://github.com/haslab/JasminCode)
#


def encode_jfuns (P: SCloudPlusParams):
  l = [ funD('compute_v','j_compute_v',[argT.oarr(np.complex128,(16,)),argT.iarr(np.uint8,(P.mu//8,))])
      , funD('compute_w','j_compute_w',[argT.oarr(np.uint16,(32,)),argT.iarr(np.complex128,(16,))])
      , funD('reduce_w','j_reduce_w',[argT.ioarr(np.complex128,(16,))])
      , funD('recover_m','j_recover_m',[argT.oarr(np.uint8,(P.mu//8,)),argT.iarr(np.complex128,(16,))])
      , funD('recover_v','j_recover_v',[argT.oarr(np.complex128,(16,)),argT.iarr(np.complex128,(16,))])
      , funD('msgencode','j_msgencode',[argT.oarr(np.uint16,(P.mbar,P.nbar)),argT.iarr(np.uint8,(P.l//8,))])
      , funD('bddbw32','j_bddbw32',[argT.oarr(np.complex128,(16,)),argT.iarr(np.complex128,(16,))])
      , funD('msgdecode','j_msgdecode',[argT.oarr(np.uint8,(P.l//8,)),argT.iarr(np.uint16,(P.mbar,P.nbar))])
      , funD('packpk','j_packpk', [argT.oarr(np.uint8,((P.m*P.nbar*P.logq)//8,)),argT.iarr(np.uint16,(P.m,P.nbar))])
      , funD('unpackpk','j_unpackpk', [argT.oarr(np.uint16,(P.m,P.nbar)),argT.iarr(np.uint8,((P.m*P.nbar*P.logq)//8,))])
      , funD('packsk','j_packsk', [argT.oarr(np.uint8,(P.pke_sk,)),argT.iarr(np.uint16,(P.nbar,P.n))])
      , funD('unpacksk','j_unpacksk', [argT.oarr(np.uint16,(P.nbar,P.n)),argT.iarr(np.uint8,(P.pke_sk,))])
      , funD('compressc1','j_compressc1', [argT.ioarr(np.uint16,(P.mbar,P.n))])
      , funD('decompressc1','j_decompressc1', [argT.ioarr(np.uint16,(P.mbar,P.n))])
      , funD('compressc2','j_compressc2', [argT.ioarr(np.uint16,(P.mbar,P.nbar))])
      , funD('decompressc2','j_decompressc2', [argT.ioarr(np.uint16,(P.mbar,P.nbar))])
      , funD('packc1','j_packc1', [argT.oarr(np.uint8,(P.c1,)),argT.iarr(np.uint16,(P.mbar,P.n))])
      , funD('unpackc1','j_unpackc1', [argT.oarr(np.uint16,(P.mbar,P.n)),argT.iarr(np.uint8,(P.c1,))])
      , funD('packc2','j_packc2', [argT.oarr(np.uint8,(P.c2,)),argT.iarr(np.uint16,(P.mbar,P.nbar))])
      , funD('unpackc2','j_unpackc2', [argT.oarr(np.uint16,(P.mbar,P.nbar)),argT.iarr(np.uint8,(P.c2,))])
      ]
  return l

def keccak_jfuns (P: SCloudPlusParams):
  l = [ funD('F','j_F',[argT.oarr(np.uint8,(80,)),argT.iarr(np.uint8,(32,))])
      , funD('shake256_eta1','j_shake256_eta1',[argT.oarr(np.uint8,((P.m*P.nbar*2*P.eta1)//8,)),argT.iarr(np.uint8,(32,))])
      , funD('shake256_eta2','j_shake256_eta2',[argT.oarr(np.uint8,((P.mbar*(P.n+P.nbar)*2*P.eta2)//8,)),argT.iarr(np.uint8,(32,))])
      , funD('XOF_init','j_XOF_init',[argT.oarr(np.uint64,(25,)),argT.iarr(np.uint8,(32,))])
      , funD('XOF_rejblocks','j_XOF_rejblocks',[argT.oarr(np.uint8,(P.rejblocks*136,)),argT.ioarr(np.uint64,(25,))])
      ]
  return l

def sample_jfuns (P: SCloudPlusParams):
  l = [ funD('readu8ton','j_rejection_n',[argT.oarr(np.uint16,(P.mnout,)),argT.iarr(np.uint8,(P.rejblocks*136,))], rty=np.uint64)
      , funD('readu8tom','j_rejection_m',[argT.oarr(np.uint16,(P.mnout,)),argT.iarr(np.uint8,(P.rejblocks*136,))], rty=np.uint64)
      , funD('bm_set','j_bm_set',[argT.ioarr(np.uint64,(P.bm_size,)),argT.ioarr(np.uint64,(P.bm_size,)),argT.cty(ct.c_longlong),argT.cty(ct.c_longlong)], rty=np.uint64)
      , funD('bm_dump_n','j_bm_dump_n',[argT.oarr(np.uint16,(P.n,)),argT.iarr(np.uint64,(P.bm_size,)),argT.iarr(np.uint64,(P.bm_size,))])
      , funD('samplepsi','j_samplepsi',[argT.oarr(np.uint16,(P.nbar,P.n)),argT.iarr(np.uint8,(32,))])
      , funD('samplephi','j_samplephi',[argT.oarr(np.uint16,(P.mbar,P.m)),argT.iarr(np.uint8,(32,))])
      , funD('sampleeta1','j_sampleeta1',[argT.oarr(np.uint16,(P.m,P.nbar)),argT.iarr(np.uint8,(32,))])
      , funD('sampleeta2','j_sampleeta2',[argT.oarr(np.uint16,(P.mbar,P.n)),argT.oarr(np.uint16,(P.mbar,P.nbar)),argT.iarr(np.uint8,(32,))])
      , funD('genMat','j_genMat',[argT.oarr(np.uint16,(P.m,P.n)),argT.iarr(np.uint8,(16,))])
      ]
  return l

def matrix_jfuns (P: SCloudPlusParams):
  l = [ funD('AxStxE','j_AxStxE',[argT.oarr(np.uint16,(P.m,P.nbar)),argT.iarr(np.uint16,(P.m,P.n)),argT.iarr(np.uint16,(P.nbar,P.n)),argT.iarr(np.uint16,(P.m,P.nbar))])
      , funD('SxAxE','j_SxAxE',[argT.oarr(np.uint16,(P.mbar,P.n)),argT.iarr(np.uint16,(P.mbar,P.m)),argT.iarr(np.uint16,(P.m,P.n)),argT.iarr(np.uint16,(P.mbar,P.n))])
      , funD('SxBxExM','j_SxBxExM',[argT.oarr(np.uint16,(P.mbar,P.nbar)),argT.iarr(np.uint16,(P.mbar,P.m)),argT.iarr(np.uint16,(P.m,P.nbar)),argT.iarr(np.uint16,(P.mbar,P.nbar)),argT.iarr(np.uint16,(P.mbar,P.nbar))])
      , funD('C2xC1xS','j_C2xC1xS',[argT.oarr(np.uint16,(P.mbar,P.nbar)),argT.iarr(np.uint16,(P.mbar,P.nbar)),argT.iarr(np.uint16,(P.mbar,P.n)),argT.iarr(np.uint16,(P.nbar,P.n))])
      ]
  return l

def pke_jfuns (P: SCloudPlusParams):
  l = [ funD('keygen_plain','j_keygen_plain',[argT.oarr(np.uint16,(P.m,P.nbar)),argT.oarr(np.uint8,(16,)),argT.oarr(np.uint16,(P.nbar,P.n)),argT.iarr(np.uint8,(32,))])
      , funD('keygen','j_keygen',[argT.oarr(np.uint8,(P.pk,)),argT.oarr(np.uint8,(P.pke_sk,)),argT.iarr(np.uint8,(32,))])
      , funD('enc_derand_plain','j_enc_derand_plain',[argT.oarr(np.uint16,(P.mbar,P.n)),argT.oarr(np.uint16,(P.mbar,P.nbar)),argT.iarr(np.uint16,(P.m,P.nbar)),argT.iarr(np.uint8,(16,)),argT.iarr(np.uint8,(P.l//8,)),argT.iarr(np.uint8,(32,))])
      , funD('enc_derand','j_enc_derand',[argT.oarr(np.uint8,(P.c1+P.c2,)),argT.iarr(np.uint8,(P.pk,)),argT.iarr(np.uint8,(P.l/8,)),argT.iarr(np.uint8,(32,))])
      , funD('dec_plain','j_dec_plain',[argT.oarr(np.uint8,(P.ss,)),argT.iarr(np.uint16,(P.nbar,P.n)),argT.iarr(np.uint16,(P.mbar, P.n)),argT.iarr(np.uint16,(P.mbar,P.nbar))])
      , funD('dec','j_dec',[argT.oarr(np.uint8,(P.ss,)),argT.iarr(np.uint8,(P.pke_sk,)),argT.iarr(np.uint8,(P.c1+P.c2,))])
      ]
  return l


jlib128 = np.ctypeslib.load_library('libjscloud128.so','../scloud_jasmin_ref')
jlib192 = np.ctypeslib.load_library('libjscloud192.so','../scloud_jasmin_ref')
jlib256 = np.ctypeslib.load_library('libjscloud256.so','../scloud_jasmin_ref')

Sj128= C_FFI(jlib128)
Sj128.add_funs(encode_jfuns(scloudplus128))
Sj128.add_funs(keccak_jfuns(scloudplus128))
Sj128.add_funs(sample_jfuns(scloudplus128))
Sj128.add_funs(matrix_jfuns(scloudplus128))
Sj128.add_funs(pke_jfuns(scloudplus128))

Sj192 = C_FFI(jlib192)
Sj192.add_funs(encode_jfuns(scloudplus192))
Sj192.add_funs(keccak_jfuns(scloudplus192))
Sj192.add_funs(sample_jfuns(scloudplus192))
Sj192.add_funs(matrix_jfuns(scloudplus192))
Sj192.add_funs(pke_jfuns(scloudplus192))

Sj256 = C_FFI(jlib256)
Sj256.add_funs(encode_jfuns(scloudplus256))
Sj256.add_funs(keccak_jfuns(scloudplus256))
Sj256.add_funs(sample_jfuns(scloudplus256))
Sj256.add_funs(matrix_jfuns(scloudplus256))
Sj256.add_funs(pke_jfuns(scloudplus256))

Sjref = {128: Sj128, 192: Sj192, 256: Sj256}

