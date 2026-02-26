import ctypes as ct
import numpy as np
import numpy.typing as npt

rndgen = np.random.default_rng(12345)

from scloudplus_params import *
from encode import *
from sample import *


import ctypes

def mptr(ty, sz):
    return np.ctypeslib.ndpointer(dtype=ty, ndim=1, shape=(sz,), flags='CONTIGUOUS,ALIGNED,WRITEABLE')
def cptr(ty, sz):
    return np.ctypeslib.ndpointer(dtype=ty, ndim=1, shape=(sz,), flags='CONTIGUOUS,ALIGNED')


clib128 = np.ctypeslib.load_library('scloudplus128lib.so','../scloud_cref')
jlib128 = np.ctypeslib.load_library('libjscloud128.so','../scloud_jasmin_ref')


clib_compute_v = clib128['compute_v']
clib_compute_v.restype = None
clib_compute_v.argtypes = [ cptr(np.uint8, 64/8) , mptr(np.complex128 , 16)]

jlib_compute_v = jlib128['j_compute_v']
jlib_compute_v.restype = None
jlib_compute_v.argtypes = [ mptr(np.complex128 , 16), cptr(np.uint8, 64/8)]

def c_compute_v(m: npt.NDArray[np.uint8]) -> npt.NDArray[np.complex128]:
  np.require(m, np.uint8, ['C'])
  v = np.array([np.complex128(0j)]*16)
  np.require(v, np.complex128, ['C', 'W'])
  clib_compute_v(m, v)
  return v

def j_compute_v(m: npt.NDArray[np.uint8]) -> npt.NDArray[np.complex128]:
  np.require(m, np.uint8, ['C'])
  v = np.array([np.complex128(0j)]*16)
  np.require(v, np.complex128, ['C', 'W'])
  jlib_compute_v(v, m)
  return v

def test_compute_v(n=1):
  for _ in range(n):
    m = np.array(list(rndgen.bytes(8)), dtype=np.uint8)
    cr = c_compute_v(m)
    jr = j_compute_v(m)
    pr = compute_v(m)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(m)+'\n'+str(cr)+"\n"+str(pr)
    assert np.array_equal(cr,jr), 'C vs J\n'+str(m)+'\n'+str(cr)+"\n"+str(jr)
  print("%d random tests OK!" % n)


 #l.append(testFunc('compute_w' , 'compute_w'            , 'j_compute_w'    ,  None,   [ mptr(np.uint16 , 32             ) , cptr(c_complex , 16            ) ]))
clib_compute_w = clib128['compute_w']
clib_compute_w.restype = None
clib_compute_w.argtypes = [cptr(np.complex128, 16), mptr(np.uint16, 32)]

jlib_compute_w = jlib128['j_compute_w']
jlib_compute_w.restype = None
jlib_compute_w.argtypes = [mptr(np.uint16, 32), cptr(np.complex128, 16)]

def c_compute_w(v: npt.NDArray[np.complex128]) -> npt.NDArray[np.uint16]:
  np.require(v, np.complex128, ['C'])
  w = np.array([np.uint16(0)]*32)
  np.require(w, np.uint16, ['C', 'W'])
  clib_compute_w(v, w)
  return w

def j_compute_w(v: npt.NDArray[np.complex128]) -> npt.NDArray[np.uint16]:
  np.require(v, np.complex128, ['C'])
  w = np.array([np.uint16(0)]*32)
  np.require(w, np.uint16, ['C', 'W'])
  jlib_compute_w(w, v)
  return w

#l.append(testFunc('reduce_w'  , 'reduce_w'             , 'j_reduce_w'     ,  None,   [ mptr(c_complex , 16             ) , cptr(c_complex , 16            ) ]))
clib_reduce_w = clib128['reduce_w']
clib_reduce_w.restype = None
clib_reduce_w.argtypes = [mptr(np.complex128, 16)]

jlib_reduce_w = jlib128['j_reduce_w']
jlib_reduce_w.restype = None
jlib_reduce_w.argtypes = [mptr(np.complex128, 16)]

def c_reduce_w(v: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
  w = np.array(v.tolist(), dtype=np.complex128)
  np.require(w, np.complex128, ['C', 'W'])
  clib_reduce_w(w)
  return w

def j_reduce_w(v: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
  w = np.array(v.tolist(), dtype=np.complex128)
  np.require(w, np.complex128, ['C', 'W'])
  jlib_reduce_w(w)
  return w

#  l.append(testFunc('recover_m' , 'recover_m'            , 'j_recover_m'    ,  None,   [ mptr(np.uint8  , p.mu//8        ) , cptr(np.uint8  , p.mu//8       ) ]))
clib_recover_m = clib128['recover_m']
clib_recover_m.restype = None
clib_recover_m.argtypes = [ cptr(np.complex128, 16), mptr(np.uint8, 8) ]

jlib_recover_m = jlib128['j_recover_m']
jlib_recover_m.restype = None
jlib_recover_m.argtypes = [ mptr(np.uint8, 8), cptr(np.complex128, 16) ]

def c_recover_m(v: npt.NDArray[np.complex128]) -> npt.NDArray[np.uint8]:
  np.require(v, np.complex128, ['C'])
  m = np.array([0]*8, dtype=np.uint8)
  np.require(m, np.uint8, ['C', 'W'])
  clib_recover_m(v, m)
  return m

def j_recover_m(v: npt.NDArray[np.complex128]) -> npt.NDArray[np.uint8]:
  np.require(v, np.complex128, ['C'])
  m = np.array([0]*8, dtype=np.uint8)
  np.require(m, np.uint8, ['C', 'W'])
  jlib_recover_m(m, v)
  return m

#  l.append(testFunc('recover_v' , 'recover_v'            , 'j_recover_v'    ,  None,   [ mptr(c_complex , 16             ) , cptr(c_complex , 16       ) ]))
clib_recover_v = clib128['recover_v']
clib_recover_v.restype = None
clib_recover_v.argtypes = [ cptr(np.complex128, 16), mptr(np.complex128, 16) ]

jlib_recover_v = jlib128['j_recover_v']
jlib_recover_v.restype = None
jlib_recover_v.argtypes = [ mptr(np.complex128, 16), cptr(np.complex128, 16) ]

def c_recover_v(w: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
  np.require(w, np.complex128, ['C'])
  v = np.array([0j]*16, dtype=np.complex128)
  np.require(v, np.complex128, ['C','W'])
  clib_recover_v(w, v)
  return v

def j_recover_v(w: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
  np.require(w, np.complex128, ['C'])
  v = np.array([0j]*16, dtype=np.complex128)
  np.require(v, np.complex128, ['C', 'W'])
  jlib_recover_v(v, w)
  return v

#  l.append(testFunc('msgencode' , 'scloudplus_msgencode' , 'j_msgencode'    ,  None,   [ mptr(np.uint16 , p.mbar*p.nbar  ) , cptr(np.uint8  , p.mu//8       ) ]))
clib_msgencode = clib128['scloudplus_msgencode']
clib_msgencode.restype = None
clib_msgencode.argtypes = [ cptr(np.uint8, 16), mptr(np.uint16, 8*8) ]

jlib_msgencode = jlib128['j_msgencode']
jlib_msgencode.restype = None
jlib_msgencode.argtypes = [ mptr(np.uint16, 8*8), cptr(np.uint8, 16) ]

def c_msgencode(m: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint16]:
  np.require(m, np.uint8, ['C'])
  v = np.array([0]*64, dtype=np.uint16)
  np.require(v, np.uint16, ['C','W'])
  clib_msgencode(m, v)
  return v

def j_msgencode(m: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint16]:
  np.require(m, np.uint8, ['C'])
  v = np.array([0]*64, dtype=np.uint16)
  np.require(v, np.uint16, ['C', 'W'])
  jlib_msgencode(v, m)
  return v



#  l.append(testFunc('bddbw32'   , 'bddbwn'               , 'j_bddbw32'      ,  None,   [ mptr(c_complex , 16             ) , cptr(c_complex , 16            ) ]))
clib_bddbw32 = clib128['bddbwn']
clib_bddbw32.restype = None
clib_bddbw32.argtypes = [ cptr(np.complex128, 16), mptr(np.complex128, 16), ct.c_int ]

jlib_bddbw32 = jlib128['j_bddbw32']
jlib_bddbw32.restype = None
jlib_bddbw32.argtypes = [ mptr(np.complex128, 16), cptr(np.complex128, 16) ]

def c_bddbw32(t: npt.NDArray[np.complex128])-> npt.NDArray[np.complex128]:
  np.require(t, np.complex128, ['C'])
  y = np.array([0j]*16, dtype=np.complex128)
  np.require(y, np.complex128, ['C','W'])
  clib_bddbw32(t, y, 32)
  return y

def j_bddbw32(t: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
  np.require(t, np.complex128, ['C'])
  y = np.array([0j]*16, dtype=np.complex128)
  np.require(y, np.complex128, ['C', 'W'])
  jlib_bddbw32(y, t)
  return y

jlib_bddbw16 = jlib128['j_bddbw16']
jlib_bddbw16.restype = None
jlib_bddbw16.argtypes = [ mptr(np.complex128, 8), cptr(np.complex128, 8) ]

def j_bddbw16(t: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
  np.require(t, np.complex128, ['C'])
  y = np.array([0j]*8, dtype=np.complex128)
  np.require(y, np.complex128, ['C', 'W'])
  jlib_bddbw16(y, t)
  return y

jlib_bddbw8 = jlib128['j_bddbw8']
jlib_bddbw8.restype = None
jlib_bddbw8.argtypes = [ mptr(np.complex128, 4), cptr(np.complex128, 4) ]

def j_bddbw8(t: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
  np.require(t, np.complex128, ['C'])
  y = np.array([0j]*4, dtype=np.complex128)
  np.require(y, np.complex128, ['C', 'W'])
  jlib_bddbw8(y, t)
  return y

jlib_bddbw4 = jlib128['j_bddbw4']
jlib_bddbw4.restype = None
jlib_bddbw4.argtypes = [ mptr(np.complex128, 2), cptr(np.complex128, 2) ]

def j_bddbw4(t: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
  np.require(t, np.complex128, ['C'])
  y = np.array([0j]*2, dtype=np.complex128)
  np.require(y, np.complex128, ['C', 'W'])
  jlib_bddbw4(y, t)
  return y

jlib_bddbw2 = jlib128['j_bddbw2']
jlib_bddbw2.restype = None
jlib_bddbw2.argtypes = [ mptr(np.complex128, 1), cptr(np.complex128, 1) ]

def j_bddbw2(t: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
  np.require(t, np.complex128, ['C'])
  y = np.array([0j]*1, dtype=np.complex128)
  np.require(y, np.complex128, ['C', 'W'])
  jlib_bddbw2(y, t)
  return y

#  l.append(testFunc('msgdecode' , 'scloudplus_msgdecode' , 'j_msgdecode'    ,  None,   [ mptr(np.uint8  , p.subm*p.mu//8 ) , cptr(np.uint16 , p.mbar*p.nbar ) ]))
clib_msgdecode = clib128['scloudplus_msgdecode']
clib_msgdecode.restype = None
clib_msgdecode.argtypes = [ cptr(np.uint16, 8*8), mptr(np.uint8, 16) ]

jlib_msgdecode = jlib128['j_msgdecode']
jlib_msgdecode.restype = None
jlib_msgdecode.argtypes = [ mptr(np.uint8, 16), cptr(np.uint16, 8*8) ]

def c_msgdecode(v: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint8]:
  np.require(v, np.uint16, ['C'])
  m = np.array([0]*16, dtype=np.uint8)
  np.require(m, np.uint8, ['C','W'])
  clib_msgdecode(v, m)
  return m

def j_msgdecode(v: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint8]:
  np.require(v, np.uint16, ['C'])
  m = np.array([0]*16, dtype=np.uint8)
  np.require(v, np.uint8, ['C', 'W'])
  jlib_msgdecode(m, v)
  return m



from collections.abc import Iterator

def iter_rnd_double() -> Iterator[np.double]:
    """Generate random positive and negative numbers."""
    # some floats
    yield from (random.uniform(-1e6, 1e6) for _ in range(20))
    # some integers
    yield from (random.randint(-1000, 1000) for _ in range(20))
    # even where these are included in our random range, likehood of any specific
    # value is very low, throw in some of the usual suspects for breaking things.
    yield 0
    yield math.inf
    yield -math.inf

def rnd_uint16(n=1):
  l = []
  for _ in range(n):
    bs = rndgen.bytes(2)
    l.append(np.uint16(int.from_bytes(bs)))
  return l

def rnd_complex128(n=1):
  l = []
  for _ in range(n):
    re = rndgen.uniform(-11, 11)
    im = rndgen.uniform(-11, 11)
    l.append(np.complex128(re,im))
  return l


def test_compute_w(n=1):
  for _ in range(n):
    v = np.array(rnd_complex128(16), dtype=np.complex128)
    cr = c_compute_w(v)
    jr = j_compute_w(v)
    pr = compute_w(v)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(v)+'\n'+str(cr)+"\n"+str(pr)
    assert np.array_equal(pr,jr), 'P vs J\n'+str(v)+'\n'+str(pr)+"\n"+str(jr)
  print("%d random tests for compute_w: OK!" % n)

def test_reduce_w(n=1):
  for _ in range(n):
    v = np.array(rnd_complex128(16), dtype=np.complex128)
    cr = c_reduce_w(v)
    jr = j_reduce_w(v)
    pr = reduce_w(v)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(v)+'\n'+str(cr)+"\n"+str(pr)
    assert np.array_equal(pr,jr), 'P vs J\n'+str(v)+'\n'+str(pr)+"\n"+str(jr)
  print("%d random tests for reduce_w: OK!" % n)

def test_recover_m(n=1):
  for _ in range(n):
    v = np.array(rnd_complex128(16), dtype=np.complex128)
    cr = c_recover_m(v)
    jr = j_recover_m(v)
    pr = recover_m(v)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(v)+'\n'+str(cr)+"\n"+str(pr)
    assert np.array_equal(pr,jr), 'P vs J\n'+str(v)+'\n'+str(pr)+"\n"+str(jr)
  print("%d random tests for recover_m: OK!" % n)

def test_recover_v(n=1):
  for _ in range(n):
    w = np.array(rnd_complex128(16), dtype=np.complex128)
    cr = c_recover_v(w)
    jr = j_recover_v(w)
    pr = recover_v(w)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(w)+'\n'+str(cr)+"\n"+str(pr)
    assert np.array_equal(cr,jr), 'C vs J\n'+str(w)+'\n'+str(cr)+"\n"+str(jr)
  print("%d random tests for recover_v: OK!" % n)

def test_msgencode(n=1):
  for _ in range(n):
    m = np.array(list(rndgen.bytes(16)), dtype=np.uint8)
    cr = c_msgencode(m)
    cr = cr.reshape(8,8)
    jr = jr = j_msgencode(m)
    jr = jr.reshape(8,8)
    pr = scloudplus_msgencode(m)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(m)+'\n'+str(cr)+"\n"+str(pr)
    assert np.array_equal(cr,jr), 'C vs J\n'+str(m)+'\n'+str(cr)+"\n"+str(jr)
  print("%d random tests for msgencode: OK!" % n)

def test_msgdecode(n=1):
  for _ in range(n):
    v = np.array(rnd_uint16(64), dtype=np.uint16).reshape((8,8))
    cr = c_msgdecode(v)
    jr = j_msgdecode(v)
    pr = scloudplus_msgdecode(v)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(v)+'\n'+str(cr)+"\n"+str(pr)
    assert np.array_equal(cr,jr), 'C vs J\n'+str(v)+'\n'+str(cr)+"\n"+str(jr)
  print("%d random tests for msgdecode: OK!" % n)

def test_bddbw32(n=1):
  for _ in range(n):
    t = np.array(rnd_complex128(16), dtype=np.complex128)
    cr = c_bddbw32(t)
    jr = j_bddbw32(t)
    pr = bddbwn(t, 32)
    #
    assert np.array_equal(cr,pr), 'C vs P\n'+str(t)+'\n'+str(cr)+"\n"+str(pr)
    assert np.array_equal(cr,jr), 'C vs J\n'+str(t)+'\n'+str(cr)+"\n"+str(jr)
  print("%d random tests for bddbw32: OK!" % n)

def test_bddbw16(n=1):
  for _ in range(n):
    t = np.array(rnd_complex128(8), dtype=np.complex128)
    jr = j_bddbw16(t)
    pr = bddbwn(t, 16)
    assert np.array_equal(pr,jr), 'P vs J\n'+str(t)+'\n'+str(pr)+"\n"+str(jr)
  print("%d random tests for bddbw16: OK!" % n)

def test_bddbw8(n=1):
  for _ in range(n):
    t = np.array(rnd_complex128(4), dtype=np.complex128)
    jr = j_bddbw8(t)
    pr = bddbwn(t, 8)
    assert np.array_equal(pr,jr), 'P vs J\n'+str(t)+'\n'+str(pr)+"\n"+str(jr)
  print("%d random tests for bddbw8: OK!" % n)

def test_bddbw4(n=1):
  for _ in range(n):
    t = np.array(rnd_complex128(2), dtype=np.complex128)
    jr = j_bddbw4(t)
    pr = bddbwn(t, 4)
    assert np.array_equal(pr,jr), 'P vs J\n'+str(t)+'\n'+str(pr)+"\n"+str(jr)
  print("%d random tests for bddbw4: OK!" % n)

def test_bddbw2(n=1):
  for _ in range(n):
    t = np.array(rnd_complex128(1), dtype=np.complex128)
    jr = j_bddbw2(t)
    pr = bddbwn(t, 2)
    assert np.array_equal(pr,jr), 'P vs J\n'+str(t)+'\n'+str(pr)+"\n"+str(jr)
  print("%d random tests for bddbw2: OK!" % n)




# sample.py

clib_F = clib128['scloudplus_F']
clib_F.restype = None
clib_F.argtypes = [ mptr(np.uint8, 80), ct.c_int64, cptr(np.uint8, 32), ct.c_int64]

def c_F(seed: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
  np.require(seed, np.uint8, ['C'])
  assert seed.shape==((32,)), "typing error: wrong seed size on F"
  r = np.array([0]*80, dtype=np.uint8)
  np.require(r, np.uint8, ['C','W'])
  clib_F(r, 80, seed, 32)
  return r

def test_F(n=1):
  for _ in range(n):
    alpha = np.array(list(rndgen.bytes(32)), dtype=np.uint8)
    cr = c_F(alpha)
    seed, r1, r2 = F(alpha)
    pr = np.array(list(seed)+list(r1)+list(r2), dtype=np.uint8)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(v)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for F: OK!" % n)


clib_readu8ton = clib128['readu8ton']
clib_readu8ton.restype = None
clib_readu8ton.argtypes = [ cptr(np.uint8, 680), ct.c_int, mptr(np.uint16, scloudplus128.mnout), ct.POINTER(ct.c_int)]

def c_readu8ton(buf: npt.NDArray[np.uint8]) -> tuple[npt.NDArray[np.uint16], np.uint16]:
  np.require(buf, np.uint8, ['C'])
  assert buf.shape==((680,)), "typing error: wrong size in buf on readu8ton"
  out = np.array([0]*scloudplus128.mnout, dtype=np.uint16)
  outlen = ct.c_int(0)
  np.require(out, np.uint16, ['C','W'])
  clib_readu8ton(buf, scloudplus128.mnin, out, ct.pointer(outlen))
  return out, outlen.value

def test_readu8ton(n=1):
  for _ in range(n):
    buf = np.array(list(rndgen.bytes(680)), dtype=np.uint8)
    cr, crn = c_readu8ton(buf)
    pr, prn = rej_upto(buf, scloudplus128.n)
    assert crn==prn, "C vs P: wrong acceptance rate -- "+str(crn)+' vs '+str(prn)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(buf)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for readu8ton: OK!" % n)


clib_samplepsi = clib128['scloudplus_samplepsi']
clib_samplepsi.restype = None
clib_samplepsi.argtypes = [ cptr(np.uint8, 32), mptr(np.uint16, scloudplus128.n*scloudplus128.nbar)]

def c_samplepsi(r1: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint16]:
  np.require(r1, np.uint8, ['C'])
  assert r1.shape==((32,)), "typing error: wrong seed r1 on samplepsi"
  S = np.array([0]*scloudplus128.n*scloudplus128.nbar, dtype=np.uint16)
  np.require(S, np.uint16, ['C','W'])
  clib_samplepsi(r1, S)
  S = S.reshape((scloudplus128.nbar,scloudplus128.n))
  S = S.transpose()
  return S

def test_samplepsi(n=1):
  for _ in range(n):
    r1 = np.array(list(rndgen.bytes(32)), dtype=np.uint8)
    cr = c_samplepsi(r1)
    pr = scloud_samplepsi(r1)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(r1)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for samplepsi: OK!" % n)

clib_samplephi = clib128['scloudplus_samplephi']
clib_samplephi.restype = None
clib_samplephi.argtypes = [ cptr(np.uint8, 32), mptr(np.uint16, scloudplus128.mbar*scloudplus128.m)]

def c_samplephi(r1: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint16]:
  np.require(r1, np.uint8, ['C'])
  assert r1.shape==((32,)), "typing error: wrong seed r1 on samplephi"
  S = np.array([0]*scloudplus128.mbar*scloudplus128.m, dtype=np.uint16)
  np.require(S, np.uint16, ['C','W'])
  clib_samplephi(r1, S)
  S = S.reshape((scloudplus128.mbar,scloudplus128.m))
  return S

def test_samplephi(n=1):
  for _ in range(n):
    r1 = np.array(list(rndgen.bytes(32)), dtype=np.uint8)
    cr = c_samplephi(r1)
    pr = scloud_samplephi(r1)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(r1)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for samplephi: OK!" % n)

clib_sampleeta1 = clib128['scloudplus_sampleeta1']
clib_sampleeta1.restype = None
clib_sampleeta1.argtypes = [ cptr(np.uint8, 32), mptr(np.uint16, scloudplus128.m*scloudplus128.nbar)]

def c_sampleeta1(r2: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint16]:
  np.require(r2, np.uint8, ['C'])
  assert r2.shape==((32,)), "typing error: wrong seed r2 on sampleeta1"
  E = np.array([0]*scloudplus128.m*scloudplus128.nbar, dtype=np.uint16)
  np.require(E, np.uint16, ['C','W'])
  clib_sampleeta1(r2, E)
  E = E.reshape((scloudplus128.m,scloudplus128.nbar))
  return E

def test_sampleeta1(n=1):
  for _ in range(n):
    r2 = np.array(list(rndgen.bytes(32)), dtype=np.uint8)
    cr = c_sampleeta1(r2)
    pr = scloudplus_sampleeta1(r2)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(r2)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for sampleeta1: OK!" % n)


clib_sampleeta2 = clib128['scloudplus_sampleeta2']
clib_sampleeta2.restype = None
clib_sampleeta2.argtypes = [ cptr(np.uint8, 32), mptr(np.uint16, scloudplus128.mbar*scloudplus128.n), mptr(np.uint16, scloudplus128.mbar*scloudplus128.nbar)]

def c_sampleeta2(r2: npt.NDArray[np.uint8]) -> tuple[npt.NDArray[np.uint16],npt.NDArray[np.uint16]]:
  np.require(r2, np.uint8, ['C'])
  assert r2.shape==((32,)), "typing error: wrong seed r2 on sampleeta1"
  E1 = np.array([0]*scloudplus128.mbar*scloudplus128.n, dtype=np.uint16)
  E2 = np.array([0]*scloudplus128.mbar*scloudplus128.nbar, dtype=np.uint16)
  np.require(E1, np.uint16, ['C','W'])
  np.require(E2, np.uint16, ['C','W'])
  clib_sampleeta2(r2, E1, E2)
  E1 = E1.reshape((scloudplus128.mbar,scloudplus128.n))
  E2 = E2.reshape((scloudplus128.mbar,scloudplus128.nbar))
  return E1, E2

def test_sampleeta2(n=1):
  for _ in range(n):
    r2 = np.array(list(rndgen.bytes(32)), dtype=np.uint8)
    cr1, cr2 = c_sampleeta2(r2)
    pr1, pr2 = scloudplus_sampleeta2(r2)
    assert np.array_equal(cr1,pr1), 'C vs P(1)\n'+str(r2)+'\n'+str(cr1)+"\n"+str(pr1)
    assert np.array_equal(cr2,pr2), 'C vs P(2)\n'+str(r2)+'\n'+str(cr2)+"\n"+str(pr2)
  print("%d random tests for sampleeta2: OK!" % n)



clib_mul_add_as_e = clib128['scloudplus_mul_add_as_e']
clib_mul_add_as_e.restype = None
clib_mul_add_as_e.argtypes = [ cptr(np.uint8, 16), cptr(np.uint16, scloudplus128.n*scloudplus128.nbar), cptr(np.uint16, scloudplus128.m*scloudplus128.nbar), mptr(np.uint16, scloudplus128.m*scloudplus128.nbar)]

def c_mul_add_as_e(seed: npt.NDArray[np.uint8], S: npt.NDArray[np.uint16], E: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint16]:
  np.require(seed, np.uint8, ['C'])
  assert seed.shape==((16,)), "typing error: wrong seed seed on mul_add_as_e"
  np.require(S, np.uint16, ['C'])
  S = S.transpose()
  assert S.shape==(scloudplus128.nbar,scloudplus128.n), "typing error: wrong dim for S on mul_add_as_e"
  S = S.reshape((scloudplus128.n*scloudplus128.nbar,))
  np.require(E, np.uint16, ['C'])
  assert E.shape==(scloudplus128.m,scloudplus128.nbar), "typing error: wrong dim for E on mul_add_as_e"
  E = E.reshape((scloudplus128.m*scloudplus128.nbar,))
  B = np.array([0]*scloudplus128.m*scloudplus128.nbar, dtype=np.uint16)
  np.require(B, np.uint16, ['C','W'])
  clib_mul_add_as_e(seed, S, E, B)
  B = B.reshape((scloudplus128.m,scloudplus128.nbar))
  return B

def test_mul_add_as_e(n=1):
  for _ in range(n):
    alpha = np.array(list(rndgen.bytes(32)), dtype=np.uint8)
    seed, r1, r2 = F(alpha)
    S = scloud_samplepsi(r1)
    E = scloudplus_sampleeta1(r2)
    A = genMat(seed, scloudplus128)
    pr = np.dot(A,S) + E
    cr = c_mul_add_as_e(seed, S, E)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(alpha)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for mul_add_as_e: OK!" % n)

clib_mul_add_sa_e = clib128['scloudplus_mul_add_sa_e']
clib_mul_add_sa_e.restype = None
clib_mul_add_sa_e.argtypes = [ cptr(np.uint8, 16), cptr(np.uint16, scloudplus128.mbar*scloudplus128.m), cptr(np.uint16, scloudplus128.mbar*scloudplus128.n), mptr(np.uint16, scloudplus128.mbar*scloudplus128.n)]

def c_mul_add_sa_e(seed: npt.NDArray[np.uint8], S: npt.NDArray[np.uint16], E: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint16]:
  np.require(seed, np.uint8, ['C'])
  assert seed.shape==((16,)), "typing error: wrong seed seed on mul_add_sa_e"
  np.require(S, np.uint16, ['C'])
  assert S.shape==(scloudplus128.mbar,scloudplus128.m), "typing error: wrong dim for S on mul_add_sa_e"
  S = S.reshape((scloudplus128.mbar*scloudplus128.m,))
  np.require(E, np.uint16, ['C'])
  assert E.shape==(scloudplus128.mbar,scloudplus128.n), "typing error: wrong dim for E on mul_add_sa_e"
  E = E.reshape((scloudplus128.mbar*scloudplus128.n,))
  C = np.array([0]*scloudplus128.mbar*scloudplus128.n, dtype=np.uint16)
  np.require(C, np.uint16, ['C','W'])
  clib_mul_add_sa_e(seed, S, E, C)
  C = C.reshape((scloudplus128.mbar,scloudplus128.n))
  return C

def test_mul_add_sa_e(n=1):
  for _ in range(n):
    alpha = np.array(list(rndgen.bytes(32)), dtype=np.uint8)
    seed, r1, r2 = F(alpha)
    S = scloud_samplephi(r1)
    E1, E2 = scloudplus_sampleeta2(r2)
    A = genMat(seed, scloudplus128)
    pr = np.dot(S,A) + E1
    cr = c_mul_add_sa_e(seed, S, E1)
    assert np.array_equal(cr,pr), 'C vs P\n'+str(alpha)+'\n'+str(cr)+"\n"+str(pr)
  print("%d random tests for mul_add_sa_e: OK!" % n)



