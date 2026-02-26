import ctypes as ct
import numpy as np
import numpy.typing as npt



#
# Infrastructure for unit testing
#

rndgen = np.random.default_rng(12345)

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

def rnd_uint8(n=1):
  return [np.uint8(x) for x in rndgen.bytes(n)]

def rnd_uint16(n=1):
  l = []
  for _ in range(n):
    bs = rndgen.bytes(2)
    l.append(np.uint16(int.from_bytes(bs)))
  return l

def rnd_uint32(n=1):
  l = []
  for _ in range(n):
    bs = rndgen.bytes(4)
    l.append(np.uint32(int.from_bytes(bs)))
  return l

def rnd_uint64(n=1):
  l = []
  for _ in range(n):
    bs = rndgen.bytes(8)
    l.append(np.uint64(int.from_bytes(bs)))
  return l

def rnd_complex128(n=1):
  l = []
  for _ in range(n):
    re = rndgen.uniform(-11, 11)
    im = rndgen.uniform(-11, 11)
    l.append(np.complex128(re,im))
  return l

def rnd_u16matrix(dim):
  return np.array([ rnd_uint16(dim[1]) for _ in range(dim[0])])

#def rnd_u16array(dim):
#  if not isinstance(dim, tuple):
#    dim = (dim,)
#  for i in range(len(dim)-1,-1,-1):


from scloudplus_params import *

class argT:
  def __init__(self, kind: str, ty: type, sz: tuple[int, ...] = (0,), io: str = 'I'):
    self.kind = kind
    self.ty = ty
    self.sz = sz
    self.io = io
  @classmethod
  def iarr(cls, ty, sz):
    return cls('Arr', ty, sz, io='I')
  @classmethod
  def oarr(cls, ty, sz):
    return cls('Arr', ty, sz, io='O')
  @classmethod
  def ioarr(cls, ty, sz):
    return cls('Arr', ty, sz, io='IO')
  @classmethod
  def cty(cls, ty):
    return cls('Ctype', ty)
  def is_iarr(self):
    return 'I' in self.io
  def is_oarr(self):
    return 'O' in self.io
  def newlvar(self):
    if self.kind != 'Arr':
      return self.ty(0)
    else:
      return np.full(self.sz, self.ty(np.int64(-1)), dtype=self.ty)
#      return np.zeros(self.sz, dtype=self.ty)
  def lvar_arg(self, args):
    lvar = ()
    larg = ()
    if self.kind == 'Arr':
      if self.is_iarr():
        arg = args[0]
        args = args[1:]
        assert arg.shape == self.sz, "Wrong shape in input arg: expected "+str(arg.shape)+", got "+str(self.sz)
        larg = (np.require(arg, self.ty, ['C', 'A']),)
        if self.is_oarr():
          lvar = (arg,)
      else:
        lvar += (self.newlvar(),)
        larg += (lvar[0],)
    else:
      arg = args[0]
      args = args[1:]
      larg += (self.ty(arg),)
    return lvar, larg, args
  def __repr__(self):
    if self.kind == 'Arr':
      if 'I' in self.io and 'O' in self.io: astr = 'ioarr'
      elif 'O' in self.io: astr = 'oarr'
      else: astr = 'iarr'
      return '%s(%s, %s)' % (astr, str(self.ty.__name__), str(self.sz))
    else:
      return str(self.ty)
  def argDesc(self, mut=False):
    #sz = int(np.prod(self.sz))
#    flags = 'CONTIGUOUS,ALIGNED' + (',WRITEABLE' if self.is_oarr() else '')
    flags = 'CONTIGUOUS' + (',WRITEABLE' if self.is_oarr() else '')
    if self.kind == 'Arr':
      #return np.ctypeslib.ndpointer(dtype=self.ty, ndim=1, shape=(sz,), flags=flags)
      return np.ctypeslib.ndpointer(dtype=self.ty, shape=self.sz, flags=flags)
    else:
      return self.ty


class funD:
  def __init__(self, name: str, cname: str, argsty: list[argT], rty: type = None, pfun = None):
    self.name = name
    self.cname = cname
    self.argsty = argsty
    self.rty = rty
    self.pfun = pfun


class C_FFI:
  def __init__(self, clib):
    self.clib = clib
    self.funs = {}
  def lvars_args(self, args, argsty):
    lvars = ()
    largs = ()
    for a in argsty:
      lvar, larg, args = a.lvar_arg(args)
      lvars += lvar
      largs += larg
    assert len(args)==0, "wrong number of arguments! "+str(args)
    return lvars, largs
  def add_fun(self, fd):
    #print("processing ", fd.name)
    cfdef = self.clib[fd.cname]
    cfdef.restype = fd.rty
    args_desc = [ a.argDesc() for a in fd.argsty ]
    cfdef.argtypes = args_desc
    def call_cfun(args):
      lvars, largs = self.lvars_args(args, fd.argsty)
      #print("calling %s with %s" % (fd.cname, str(largs)))
      rval = cfdef(*largs)
      if fd.rty != None: lvars = (rval,) + lvars
      return lvars
    self.funs[fd.name] = {'cdef':cfdef, 'fd': fd, 'cfun': call_cfun}
  def add_funs(self, fds):
    for fd in fds:
      self.add_fun(fd)
  def run(self, fname, args):
    f = self.funs[fname]['cfun']
    return f(args)







#
# C reference implementation (https://github.com/scloudplus/scloudplus)
#


def encode_funs (P: SCloudPlusParams):
  l = [ funD('compute_v','compute_v',[argT.iarr(np.uint8,(P.mu//8,)),argT.oarr(np.complex128,(16,))])
      , funD('compute_w','compute_w',[argT.iarr(np.complex128,(16,)),argT.oarr(np.uint16,(32,))])
      , funD('reduce_w','reduce_w',[argT.ioarr(np.complex128,(16,))])
      , funD('recover_m','recover_m',[argT.iarr(np.complex128,(16,)),argT.oarr(np.uint8,(P.mu//8,))])
      , funD('recover_v','recover_v',[argT.iarr(np.complex128,(16,)),argT.oarr(np.complex128,(16,))])
      , funD('msgencode','scloudplus_msgencode',[argT.iarr(np.uint8,(P.l//8,)),argT.oarr(np.uint16,(P.mbar,P.nbar))])
      , funD('bddbw32','bddbwn',[argT.iarr(np.complex128,(16,)),argT.oarr(np.complex128,(16,)),argT.cty(ct.c_int)])
      , funD('msgdecode','scloudplus_msgdecode',[argT.iarr(np.uint16,(P.mbar,P.nbar)),argT.oarr(np.uint8,(P.l//8,))])
      , funD('packpk','scloudplus_packpk', [argT.iarr(np.uint16,(P.m,P.nbar)),argT.oarr(np.uint8,((P.m*P.nbar*P.logq)//8,))])
      , funD('unpackpk','scloudplus_unpackpk', [argT.iarr(np.uint8,((P.m*P.nbar*P.logq)//8,)),argT.oarr(np.uint16,(P.m,P.nbar))])
      , funD('packsk','scloudplus_packsk', [argT.iarr(np.uint16,(P.nbar,P.n)),argT.oarr(np.uint8,(P.pke_sk,))])
      , funD('unpacksk','scloudplus_unpacksk', [argT.iarr(np.uint8,(P.pke_sk,)),argT.oarr(np.uint16,(P.nbar,P.n))])
      , funD('compressc1','scloudplus_compressc1', [argT.iarr(np.uint16,(P.mbar,P.n)),argT.oarr(np.uint16,(P.mbar,P.n))])
      , funD('decompressc1','scloudplus_decompressc1', [argT.iarr(np.uint16,(P.mbar,P.n)),argT.oarr(np.uint16,(P.mbar,P.n))])
      , funD('compressc2','scloudplus_compressc2', [argT.iarr(np.uint16,(P.mbar,P.nbar)),argT.oarr(np.uint16,(P.mbar,P.nbar))])
      , funD('decompressc2','scloudplus_decompressc2', [argT.iarr(np.uint16,(P.mbar,P.nbar)),argT.oarr(np.uint16,(P.mbar,P.nbar))])
      , funD('packc1','scloudplus_packc1', [argT.iarr(np.uint16,(P.mbar,P.n)),argT.oarr(np.uint8,(P.c1,))])
      , funD('unpackc1','scloudplus_unpackc1', [argT.iarr(np.uint8,(P.c1,)),argT.oarr(np.uint16,(P.mbar,P.n))])
      , funD('packc2','scloudplus_packc2', [argT.iarr(np.uint16,(P.mbar,P.nbar)),argT.oarr(np.uint8,(P.c2,))])
      , funD('unpackc2','scloudplus_unpackc2', [argT.iarr(np.uint8,(P.c2,)),argT.oarr(np.uint16,(P.mbar,P.nbar))])
      ]
  return l

def sample_funs (P: SCloudPlusParams):
  l = [ funD('F','scloudplus_F',[argT.oarr(np.uint8,(80,)),argT.cty(ct.c_uint64),argT.iarr(np.uint8,(32,)),argT.cty(ct.c_uint64)])
      , funD('readu8ton','readu8ton',[argT.iarr(np.uint8,(P.rejblocks*136,)),argT.cty(ct.c_int),argT.oarr(np.uint16,(P.mnout,)),argT.cty(ct.POINTER(ct.c_int))])
      , funD('readu8tom','readu8tom',[argT.iarr(np.uint8,(P.rejblocks*136,)),argT.cty(ct.c_int),argT.oarr(np.uint16,(P.mnout,)),argT.cty(ct.POINTER(ct.c_int))])
      , funD('samplepsi','scloudplus_samplepsi',[argT.iarr(np.uint8,(32,)),argT.oarr(np.uint16,(P.nbar,P.n))])
      , funD('samplephi','scloudplus_samplephi',[argT.iarr(np.uint8,(32,)),argT.oarr(np.uint16,(P.mbar,P.m))])
      , funD('sampleeta1','scloudplus_sampleeta1',[argT.iarr(np.uint8,(32,)),argT.oarr(np.uint16,(P.m,P.nbar))])
      , funD('sampleeta2','scloudplus_sampleeta2',[argT.iarr(np.uint8,(32,)),argT.oarr(np.uint16,(P.mbar,P.n)),argT.oarr(np.uint16,(P.mbar,P.nbar))])
      , funD('mul_add_as_e','scloudplus_mul_add_as_e',[argT.iarr(np.uint8,(16,)),argT.iarr(np.uint16,(P.nbar,P.n)),argT.iarr(np.uint16,(P.m,P.nbar)),argT.oarr(np.uint16,(P.m,P.nbar))])
      , funD('mul_add_sa_e','scloudplus_mul_add_sa_e',[argT.iarr(np.uint8,(16,)),argT.iarr(np.uint16,(P.mbar,P.m)),argT.iarr(np.uint16,(P.mbar,P.n)),argT.oarr(np.uint16,(P.mbar,P.n))])
      ]
  return l

clib128 = np.ctypeslib.load_library('scloudplus128lib.so','../scloud_cref')
clib192 = np.ctypeslib.load_library('scloudplus192lib.so','../scloud_cref')
clib256 = np.ctypeslib.load_library('scloudplus256lib.so','../scloud_cref')

Sc128= C_FFI(clib128)
Sc128.add_funs(encode_funs(scloudplus128))
Sc128.add_funs(sample_funs(scloudplus128))

Sc192 = C_FFI(clib192)
Sc192.add_funs(encode_funs(scloudplus192))
Sc192.add_funs(sample_funs(scloudplus192))

Sc256 = C_FFI(clib256)
Sc256.add_funs(encode_funs(scloudplus256))
Sc256.add_funs(sample_funs(scloudplus256))

Scref = {128: Sc128, 192: Sc192, 256: Sc256}
