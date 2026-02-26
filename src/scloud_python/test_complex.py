import ctypes as ct
import numpy as np


class c_complex(ct.Structure): 
    """complex is a c structure
    https://docs.python.org/3/library/ctypes.html#module-ctypes suggests
    to use ctypes.Structure to pass structures (and, therefore, complex)
    """
    def __init__(self, c):
        self.real = c.real
        self.imag = c.imag
    _fields_ = [("real", ct.c_double),("imag", ct.c_double)]
    @property
    def value(self):
       return self.real+1j*self.imag # fields declared above
    def __repr__(self):
       return str(self.value)

c_complex_p =ct.POINTER(c_complex) # pointer to our complex
c_double_p = ct.POINTER(ct.c_double) # similar to ctypes.c_char_p, i guess?


clib = np.ctypeslib.load_library('scloudplus128lib.so','../scloud_cref')
jlib = np.ctypeslib.load_library('libjcomplex.so','../scloud_jasmin_ref')

# C_REF
c_funcs = [ ('dmyround' , 'my_round'    , ct.c_double , [ct.c_double] )
          , ('cadd'     , 'complex_add' , c_complex   , [c_complex, c_complex] )
          , ('csub'     , 'complex_sub' , c_complex   , [c_complex, c_complex] )
          , ('cmul'     , 'complex_mul' , c_complex   , [c_complex, c_complex] )
          ]

# Jasmin REF
j_funcs = [ ('cadd'      , 'jj_cadd'      , c_complex   , [c_complex, c_complex] )
          , ('csub'      , 'jj_csub'      , c_complex   , [c_complex, c_complex] )
          , ('cmul'      , 'jj_cmul'      , c_complex   , [c_complex, c_complex] )
          , ('cround'    , 'jj_cround'    , c_complex   , [c_complex] )
          , ('ctrunc'    , 'jj_ctrunc'    , c_complex   , [c_complex] )
          , ('cre'       , 'jj_cre'       , ct.c_double , [c_complex] )
          , ('cim'       , 'jj_cim'       , ct.c_double , [c_complex] )
          , ('cdist2'    , 'jj_cdist2'    , ct.c_double , [c_complex] )
          , ('dadd'      , 'j_dadd'      , ct.c_double , [ct.c_double, ct.c_double] )
          , ('dsub'      , 'j_dsub'      , ct.c_double , [ct.c_double, ct.c_double] )
          , ('dmul'      , 'j_dmul'      , ct.c_double , [ct.c_double, ct.c_double] )
          , ('deq'       , 'j_deq'       , ct.c_double , [ct.c_double, ct.c_double] )
          , ('dlt'       , 'j_dlt'       , ct.c_double , [ct.c_double, ct.c_double] )
          , ('dle'       , 'j_dle'       , ct.c_double , [ct.c_double, ct.c_double] )
          , ('dunord'    , 'j_dunord'    , ct.c_double , [ct.c_double, ct.c_double] )
          , ('dneq'      , 'j_dneq'      , ct.c_double , [ct.c_double, ct.c_double] )
          , ('dge'       , 'j_dge'       , ct.c_double , [ct.c_double, ct.c_double] )
          , ('dgt'       , 'j_dgt'       , ct.c_double , [ct.c_double, ct.c_double] )
          , ('dord'      , 'j_dord'      , ct.c_double , [ct.c_double, ct.c_double] )
          , ('dcmpmask'  , 'j_dcmpmask'  , ct.c_uint64 , [ct.c_double] )
          , ('dround'    , 'j_dround'    , ct.c_double , [ct.c_double] )
          , ('dfloor'    , 'j_dfloor'    , ct.c_double , [ct.c_double] )
          , ('dceil'     , 'j_dceil'     , ct.c_double , [ct.c_double] )
          , ('dtrunc'    , 'j_dtrunc'    , ct.c_double , [ct.c_double] )
          , ('dmyround'  , 'j_dmyround'  , ct.c_double , [ct.c_double] )
          , ('dround_u64', 'j_dround_u64', ct.c_uint64 , [ct.c_double] )
          , ('dtrunc_u64', 'j_dtrunc_u64', ct.c_uint64 , [ct.c_double] )
          , ('dfrom_u64' , 'j_dfrom_u64' , ct.c_double , [ct.c_uint64] )
          ]

class C_FFI:
  def __init__(self, lib, funcs):
    self.lib = lib
    self.funs = {}
    for f in funcs:
      fdef = lib[f[1]]
      fdef.restype = f[2]
      fdef.argtypes = f[3]
      self.funs[f[0]] = fdef
  def run(self, fname, args):
    fdef = self.funs[fname]
    r = fdef(*args)
    return r
   
C = C_FFI(clib, c_funcs)
J = C_FFI(jlib, j_funcs)


import random, math
import pytest
from collections.abc import Iterator

def iter_random_floats(n=10) -> Iterator[float]:
  yield from (random.uniform(-1e6, 1e6) for _ in range(n))
  yield from (random.randint(-1000, 1000) for _ in range(n))
  yield from [0, 1.5, 2.5, -1.5, -2.5]
  #yield from [math.inf, -math.inf] #cref does not handle +/-inf

@pytest.mark.parametrize("x", iter_random_floats())
def test_myround(x: float) -> None:
  assert C.run('dmyround', (x,))==J.run('dmyround', (x,))

def test_cadd(x, y) -> None:
  cx = c_complex(x)
  cy = c_complex(y)
  r = J.run('cadd', (cx,cy))
  assert r.value==(cx.value+cy.value), str(r)+' != '+str(cx.value+cy.value)

def test_cmul(x,y) -> None:
  cx = c_complex(x)
  cy = c_complex(y)
  r = J.run('cmul', (cx,cy))
  assert r.value==(x*y), str(r)+' != '+str(x*y)


def test_cre(x) -> None:
  cx = c_complex(x)
  r = J.run('cre', (cx,))
  assert r==cx.real, str(r)+' != '+str(cx.real)

def test_cim(x: c_complex) -> None:
  cx = c_complex(x)
  r = J.run('cim', (cx,))
  print(cx, r)
  assert r==x.imag, str(r)+' != '+str(x.imag)

def my_round(x: np.float64) -> np.float64:
 i = np.float64(int(x))
 f = x - i
 if x >= 0:
  if f >= .5:
   i += 1.0
 else:
  if f <= -0.5:
   i -= 1.0
 return i

def test_cround(n=1) -> None:
  for _ in range(n):
    cx = c_complex(np.complex128(rndgen.uniform(-11, 11),rndgen.uniform(-11, 11)))
    r = J.run('cround', (cx,))
    assert r.real==my_round(cx.real), str(r.real)+' != my_round '+str(cx.real)
    assert r.imag==my_round(cx.imag), str(r.imag)+' != my_round '+str(cx.imag)


