import numpy as np
import numpy.typing as npt

import pytest
testrep = 10
#todo: https://pypi.org/project/pytest-html-plus/ (https://marketplace.visualstudio.com/items?itemName=reporterplus.pytest-html-plus-vscode)
#or https://pypi.org/project/pytest-md-report/

from scloudplus_params import *

from clib import Scref, rndgen, rnd_complex128, rnd_uint16

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

#
# compute_v
#

def compute_v(m: npt.NDArray[np.uint8], param_l: int = 128) -> npt.NDArray[np.complex128]:
 params = ScpParams[param_l]
 # (m: u8[params.mu//8]) -> complex[16]
 assert m.shape == (params.mu//8,), "type error: wrong 'm' shape (compute_v)! "+str(m.shape)
 A = np.array([np.uint8(0)]*6)
 B = np.array([np.uint8(0)]*20)
 C = np.array([np.uint8(0)]*6)

 if params.tau == 3:
  A[0] = (m[0] >> 0) & 0x07
  A[1] = (m[0] >> 3) & 0x07
  A[2] = ((m[0] >> 6) & 0x03) | ((m[1] << 2) & 0x04)
  A[3] = (m[1] >> 1) & 0x07
  A[4] = (m[1] >> 4) & 0x07
  A[5] = ((m[1] >> 7) & 0x01) | ((m[2] << 1) & 0x06)

  for i in range(3):
   B[i] = (m[2] >> (2 + 2 * i)) & 0x03
  for i in range(4):
   B[3 + i] = (m[3] >> (2 * i)) & 0x03
   B[7 + i] = (m[4] >> (2 * i)) & 0x03
   B[11 + i] = (m[5] >> (2 * i)) & 0x03
   B[15 + i] = (m[6] >> (2 * i)) & 0x03

  B[19] = m[7] & 0x03
  C[0] = (m[7] >> 2) & 0x01
  C[1] = (m[7] >> 3) & 0x01
  C[2] = (m[7] >> 4) & 0x01
  C[3] = (m[7] >> 5) & 0x01
  C[4] = (m[7] >> 6) & 0x01
  C[5] = (m[7] >> 7) & 0x01
 elif params.tau == 4:
  A[0] = m[0] & 0x0F
  A[1] = (m[0] >> 4) & 0x0F
  A[2] = m[1] & 0x0F
  A[3] = (m[1] >> 4) & 0x0F
  A[4] = m[2] & 0x0F
  A[5] = (m[2] >> 4) & 0x0F

  B[0] = m[3] & 0x07
  B[1] = (m[3] >> 3) & 0x07
  B[2] = ((m[3] >> 6) & 0x03) | ((m[4] << 2) & 0x04)
  B[3] = (m[4] >> 1) & 0x07
  B[4] = (m[4] >> 4) & 0x07
  B[5] = ((m[4] >> 7) & 0x01) | ((m[5] << 1) & 0x06)
  B[6] = (m[5] >> 2) & 0x07
  B[7] = (m[5] >> 5) & 0x07

  B[8] = m[6] & 0x07
  B[9] = (m[6] >> 3) & 0x07
  B[10] = ((m[6] >> 6) & 0x03) | ((m[7] << 2) & 0x04)
  B[11] = (m[7] >> 1) & 0x07
  B[12] = (m[7] >> 4) & 0x07
  B[13] = ((m[7] >> 7) & 0x01) | ((m[8] << 1) & 0x06)
  B[14] = (m[8] >> 2) & 0x07
  B[15] = (m[8] >> 5) & 0x07

  B[16] = m[9] & 0x07
  B[17] = (m[9] >> 3) & 0x07
  B[18] = ((m[9] >> 6) & 0x03) | ((m[10] << 2) & 0x04)
  B[19] = (m[10] >> 1) & 0x07

  C[0] = (m[10] >> 4) & 0x03
  C[1] = (m[10] >> 6) & 0x03
  C[2] = m[11] & 0x03
  C[3] = (m[11] >> 2) & 0x03
  C[4] = (m[11] >> 4) & 0x03
  C[5] = (m[11] >> 6) & 0x03
  
 else:
  assert False, "Wrong params.tau (!= 3,4)"

 D = np.array( [ A[0], A[1], A[2], B[0], A[3], B[1], B[2], B[3]
               , A[4], B[4], B[5], B[6], B[7], B[8], B[9], C[0]
               , A[5], B[10], B[11], B[12], B[13], B[14], B[15], C[1]
               , B[16], B[17], B[18], C[2], B[19], C[3], C[4], C[5]
                ])

 v = np.array([np.complex128(0j)]*16)
 for i in range(16):
  v[i] = np.complex128(D[2 * i], D[2 * i + 1])

 assert v.shape == (16,), "Typing error: wrong v shape (compute_v)! "+str(v.shape)
 return v


#
# compute_w
#

def compute_w(v: npt.NDArray[np.complex128], param_l: int = 128) -> npt.NDArray[np.uint16]:
 params = ScpParams[param_l]
 assert v.shape == (16,), "Typing error: wrong v shape (compute_w)!"+str(v.shape)
 w = np.array([np.uint16(0)]*32)
 tmp = np.array([np.complex128(0j)]*16)
 base = np.complex128(1.0, 1.0)

 for i in range(16):
  tmp[i] = v[i]

 for i in range(8):
  tmp[2*i+1] = tmp[2*i] + tmp[2*i+1]*base

 for i in range(4):
  tmp[4*i+2] = tmp[4*i] + tmp[4*i+2]*base
  tmp[4*i+3] = tmp[4*i+1] + tmp[4*i+3]*base

 for i in range(2):
  tmp[8*i+4] = tmp[8*i] + tmp[8*i+4]*base
  tmp[8*i+5] = tmp[8*i+1] + tmp[8*i+5]*base
  tmp[8*i+6] = tmp[8*i+2] + tmp[8*i+6]*base
  tmp[8*i+7] = tmp[8*i+3] + tmp[8*i+7]*base

 for i in range(8):
  tmp[8+i] = tmp[i] + tmp[8+i]*base
  
 if params.tau == 3:
  for i in range(16):
   w[2*i] = np.uint16(int(tmp[i].real)&0x7) * (1 << (params.logq-3)) & 0xFFF
   w[2*i+1] = np.uint16(int(tmp[i].imag) & 0x7) * (1 << (params.logq-3)) & 0xFFF

 elif params.tau == 4:
  for i in range(16):
   w[2*i] = np.uint16(int(tmp[i].real)&0xF) * (1 << (params.logq-4)) & 0xFFF
   w[2*i+1] = np.uint16(int(tmp[i].imag) & 0xF) * (1 << (params.logq-4)) & 0xFFF

 else:
  assert False, "Wrong params.tau"

 assert w.shape == (32,), "Typing error: wrong w size!"
 return w


#
# reduce_w
#

def reduce_w(inarray: npt.NDArray[np.complex128], param_l: int = 128) -> npt.NDArray[np.complex128]:
 params = ScpParams[param_l]
 assert inarray.shape == (16,), "Typing error: wrong inarray size!"
 inout = np.array(inarray)
 if params.tau == 3:
  inout[0] = np.float64(int(inout[0].real) & 0x7) + np.float64(int(inout[0].imag) & 0x7)*1j
  inout[3] = np.float64(int(inout[3].real) & 0x3) + np.float64(int(inout[3].imag) & 0x3)*1j
  inout[5] = np.float64(int(inout[5].real) & 0x3) + np.float64(int(inout[5].imag) & 0x3)*1j
  inout[6] = np.float64(int(inout[6].real) & 0x3) + np.float64(int(inout[6].imag) & 0x3)*1j
  inout[9] = np.float64(int(inout[9].real) & 0x3) + np.float64(int(inout[9].imag) & 0x3)*1j
  inout[10] = np.float64(int(inout[10].real) & 0x3) + np.float64(int(inout[10].imag) & 0x3)*1j
  inout[12] = np.float64(int(inout[12].real) & 0x3) + np.float64(int(inout[12].imag) & 0x3)*1j
  inout[15] = np.float64(int(inout[15].real) & 0x1) + np.float64(int(inout[15].imag) & 0x1)*1j

  mod = int(inout[1].imag) & 0x3
  sub = mod - int(inout[1].imag)
  inout[1] = np.float64(int(inout[1].real) + int(sub) & 0x7) + np.float64(mod)*1j

  mod = int(inout[2].imag) & 0x3
  sub = mod - int(inout[2].imag)
  inout[2] = np.float64(int(inout[2].real) + int(sub) & 0x7) + np.float64(mod)*1j

  mod = int(inout[4].imag) & 0x3
  sub = mod - int(inout[4].imag)
  inout[4] = np.float64(int(inout[4].real) + int(sub) & 0x7) + np.float64(mod)*1j

  mod = int(inout[8].imag) & 0x3
  sub = mod - int(inout[8].imag)
  inout[8] = np.float64(int(inout[8].real) + int(sub) & 0x7) + np.float64(mod)*1j

  mod = int(inout[7].imag) & 0x1
  sub = mod - int(inout[7].imag)
  inout[7] = np.float64(int(inout[7].real) + int(sub) & 0x3) + np.float64(mod)*1j

  mod = int(inout[11].imag) & 0x1
  sub = mod - int(inout[11].imag)
  inout[11] = np.float64(int(inout[11].real) + int(sub) & 0x3) + np.float64(mod)*1j

  mod = int(inout[13].imag) & 0x1
  sub = mod - int(inout[13].imag)
  inout[13] = np.float64(int(inout[13].real) + int(sub) & 0x3) + np.float64(mod)*1j

  mod = int(inout[14].imag) & 0x1;
  sub = mod - int(inout[14].imag);
  inout[14] = np.float64(int(inout[14].real) + int(sub) & 0x3) + float(mod)*1j

 elif params.tau == 4:
  inout[0] = np.float64(int(inout[0].real) & 0xF) + np.float64(int(inout[0].imag) & 0xF)*1j
  inout[3] = np.float64(int(inout[3].real) & 0x7) + np.float64(int(inout[3].imag) & 0x7)*1j
  inout[5] = np.float64(int(inout[5].real) & 0x7) + np.float64(int(inout[5].imag) & 0x7)*1j
  inout[6] = np.float64(int(inout[6].real) & 0x7) + np.float64(int(inout[6].imag) & 0x7)*1j
  inout[9] = np.float64(int(inout[9].real) & 0x7) + np.float64(int(inout[9].imag) & 0x7)*1j
  inout[10] = np.float64(int(inout[10].real) & 0x7) + np.float64(int(inout[10].imag) & 0x7)*1j
  inout[12] = np.float64(int(inout[12].real) & 0x7) + np.float64(int(inout[12].imag) & 0x7)*1j
  inout[15] = np.float64(int(inout[15].real) & 0x3) + np.float64(int(inout[15].imag) & 0x3)*1j

  mod = int(inout[1].imag) & 0x7
  sub = mod - int(inout[1].imag)
  inout[1] = np.float64(int(inout[1].real) + int(sub) & 0xF) + np.float64(mod)*1j

  mod = int(inout[2].imag) & 0x7
  sub = mod - int(inout[2].imag)
  inout[2] = np.float64(int(inout[2].real) + int(sub) & 0xF) + np.float64(mod)*1j

  mod = int(inout[4].imag) & 0x7
  sub = mod - int(inout[4].imag)
  inout[4] = np.float64(int(inout[4].real) + int(sub) & 0xF) + np.float64(mod)*1j

  mod = int(inout[8].imag) & 0x7
  sub = mod - int(inout[8].imag)
  inout[8] = np.float64(int(inout[8].real) + int(sub) & 0xF) + np.float64(mod)*1j

  mod = int(inout[7].imag) & 0x3
  sub = mod - int(inout[7].imag)
  inout[7] = np.float64(int(inout[7].real) + int(sub) & 0x7) + np.float64(mod)*1j

  mod = int(inout[11].imag) & 0x3
  sub = mod - int(inout[11].imag)
  inout[11] = np.float64(int(inout[11].real) + int(sub) & 0x7) + np.float64(mod)*1j

  mod = int(inout[13].imag) & 0x3
  sub = mod - int(inout[13].imag)
  inout[13] = np.float64(int(inout[13].real) + int(sub) & 0x7) + np.float64(mod)*1j

  mod = int(inout[14].imag) & 0x3
  sub = mod - int(inout[14].imag)
  inout[14] = np.float64(int(inout[14].real) + int(sub) & 0x7) + np.float64(mod)*1j

 else:
  assert False, "Wrong params.tau"

 assert inout.shape == (16,), "Typing error: wrong inout size!"
 return inout


#
# recover_m
#

def recover_m(v: npt.NDArray[np.complex128], param_l: int = 128) -> npt.NDArray[np.uint8]:
 params = ScpParams[param_l]
 assert v.shape == (16,), "Typing error: wrong v size!"
 A = np.array([ 0, 1, 2, 4, 8, 16])
 B = np.array([ 3, 5, 6, 7, 9, 10, 11, 12, 13, 14
              ,17, 18, 19, 20, 21, 22, 24, 25, 26, 28])
 C = np.array([ 15, 23, 27, 29, 30, 31])
 vecv = np.array([np.uint16(0)]*32)

 for i in range(16):
  vecv[2*i] = my_round(v[i].real)
  vecv[2*i+1] = my_round(v[i].imag)

 if params.tau == 3:
  m = np.array([np.uint8(0)]*8)
  for i in range(5,-1,-1):
   m[7] = (m[7] << 1 | vecv[C[i]])
  m[7] = (m[7] << 2) | vecv[B[19]]
  m[6] = (m[6] | vecv[B[18]]) << 2
  m[6] = (m[6] | vecv[B[17]]) << 2
  m[6] = (m[6] | vecv[B[16]]) << 2
  m[6] = (m[6] | vecv[B[15]]) << 0
  m[5] = (m[5] | vecv[B[14]]) << 2
  m[5] = (m[5] | vecv[B[13]]) << 2
  m[5] = (m[5] | vecv[B[12]]) << 2
  m[5] = (m[5] | vecv[B[11]]) << 0
  m[4] = (m[4] | vecv[B[10]]) << 2
  m[4] = (m[4] | vecv[B[9]]) << 2
  m[4] = (m[4] | vecv[B[8]]) << 2
  m[4] = (m[4] | vecv[B[7]]) << 0
  m[3] = (m[3] | vecv[B[6]]) << 2
  m[3] = (m[3] | vecv[B[5]]) << 2
  m[3] = (m[3] | vecv[B[4]]) << 2
  m[3] = (m[3] | vecv[B[3]]) << 0
  m[2] = (m[2] | vecv[B[2]]) << 2
  m[2] = (m[2] | vecv[B[1]]) << 2
  m[2] = (m[2] | vecv[B[0]]) << 2
  m[2] = m[2] | (vecv[A[5]] >> 1)
  m[1] = m[1] | (vecv[A[5]] << 7)
  m[1] = m[1] | (vecv[A[4]] << 4)
  m[1] = m[1] | (vecv[A[3]] << 1)
  m[1] = m[1] | (vecv[A[2]] >> 2)
  m[0] = m[0] | (vecv[A[2]] << 6)
  m[0] = m[0] | (vecv[A[1]] << 3)
  m[0] = m[0] | (vecv[A[0]] << 0)

 elif params.tau == 4:
  m = np.array([np.uint8(0)]*12)
  m[11] = (vecv[C[5]] << 6) | (vecv[C[4]] << 4) | (vecv[C[3]] << 2) | (vecv[C[2]])
  m[10] = (vecv[C[1]] << 6) | (vecv[C[0]] << 4) | (vecv[B[19]] << 1) | (vecv[B[18]] >> 2)
  m[9] = (vecv[B[18]] << 6) | (vecv[B[17]] << 3) | vecv[B[16]]
  m[8] = (vecv[B[15]] << 5) | (vecv[B[14]] << 2) | (vecv[B[13]] >> 1)
  m[7] = (vecv[B[13]] << 7) | (vecv[B[12]] << 4) | (vecv[B[11]] << 1) | (vecv[B[10]] >> 2)
  m[6] = (vecv[B[10]] << 6) | (vecv[B[9]] << 3) | vecv[B[8]]
  m[5] = (vecv[B[7]] << 5) | (vecv[B[6]] << 2) | (vecv[B[5]] >> 1)
  m[4] = (vecv[B[5]] << 7) | (vecv[B[4]] << 4) | (vecv[B[3]] << 1) | (vecv[B[2]] >> 2)
  m[3] = (vecv[B[2]] << 6) | (vecv[B[1]] << 3) | vecv[B[0]]
  m[2] = (vecv[A[5]] << 4) | (vecv[A[4]])
  m[1] = (vecv[A[3]] << 4) | (vecv[A[2]])
  m[0] = (vecv[A[1]] << 4) | (vecv[A[0]])

 else:
  assert False, "Wrong params.tau"

 assert m.shape == (params.mu//8,), "Typing error: wrong m size!"
 return m

#
# recover_v
#

def recover_v(w: npt.NDArray[np.complex128], param_l: int = 128) -> npt.NDArray[np.complex128]:
 params = ScpParams[param_l]
 assert w.shape == (16,), "Typing error: wrong w size!"
 v = np.array([np.complex128(0j)]*16)
 tmp = np.array([np.complex128(0j)]*16)
 base = np.complex128(0.5,-0.5)

 for i in range(16):
  tmp[i] = w[i]

 for i in range(8):
  tmp[8+i] = (tmp[8+i] - tmp[i])*base 

 for i in range(2):
  for j in range(4):
   tmp[8*i+4+j] = (tmp[8*i+4+j] - tmp[8*i+j])*base

 for i in range(4):
  for j in range(2):
   tmp[4*i+2+j] = (tmp[4*i+2+j] - tmp[4*i+j])*base

 for i in range(8):
  tmp[2*i+1] = (tmp[2*i+1] - tmp[2*i])*base

 tmp = reduce_w(tmp, param_l);

 for i in range(16):
  v[i] = tmp[i]

 assert v.shape == (16,), "Typing error: wrong v size!"
 return v


#
# bddbwn
#

def euclidean_distance(set1: npt.NDArray[np.complex128], set2: npt.NDArray[np.complex128], size:int) -> np.float64:
 """ Computes the Euclidean distance between two sets of complex numbers.
 
  This function calculates the Euclidean distance between two sets of complex
  numbers by summing the squared differences of their real and imaginary parts
  and then taking the square root of the sum.
 
  Arguments 'set1' and 'set2': arrays with both sets of complex numbers.
  Returns the Euclidean distance between the two sets of complex numbers.     """
 sum = np.float64(0)
 for i in range(size):
  real_diff = set1[i].real - set2[i].real
  imag_diff = set1[i].imag - set2[i].imag
  sum += real_diff * real_diff + imag_diff * imag_diff

 # obs: the c_ref code does not takes the "square root" of the "sum"
 return sum

def bddbwn(t: npt.NDArray[np.complex128], n: int) -> npt.NDArray[np.complex128]:
 """ Perform the Babai's nearest plane algorithm for decoding using the BDD
   Algorithm for Barnes-Wall lattices.

   This function implements the BDD Algorithm for Barnes-Wall lattices to decode
   a complex vector t into a lattice vector y. It recursively divides the input
   vector into smaller parts, computes intermediate results, and selects the
   closest lattice vector based on the Euclidean distance.
  
   BDD Algorithm for Barnes-Wall lattices:

   Input: a target vector t ∈ C^n/2 and the lattice BWn
   Output: a lattice vector y ∈ BWn
   1: if n = 2 then
   2:     return ⌊t⌉
   3: else
   4:     Write t = (t1, t2) such that t1, t2 ∈ C^n/4
   5:     Compute y1 = BDD(t1, BWn/2), y2 = BDD(t2, BWn/2)
   6:     Compute z1 = BDD(ϕ^−1(t2 − y1), BWn/2), z2 = BDD(ϕ^−1(t1 − y2), BWn/2)
   7:     Compute x = (y1, y1 + ϕz1), x′ = (y2 + ϕz2, y2)
   8:     if ||x − t|| < ||x′ − t|| then
   9:         return x
   10:    else
   11:        return x′
   12:    end if
   13: end if
  
   Argument 't': The input complex vector of length 'n'.
   Argument 'y': The output lattice vector of length 'n'.
   Argument 'n': The length of the input and output vectors.

   Recursion tree: 1*32; 2*16; 4*8; 8*4; 16*2                                         """

 tlen = n >> 1
 halftlen = tlen >> 1
 base1 = np.complex128(1,1)
 base0 = np.complex128(.5,-.5)

 y = np.array([0j]*tlen)

 if n == 2:
  y[0] = my_round(t[0].real) + my_round(t[0].imag)*1j
  return y

 t1 = np.array([np.complex128(0j)]*halftlen)
 t2 = np.array([np.complex128(0j)]*halftlen)

 for i in range(halftlen):
  t1[i] = t[i]
  t2[i] = t[i+halftlen]

 y1 = bddbwn(t1, tlen) # y1
 y2 = bddbwn(t2, tlen) # z2

 z1in = np.array([np.complex128(0j)]*halftlen)
 z2in = np.array([np.complex128(0j)]*halftlen)
 for i in range(halftlen):
  z1in[i] = (t2[i] - y1[i])*base0 # tmp
  z2in[i] = (t1[i] - y2[i])*base0 # tmp

 z1 = bddbwn(z1in, tlen) # z1
 z2 = bddbwn(z2in, tlen) # y2
 for i in range(halftlen):
  z1[i] = z1[i]*base1
  z2[i] = z2[i]*base1

 out1 = np.array([np.complex128(0j)]*tlen)
 out2 = np.array([np.complex128(0j)]*tlen)
 for i in range(halftlen):
  out1[i] = y1[i] # y1
  out1[halftlen + i] = y1[i] + z1[i]  # y2
  out2[i] = y2[i] + z2[i] # z1
  out2[halftlen + i] = y2[i] # z2

 d1 = euclidean_distance(out1, t, tlen)
 d2 = euclidean_distance(out2, t, tlen)

 for i in range(tlen):
  y[i] = out1[i] if d1 < d2 else out2[i]

 return y


#
# msgencode
#

def scloudplus_msgencode(msg: npt.NDArray[np.uint8], param_l: int = 128) -> npt.NDArray[np.uint16]:
 params = ScpParams[param_l]
 assert msg.shape == (params.subm*params.mu//8,), "Typing error: wrong msg size!"
 matrixM = np.zeros((params.mbar*params.nbar,),np.uint16)
 v = np.array([np.complex128(0j)]*16)
 mu8 = params.mu >> 3
 for i in range(params.subm):
  v = compute_v(msg[i*mu8:i*mu8+mu8],param_l)
  rows = compute_w(v,param_l)
  matrixM[i*32:i*32+32] = rows
 matrixM = matrixM.reshape((params.mbar,params.nbar))
 assert matrixM.shape == (params.mbar,params.nbar), "Typing error: wrong matM size!"
 return matrixM


#
# msgdecode
#

def scloudplus_msgdecode(matrixM: npt.NDArray[np.uint16], param_l: int = 128) -> npt.NDArray[np.uint8]:
 params = ScpParams[param_l]
 assert matrixM.shape == (params.mbar,params.nbar), "Typing error: wrong matM size!"
 matrixM = matrixM.reshape((params.mbar*params.nbar,))
 v = np.array([np.complex128(0j)]*16)
 mu8 = params.mu >> 3
 msg = np.array([np.uint8(0)]*params.subm*mu8)
 for i in range(params.subm):
  for j in range(16):
   v[j] = np.float64(matrixM[32*i+2*j]) * (1 << params.tau) * 0.000244140625 \
          + np.float64(matrixM[32*i+2*j+1]) * (1 << params.tau) * 0.000244140625 * 1j
  w = bddbwn(v, 32)
  v = recover_v(w,param_l)
  m = recover_m(v,param_l)
  mlen = 8 if params.tau == 3 else 12
  msg[i*mlen:i*mlen+mlen] = m[0:mlen]
 assert msg.shape == (params.subm*params.mu//8,), "Typing error: wrong msg size!"
 return msg


#
# packpk
#


def scloudplus_packpk(B: npt.NDArray[np.uint16], param_l: int = 128):
 params = ScpParams[param_l]
 assert B.shape == (params.m,params.nbar), "Typing error: wrong B shape!"
 l = []
 for i in range(0,params.m*params.nbar,2):
  tmp = np.uint32(B.flat[i]&0xFFF)
  tmp += np.uint32((B.flat[i+1]&0xFFF))<<12
  l = l + list(tmp.tobytes()[0:3])
 pk = np.array(l,dtype=np.uint8)
 assert pk.shape==(params.pk-16,), "Type error: wrong pk(B) shape (packpk)! "+str(pk.shape)
 return pk

#
# unpackpk
#

def scloudplus_unpackpk(pk: npt.NDArray[np.uint8], param_l: int = 128) -> npt.NDArray[np.uint16]:
 params = ScpParams[param_l]
 assert pk.shape==(params.pk-16,), "Type error: wrong pk(B) size in unpackpk"
 B = np.zeros((params.m,params.nbar), dtype=np.uint16)
 for i in range(0,params.m*params.nbar,2):
   tmp = np.uint32(int.from_bytes(pk[3*i//2:3*i//2+3],'little'))
   B.flat[i] = np.uint16(tmp & 0xFFF)
   tmp >>= 12
   B.flat[i+1] = np.uint16(tmp & 0xFFF)
 assert B.shape == (params.m,params.nbar), "Typing error: wrong B shape (unpackpk)! "+str(B.shape)
 return B


#
# packsk
#

def scloudplus_packsk(S: npt.NDArray[np.uint16], param_l: int = 128) -> npt.NDArray[np.uint8]:
 params = ScpParams[param_l]
 assert S.shape==(params.nbar,params.n), "typing error: wrong shape on S (packsk)! "+str(S.shape)
 l = []
 for i in range(0,params.n*params.nbar,4):
  temp = S.flat[i] & 0x03
  temp ^= ((S.flat[i+1] << 2) & 0x0C)
  temp ^= ((S.flat[i+2] << 4) & 0x30)
  temp ^= ((S.flat[i+3] << 6) & 0xC0)
  l.append(np.uint8(temp))
 sk = np.array(l, dtype=np.uint8)
 assert sk.shape==(params.pke_sk,), "typing error: wrong shape on sk (packsk)"+str(sk.shape)
 return sk


#
# unpacksk
#

def scloudplus_unpacksk(sk: npt.NDArray[np.uint8], param_l: int = 128) -> npt.NDArray[np.uint16]:
 params = ScpParams[param_l]
 assert sk.shape==(params.pke_sk,), "type error: wrong shape on sk (unpacksk)"+str(sk.shape)
 S = np.zeros((params.nbar,params.n),dtype=np.uint16)
 for i in range(0,params.n*params.nbar,4):
  temp = sk[i//4]
  S.flat[i] = np.int16(np.uint16(temp & 0x03)<<14)>>14
  S.flat[i+1] = np.int16(np.uint16((temp>>2) & 0x03)<<14)>>14
  S.flat[i+2] = np.int16(np.uint16((temp>>4) & 0x03)<<14)>>14
  S.flat[i+3] = np.int16(np.uint16((temp>>6) & 0x03)<<14)>>14
 assert S.shape==(params.nbar,params.n), "type error: wrong shape on S (unpacksk)"+str(S.shape)
 return S

#
# compressc1
#

def scloudplus_compressc1(C1: npt.NDArray[np.uint16], param_l: int = 128) -> npt.NDArray[np.uint16]:
 params = ScpParams[param_l]
 assert C1.shape==(params.mbar,params.n), "type error: wrong shape on C1 (compressc1) "+str(C1.shape)
 C1barl = []
 for i in range(params.mbar):
  l = []
  for j in range(params.n):
   if params.l == 128:
    l.append(np.uint16(((np.uint32(C1[i][j] & 0xFFF) << 9) + 2048) >> 12) & 0x1FF)
   elif params.l == 192:
    l.append(np.uint16(C1[i][j]))
   elif params.l == 256:
    l.append(np.uint16(((np.uint32(C1[i][j] & 0xFFF) << 10) + 2048) >> 12) & 0x3FF)
   else:
    assert False, "Wrong params.l"
  C1barl.append(l)
 C1bar = np.array(C1barl)
 assert C1bar.shape==(params.mbar,params.n), "type error: wrong shape on C1bar (compressc1) "+str(C1bar.shape)
 return C1bar

#
# decompressc1
#

def scloudplus_decompressc1(C1bar: npt.NDArray[np.uint16], param_l: int = 128) -> npt.NDArray[np.uint16]:
 params = ScpParams[param_l]
 assert C1bar.shape==(params.mbar,params.n), "type error: wrong shape on C1bar (decompressc1) "+str(C1bar.shape)
 C1l = []
 for i in range(params.mbar):
  l = []
  for j in range(params.n):
   if params.l == 128:
    l.append(np.uint16(((np.uint32(C1bar[i][j] & 0x1FF) << 12) + 256) >> 9))
   elif params.l == 192:
    l.append(C1bar[i][j])
   elif params.l == 256:
    l.append(np.uint16(((np.uint32(C1bar[i][j] & 0x3FF) << 12) + 256) >> 10))
   else:
    assert False, "Wrong params.l"
  C1l.append(l)
 C1 = np.array(C1l, dtype=np.uint16)
 assert C1.shape==(params.mbar,params.n), "type error: wrong shape on C1 (decompressc1) "+str(C1.shape)
 return C1

#
# compressc2
#

def scloudplus_compressc2(C2: npt.NDArray[np.uint16], param_l: int = 128) -> npt.NDArray[np.uint16]:
 params = ScpParams[param_l]
 assert C2.shape==(params.mbar,params.nbar), "type error: wrong shape on C2 (compressc2) "+str(C2.shape)
 C2barl = []
 if params.l == 128 or params.l == 256:
  for i in range(params.mbar):
   l = []
   for j in range(params.nbar):
    temp = np.uint16(((np.uint32(C2[i][j] & 0xFFF) << 7) + 2048) >> 12)
    remainder = (C2[i][j]%64 == 48)
    l.append(np.uint16((temp-remainder) & 0x7F))
   C2barl.append(l)
 elif params.l == 192:
  for i in range(params.mbar):
   l = []
   for j in range(params.nbar):
    temp = np.uint16(((np.uint32(C2[i][j] & 0xFFF) << 10) + 2048) >> 12)
    remainder = (C2[i][j]%8 == 6)
    l.append(np.uint16((temp-remainder) & 0x3FF))
   C2barl.append(l)
 else:
  assert False, "Wrong params.l"
 C2bar = np.array(C2barl)
 assert C2bar.shape==(params.mbar,params.nbar), "type error: wrong shape on C2bar (compressc2) "+str(C2bar.shape)
 return C2bar

#
# decompressc2
#

def scloudplus_decompressc2(C2bar: npt.NDArray[np.uint16], param_l: int = 128) -> npt.NDArray[np.uint16]:
 params = ScpParams[param_l]
 assert C2bar.shape==(params.mbar,params.nbar), "type error: wrong shape on C2bar (decompressc2) "+str(C2bar.shape)
 C2l = []
 if params.l == 128 or params.l == 256:
  for i in range(params.mbar):
   l = []
   for j in range(params.nbar):
    l.append(np.uint16(((np.uint32(C2bar[i][j] & 0x7F) << 12)+64)>>7))
   C2l.append(l)
 elif params.l == 192:
  for i in range(params.mbar):
   l = []
   for j in range(params.nbar):
    l.append(np.uint16(((np.uint32(C2bar[i][j] & 0x3FF) << 12) + 512) >> 10))
   C2l.append(l)
 else:
  assert False, "Wrong params.l"
 C2 = np.array(C2l)
 assert C2.shape==(params.mbar,params.nbar), "type error: wrong shape on C2 (decompressc2) "+str(C2.shape)
 return C2

#
# packc1
#

# C1bar: mbar * n * q1bits
#  128 bits: 8*600*9 = 43200 (5400 bytes; 675 u64)
#  192 bits: 8*896*12 = 86016 (10752 bytes; 1344 u64)
#. 256 bits: 12*1120*10 = 134400 (16800 bytes; 2100 u64)

def u16bits_u8s(nbits: int, x: np.uint16, carry: np.uint8, cbits: np.uint8) -> tuple[np.uint8, np.uint8, list[np.uint8]]:
 l = []
 xx = np.uint32(x) & ((1 << nbits) - 1)
 xx = xx << cbits
 xx = xx | carry
 size = nbits + cbits
 while 8 <= size:
    l.append(np.uint8(xx & 0xFF))
    xx = xx >> 8
    size -= 8
 return np.uint8(xx), np.uint8(size), l

def scloudplus_packc1_paper(C1bar: npt.NDArray[np.uint16], param_l: int = 128) -> npt.NDArray[np.uint8]:
 params = ScpParams[param_l]
 assert C1bar.shape==(params.mbar,params.n), "type error: wrong shape on C1bar (packc1) "+str(C1bar.shape)
 l = []
 carry = np.uint8(0)
 cbits = np.uint8(0)
 for i in range(params.mbar):
  for j in range(params.n):
   carry, cbits, temp = u16bits_u8s(params.logq1, C1bar[i][j], carry, cbits)
   l = l + temp
 assert carry == np.uint8(0) and cbits == np.uint8(0), "number os bits of C1bar should always be multiple of 8!!! " + str(carry)
 c1 = np.array(l)
 assert c1.shape==(params.c1,), "type error: wrong shape on c1 (packc1) "+str(c1.shape)
 return c1

def fix_l128(x: np.uint8):
    x = ((x & 0x0F) << 4) | ((x & 0xF0) >> 4)
    x = ((x & 0x33) << 2) | ((x & 0xCC) >> 2)
    x = ((x & 0x55) << 1) | ((x & 0xAA) >> 1)
    return x

def fix_l256(x: np.uint8):
    x = ((x & 0x0F) << 4) | ((x & 0xF0) >> 4)
    x = ((x & 0x33) << 2) | ((x & 0xCC) >> 2)
    return np.uint8(x)

def scloudplus_packc1(C1bar: npt.NDArray[np.uint16], param_l: int = 128) -> npt.NDArray[np.uint8]:
 params = ScpParams[param_l]
 assert C1bar.shape==(params.mbar,params.n), "type error: wrong shape on C1bar (packc1) "+str(C1bar.shape)
 l = []
 if params.l==128 or params.l==256:
  # leftovers are stored after full bytes...
  # NOTE: it assumes that C1bar is normalized!!!
  # NOTE2: the implementation reverses the order of leftovers!!!
  fix = fix_l128 if params.l==128 else fix_l256
  for i in range(params.mbar*params.n):
   l.append(np.uint8(C1bar.flat[i]))
  carry = np.uint8(0)
  cbits = np.uint8(0)
  for i in range(params.mbar*params.n):
   carry, cbits, temp = u16bits_u8s(params.logq1-8, C1bar.flat[i]>>8, carry, cbits)
   l = l + [ fix(x) for x in temp ]
 elif params.l==192:
  # standard pack strategy...
  carry = np.uint8(0)
  cbits = np.uint8(0)
  for i in range(params.mbar):
   for j in range(params.n):
    carry, cbits, temp = u16bits_u8s(params.logq1, C1bar[i][j], carry, cbits)
    l = l + temp
 assert carry == np.uint8(0) and cbits == np.uint8(0), "number os bits of C1bar should always be multiple of 8!!! " + str(carry)
 c1 = np.array(l)
 assert c1.shape==(params.c1,), "type error: wrong shape on c1 (packc1) "+str(c1.shape)
 return c1

#
# unpackc1
#

def u8s_u16(nbits: int, l: list[np.uint8], carry: np.uint64, cbits: np.uint8) -> tuple[list[np.uint8], np.uint64, np.uint8, np.uint16]:
 i = 0
 while i<len(l) and cbits<nbits:
  temp = np.uint64(l[i])
  i += 1
  temp <<= cbits
  cbits += 8
  carry |= temp
 r = np.uint16(carry & (2**nbits-1))
 if cbits>=nbits:
  carry >>= nbits
  cbits -= nbits
# assert i<len(l) or cbits==0, "leftovers on carry:"+str(cbits)
 l = l[i:]
 return l, carry, cbits, r

def scloudplus_unpackc1_paper(c1: npt.NDArray[np.uint8], param_l: int = 128):
 params = ScpParams[param_l]
 assert c1.shape==(params.c1,), "type error: wrong shape on c1 (unpackc1) "+str(c1.shape)
 C1barl = []
 c1l = list(c1)
 carry = np.uint64(0)
 cbits = np.uint8(0)
 for _ in range(params.mbar):
  l = []
  for _ in range(params.n):
   c1l, carry, cbits, temp = u8s_u16(params.logq1, c1l, carry, cbits)
   l.append(temp)
  C1barl.append(l)
 C1bar = np.array(C1barl)
 assert C1bar.shape==(params.mbar,params.n), "type error: wrong shape on C1bar (unpackc1) "+str(C1bar.shape)
 return C1bar

def scloudplus_unpackc1(c1: npt.NDArray[np.uint8], param_l: int = 128) -> npt.NDArray[np.uint16]:
 params = ScpParams[param_l]
 assert c1.shape==(params.c1,), "type error: wrong shape on c1 (unpackc1) "+str(c1.shape)
 carry = np.uint64(0)
 cbits = np.uint8(0)
 if params.l==128 or params.l==256:
  # leftovers are stored after full bytes...
  # NOTE: the implementation reverses the order of leftovers!!!
  fix = fix_l128 if params.l==128 else fix_l256
  C1bar = np.zeros((params.mbar,params.n),np.uint16)
  for i in range(params.mbar*params.n):
   C1bar.flat[i] = np.uint16(c1[i])
  c1l = [ fix(x) for x in list(c1)[params.mbar*params.n:] ]
  for i in range(params.mbar*params.n):
   c1l, carry, cbits, temp = u8s_u16(params.logq1-8, c1l, carry, cbits)
   C1bar.flat[i] = C1bar.flat[i] | np.uint16(temp<<8)
 elif params.l==192:
  # standard unpack strategy...
  C1barl = []
  c1l = list(c1)
  for _ in range(params.mbar):
   l = []
   for _ in range(params.n):
    c1l, carry, cbits, temp = u8s_u16(params.logq1, c1l, carry, cbits)
    l.append(temp)
   C1barl.append(l)
  C1bar = np.array(C1barl)
 assert C1bar.shape==(params.mbar,params.n), "type error: wrong shape on C1bar (unpackc1) "+str(C1bar.shape)
 return C1bar

#
# packc2
#

# C2bar: mbar * nbar * q1bits
#  128 bits: 8*8*7 = 448 (56 bytes; 7 u64)
#  192 bits: 8*8*10 = 640 (80 bytes; 10 u64)
#  256 bits: 12*11*7 = 924+4pad (116 bytes; 14.5 u64)

def scloudplus_packc2_paper(C2bar: npt.NDArray[np.uint16], param_l: int = 128) -> npt.NDArray[np.uint8]:
 params = ScpParams[param_l]
 assert C2bar.shape==(params.mbar,params.nbar), "type error: wrong shape on C2bar (packc2) "+str(C2bar.shape)
 l = []
 carry = np.uint8(0)
 cbits = np.uint8(0)
 for i in range(params.mbar):
  for j in range(params.nbar):
#   if i==params.mbar-1 and j>=params.nbar-4:
#    print("C2[%d][%d]=%d, carry=%d, cbits=%d" % (i, j, C2bar[i][j], carry, cbits))
   carry, cbits, temp = u16bits_u8s(params.logq2, C2bar[i][j], carry, cbits)
#   if i==params.mbar-1 and j>=params.nbar-4:
#    print("temp=", temp, ", carry=%d, cbits=%d" % (carry, cbits))
   l = l + temp
 if cbits!=0: l.append(np.uint8(carry))
# print("last4=", list(l[-4:]))
 c2 = np.array(l)
 assert c2.shape==(params.c2,), "type error: wrong shape on c2 (packc2) "+str(c2.shape)
 return c2

def scloudplus_packc2(C2bar: npt.NDArray[np.uint16], param_l: int = 128) -> npt.NDArray[np.uint8]:
 params = ScpParams[param_l]
 assert C2bar.shape==(params.mbar,params.nbar), "type error: wrong shape on C2bar (packc2) "+str(C2bar.shape)
 l = []
 carry = np.uint8(0)
 cbits = np.uint8(0)
 if params.l==128 or params.l==256:
  # standard pack strategy...
  for i in range(params.mbar):
   for j in range(params.nbar):
    carry, cbits, temp = u16bits_u8s(params.logq2, C2bar[i][j], carry, cbits)
    l = l + temp
  if cbits!=0: l.append(np.uint8(carry))
 elif params.l==192:
  # leftovers are stored after full bytes...
  # NOTE: it assumes that C2bar is normalized!!!
  # NOTE2: the implementation reverses the order of leftovers!!!
  fix = fix_l256
  for i in range(params.mbar*params.nbar):
   l.append(np.uint8(C2bar.flat[i]))
  for i in range(params.mbar*params.nbar):
   carry, cbits, temp = u16bits_u8s(params.logq2-8, C2bar.flat[i]>>8, carry, cbits)
   l = l + [ fix(x) for x in temp ]
 c2 = np.array(l)
 assert c2.shape==(params.c2,), "type error: wrong shape on c2 (packc2) "+str(c2.shape)
 return c2

#
# unpackc2
#

def scloudplus_unpackc2_paper(c2: npt.NDArray[np.uint8], param_l: int = 128):
 params = ScpParams[param_l]
 assert c2.shape==(params.c2,), "type error: wrong shape on c2 (unpackc2) "+str(c2.shape)
 c2l = list(c2)
 C2barl = []
 carry = np.uint64(0)
 cbits = np.uint8(0)
 for i in range(params.mbar):
  l = []
  for j in range(params.nbar):
   c2l, carry, cbits, temp = u8s_u16(params.logq2, c2l, carry, cbits)
   l.append(temp)
  C2barl.append(l)
 C2bar = np.array(C2barl)
 assert C2bar.shape==(params.mbar,params.nbar), "type error: wrong shape on C2bar (unpackc2) "+str(C2bar.shape)
 return C2bar

def scloudplus_unpackc2(c2: npt.NDArray[np.uint8], param_l: int = 128) -> npt.NDArray[np.uint16]:
 params = ScpParams[param_l]
 assert c2.shape==(params.c2,), "type error: wrong shape on c2 (unpackc2) "+str(c2.shape)
 carry = np.uint64(0)
 cbits = np.uint8(0)
 if params.l==128 or params.l==256:
  # standard unpack strategy...
  C2barl = []
  c2l = list(c2)
  for i in range(params.mbar):
   l = []
   for j in range(params.nbar):
    c2l, carry, cbits, temp = u8s_u16(params.logq2, c2l, carry, cbits)
    l.append(temp)
   C2barl.append(l)
  C2bar = np.array(C2barl)
 elif params.l==192:
  # leftovers are stored after full bytes...
  # NOTE: the implementation reverses the order of leftovers!!!
  fix = fix_l256
  C2bar = np.zeros((params.mbar,params.nbar),np.uint16)
  for i in range(params.mbar*params.nbar):
   C2bar.flat[i] = np.uint16(c2[i])
  c2l = [ fix(x) for x in list(c2)[params.mbar*params.nbar:] ]
  for i in range(params.mbar*params.nbar):
   c2l, carry, cbits, temp = u8s_u16(params.logq2-8, c2l, carry, cbits)
   C2bar.flat[i] = C2bar.flat[i] | np.uint16(temp<<8)
 assert C2bar.shape==(params.mbar,params.nbar), "type error: wrong shape on C2bar (unpackc2) "+str(C2bar.shape)
 return C2bar

