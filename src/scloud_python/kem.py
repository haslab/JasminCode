import numpy as np
import numpy.typing as npt

from cryptography.hazmat.primitives import hashes

from scloudplus_params import *
from pke import *

#
# H
#

def H(pk: npt.NDArray[np.uint8], param_l: int = 128) -> npt.NDArray[np.uint8]:
  params = ScpParams[param_l]
  assert pk.shape==(params.pk,), "typing error: wrong pk size"
  sha_ctx = hashes.Hash(hashes.SHA3_256)
  sha_ctx.update(pk.tobytes())
  buf = sha_ctx.finalize()
  hpk = np.array(list(buf), dtype=np.uint8)
  assert hpk.shape==(32,), "typing error: wrong hpk size"
  return hpk

def G(m: npt.NDArray[np.uint8], hpk: npt.NDArray[np.uint8], param_l: int = 128) -> npt.NDArray[np.uint8]:
  params = ScpParams[param_l]
  assert m.shape==(params.ss,), "typing error: wrong m size"
  assert hpk.shape==(32,), "typing error: wrong hpk size"
  sha_ctx = hashes.Hash(hashes.SHA3_512)
  sha_ctx.update(m.tobytes())
  sha_ctx.update(hpk.tobytes())
  buf = sha_ctx.finalize()
  rk = np.array(list(buf), dtype=np.uint8)
  assert rk.shape==(64,), "typing error: wrong rk size"
  return rk

def K(r: npt.NDArray[np.uint8], ctxt: npt.NDArray[np.uint8], param_l: int = 128) -> npt.NDArray[np.uint8]:
  params = ScpParams[param_l]
  assert r.shape==(32,), "typing error: wrong r size"
  assert ctxt.shape==(params.ctx,), "typing error: wrong ctxt size"
  shake_ctx = hashes.Hash(hashes.SHAKE256(params.ss))
  shake_ctx.update(r.tobytes())
  shake_ctx.update(ctxt.tobytes())
  buf = shake_ctx.finalize()
  ss = np.array(list(buf), dtype=np.uint8)
  assert ss.shape==(params.ss,), "typing error: wrong ss size"
  return ss

