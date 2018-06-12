#
# Issue submitted by @mdouze related to uint8 type support
#
import tensor_comprehensions as tc
from tensor_comprehensions.tc import set_logtostderr
from tensor_comprehensions.tc import set_debug_tc_mapper
from tensor_comprehensions.tc import set_debug_cuda

import numpy as np
import torch

debug = True
set_logtostderr(debug)
set_debug_tc_mapper(debug)
set_debug_cuda(debug)

N = 1000 # 10 ** 7
M = 32

codes = np.random.randint(1<<32, size=(N, M // 4)).astype('uint32')
codes = codes.view('uint8')
luts = np.random.randn(M, 256).astype('float32')

codes_t = torch.from_numpy(codes).cuda()
luts_t = torch.from_numpy(luts).cuda()

lang = """
def mindis(float(M, 256) L, uint8(N, M) codes) -> (s, v, min_idx) {
    s(n) +=! L(r_m, int32(codes(n, r_m)))
    v min=! s(r_n)
    min_idx min=! (s(r_n) == v) ? r_n : N
}
def mindis_1(float(M, 256) L, uint8(N, M) codes) -> (s) {
    s(n) +=! L(r_m, int32(codes(n, r_m)))
}
def mindis_2(float(N) s) -> (v) {
    v min=! s(r_n)
}
def mindis_3(float(N) s, float v) -> (min_idx) {
    min_idx min=! (s(r_n) == v) ? r_n : N
}
"""

# mindis as a single kernel will require grid synchronization to run efficiently
mindis = tc.define(lang, name="mindis")
outputs = mindis(luts_t, codes_t)
print("minval: {} minidx: {}".format(outputs[1], outputs[2]))

# Even when splitting in 3 kernels, global device reduction will be needed to
# run efficiently
mindis_1 = tc.define(lang, name="mindis_1")
mindis_2 = tc.define(lang, name="mindis_2")
mindis_3 = tc.define(lang, name="mindis_3")
outputs_1 = mindis_1(luts_t, codes_t)
outputs_2 = mindis_2(outputs_1)
outputs_3 = mindis_3(outputs_1, outputs_2)
print("minval: {} minidx: {}".format(outputs_2, outputs_3))

# Each reduction is probably easier to optimize with a 2-staged TC where we
# artifically increase parallelism and finish the reduction in a second kernel.
# Properly choosing K such that N = K * (N / K) should result in a good version
# with 5 kernels total.
# Mapping left for later
staged_lang = """
def mindis_1(float(M, 256) L, uint8(N, M) codes) -> (s) {
    s(n) +=! L(r_m, int32(codes(n, r_m)))
}
def mindis_2_1(float(K, NBYK) s) -> (v) {
    v(k) min=! s(k, r_nbyk)
}
def mindis_2_2(float(K) V) -> (v) {
    v min=! V(r_k)
}
def mindis_3_1(float(K, NBYK) s, float v) -> (min_idx) {
    min_idx(k) min=! (s(k, r_nbyk) == v) ? r_nbyk : k * NBYK + r_nbyk
}
def mindis_3_2(float(K) tmp_min_idx) -> (min_idx) {
    min_idx min=! (tmp_min_idx(r_k) < N) ? r_k : N
}
"""
