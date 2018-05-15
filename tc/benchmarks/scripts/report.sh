#! /bin/bash

BACKEND=${BACKEND:=P100}
TC_DIR=$(git rev-parse --show-toplevel)

# BENCHMARKS=$(find ${TC_DIR}/build/tc/benchmarks/benchmark*)
BENCHMARKS="batchmatmul group_convolution tmm kronecker"

for b in ${BENCHMARKS}; do
    ${TC_DIR}/build/tc/benchmarks/benchmark_$b --gtest_filter="*${BACKEND}*";
done
