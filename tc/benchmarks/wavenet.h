/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "tc/aten/aten.h"
#include "tc/core/cuda/cuda_mapping_options.h"

namespace tc {
constexpr static auto TC_WAVENET1_NAME = "wavenet1";
constexpr static auto TC_WAVENET2_NAME = "wavenet1";

constexpr static auto TC_WAVENET = R"TC(
# Original data is float(B, C, RECEPTIVE_FIELD) and undergoes a \
# Conv1d to become float(B, RESIDUAL_C, RECEPTIVE_FIELD)

def wavenet1(
    float(B, RESIDUAL_C, RECEPTIVE_FIELD) Data,
    float(DILATION_C, RESIDUAL_C, 2) FilterWeight,
    float(DILATION_C, RESIDUAL_C, 2) GateWeight,
    float(DILATION_C) Bias,
    float(RESIDUAL_C, DILATION_C) ResWeight,
    float(RESIDUAL_C) ResBias,
    float(SKIP_C, DILATION_C) SkipWeight,
    float(SKIP_C) SkipBias,
    float(DILATION_C, RESIDUAL_C, 2) FilterWeight2,
    float(DILATION_C, RESIDUAL_C, 2) GateWeight2,
    float(DILATION_C) Bias2,
    float(RESIDUAL_C, DILATION_C) ResWeight2,
    float(RESIDUAL_C) ResBias2,
    float(SKIP_C, DILATION_C) SkipWeight2,
    float(SKIP_C) SkipBias2,
    float(DILATION_FACTOR) Dilation)
    -> (FilterOut, GateOut, NonLin, Res, Skip)
{
    FilterOut(b, dil, rf)   = Bias(dil)
        where b in 0:B, dil in 0:DILATION_C, rf in 0:RECEPTIVE_FIELD
    FilterOut(b, dil, rf)  += Data(b, r_res, rf) * FilterWeight(dil, r_res, 0) +
        (
          (rf + DILATION_FACTOR < RECEPTIVE_FIELD) ?
            Data(b, r_res, rf + DILATION_FACTOR) * FilterWeight(dil, r_res, 1) :
            float(0)
        )
        where rf in 0:RECEPTIVE_FIELD

    GateOut(b, dil, rf)   = Bias(dil)
        where b in 0:B, dil in 0:DILATION_C, rf in 0:RECEPTIVE_FIELD
    GateOut(b, dil, rf)  += Data(b, r_res, rf) * GateWeight(dil, r_res, 0) +
        (
          (rf + DILATION_FACTOR < RECEPTIVE_FIELD) ?
            Data(b, r_res, rf + DILATION_FACTOR) * GateWeight(dil, r_res, 1) :
            float(0)
        )
        where rf in 0:RECEPTIVE_FIELD

    NonLin(b, dil, rf)   =         tanh(FilterOut(b, dil, rf))
        where rf in 0:RECEPTIVE_FIELD
    NonLin(b, dil, rf)  *= 1 / (1 + exp( -GateOut(b, dil, rf)))
        where rf in 0:RECEPTIVE_FIELD

       Res(b, res, rf)   =   Data(b,  res, rf) + ResBias(res)
       Res(b, res, rf)  += NonLin(b, r_in, rf) * ResWeight(res, r_in)

      Skip(b, skip, rf) +=! NonLin(b, r_dil, rf) * SkipWeight(skip, r_dil)
        where rf in 0:RECEPTIVE_FIELD
      Skip(b, skip, rf)  = Skip(b, skip, rf) + SkipBias(skip)
        where rf in 0:RECEPTIVE_FIELD
}

def wavenet2(
    float(B, RESIDUAL_C, RECEPTIVE_FIELD) Data,
    float(DILATION_C, RESIDUAL_C, 2) FilterWeight,
    float(DILATION_C, RESIDUAL_C, 2) GateWeight,
    float(DILATION_C) Bias,
    float(RESIDUAL_C, DILATION_C) ResWeight,
    float(RESIDUAL_C) ResBias,
    float(SKIP_C, DILATION_C) SkipWeight,
    float(SKIP_C) SkipBias,
    float(DILATION_FACTOR) Dilation)
    -> (FilterOut, GateOut, NonLin, Res, Skip, FilterOut2, GateOut2, NonLin2, Res2, Skip2)
{
    FilterOut(b, dil, rf)   = Bias(dil)
        where b in 0:B, dil in 0:DILATION_C, rf in 0:RECEPTIVE_FIELD
    FilterOut(b, dil, rf)  += Data(b, r_res, rf) * FilterWeight(dil, r_res, 0) +
        (
          (rf + DILATION_FACTOR < RECEPTIVE_FIELD) ?
            Data(b, r_res, rf + DILATION_FACTOR) * FilterWeight(dil, r_res, 1) :
            float(0)
        )
        where rf in 0:RECEPTIVE_FIELD

    GateOut(b, dil, rf)   = Bias(dil)
        where b in 0:B, dil in 0:DILATION_C, rf in 0:RECEPTIVE_FIELD
    GateOut(b, dil, rf)  += Data(b, r_res, rf) * GateWeight(dil, r_res, 0) +
        (
          (rf + DILATION_FACTOR < RECEPTIVE_FIELD) ?
            Data(b, r_res, rf + DILATION_FACTOR) * GateWeight(dil, r_res, 1) :
            float(0)
        )
        where rf in 0:RECEPTIVE_FIELD

    NonLin(b, dil, rf)   =         tanh(FilterOut(b, dil, rf))
        where rf in 0:RECEPTIVE_FIELD
    NonLin(b, dil, rf)  *= 1 / (1 + exp( -GateOut(b, dil, rf)))
        where rf in 0:RECEPTIVE_FIELD

       Res(b, res, rf)   =   Data(b,  res, rf) + ResBias(res)
       Res(b, res, rf)  += NonLin(b, r_in, rf) * ResWeight(res, r_in)

      Skip(b, skip, rf) +=! NonLin(b, r_dil, rf) * SkipWeight(skip, r_dil)
        where rf in 0:RECEPTIVE_FIELD
      Skip(b, skip, rf)  = Skip(b, skip, rf) + SkipBias(skip)
        where rf in 0:RECEPTIVE_FIELD



    FilterOut2(b, dil, rf)   = Bias(dil)
        where b in 0:B, dil in 0:DILATION_C, rf in 0:RECEPTIVE_FIELD
    FilterOut2(b, dil, rf)  += Res(b, r_res, rf) * FilterWeight(dil, r_res, 0) +
        (
          (rf + 2 * DILATION_FACTOR < RECEPTIVE_FIELD) ?
            Res(b, r_res, rf + 2 * DILATION_FACTOR) * FilterWeight(dil, r_res, 1) :
            float(0)
        )
        where rf in 0:RECEPTIVE_FIELD

    GateOut2(b, dil, rf)   = Bias(dil)
        where b in 0:B, dil in 0:DILATION_C, rf in 0:RECEPTIVE_FIELD
    GateOut2(b, dil, rf)  += Res(b, r_res, rf) * GateWeight(dil, r_res, 0) +
        (
          (rf + 2 * DILATION_FACTOR < RECEPTIVE_FIELD) ?
            Res(b, r_res, rf + 2 * DILATION_FACTOR) * GateWeight(dil, r_res, 1) :
            float(0)
        )
        where rf in 0:RECEPTIVE_FIELD

    NonLin2(b, dil, rf)   =         tanh(FilterOut2(b, dil, rf))
        where rf in 0:RECEPTIVE_FIELD
    NonLin2(b, dil, rf)  *= 1 / (1 + exp( -GateOut2(b, dil, rf)))
        where rf in 0:RECEPTIVE_FIELD

       Res2(b, res, rf)   =   Res(b,  res, rf) + ResBias(res)
       Res2(b, res, rf)  += NonLin2(b, r_in, rf) * ResWeight(res, r_in)

      Skip2(b, skip, rf) +=! NonLin2(b, r_dil, rf) * SkipWeight(skip, r_dil)
        where rf in 0:RECEPTIVE_FIELD
      Skip2(b, skip, rf)  = Skip2(b, skip, rf) + SkipBias(skip)
        where rf in 0:RECEPTIVE_FIELD
}
  )TC";

auto options_WaveNet1_P100_B_1_RES_32_DIL_32_SKIP_256_REC_4000_F_1 =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(tc::FusionStrategy::Min)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .fixParametersBeforeScheduling(false)
        .tile(16, 32)
        .unroll(1)
        .tileImperfectlyNested(false)
        .matchLibraryCalls(true)
        .mapToThreads(4, 125)
        .mapToBlocks(2, 2000, 1)
        .useSharedMemory(true)
        .usePrivateMemory(true)
        .unrollCopyShared(false)
        .useReadOnlyCache(true);

auto options_WaveNet2_P100_B_1_RES_32_DIL_32_SKIP_256_REC_4000_F_1 =
    tc::CudaMappingOptions::makeNaiveMappingOptions()
        .outerScheduleFusionStrategy(tc::FusionStrategy::Max)
        .outerScheduleAllowSkewing(false)
        .outerSchedulePositiveOrthant(true)
        .intraTileScheduleFusionStrategy(
            tc::FusionStrategy::Preserve3Coincident)
        .intraTileScheduleAllowSkewing(false)
        .intraTileSchedulePositiveOrthant(true)
        .fixParametersBeforeScheduling(false)
        .tile(64, 4, 2, 63)
        .unroll(32)
        .tileImperfectlyNested(false)
        .matchLibraryCalls(false)
        .mapToThreads(4, 32)
        .mapToBlocks(32, 500, 128)
        .useSharedMemory(false)
        .usePrivateMemory(true)
        .unrollCopyShared(true)
        .useReadOnlyCache(true);
} // namespace tc
