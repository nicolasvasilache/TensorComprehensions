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
#include <iostream>
#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <ATen/ATen.h>

#include "tc/aten/aten_compiler.h"
#include "tc/autotuner/genetic_autotuner_aten.h"
#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/flags.h"

#include "../test/test_harness_aten_cuda.h"

DEFINE_string(tuner_proto, "", "Filename to load and store proto cache ");

TEST(WaveNet2Layers, SimpleAutotune) {
  // 1. Define and setup the TC compilation unit with CUDA memory
  // management backed by ATen tensors.
  std::string tc = R"TC(
def wavenet2layers(float(OUT, IN, KERN) Weight0,
    float(OUT) Bias0,
    float(BATCH, IN, KERN) Data0,
    float(IN, IN) ResWeight0,
    float(IN) ResBias0,
    float(SKIP, IN) SkipWeight0,
    float(SKIP) SkipBias0,
    float(OUT, IN, KERN) Weight1,
    float(OUT) Bias1,
    float(BATCH, IN, KERN) Data1,
    float(IN, IN) ResWeight1,
    float(IN) ResBias1,
    float(SKIP, IN) SkipWeight1,
    float(SKIP) SkipBias1)
    -> (Res0, Dilate0, NonLin0, Skip0, Res1, Dilate1, NonLin1, Skip1)
{
    Dilate0(batch, out)   =   Bias0(out)
            where batch in 0:BATCH
    Dilate0(batch, out)  += Weight0(out, r_in, r_kern) * Data0(batch, r_in, r_kern)
    NonLin0(batch, out)   = 1 / (1 + exp(-1*(Dilate0(batch, out))))
    NonLin0(batch, out)  *= tanh(Dilate0(batch, out + 64))
      Skip0(batch, skip)  =   SkipBias0(skip)
            where batch in 0:BATCH
      Skip0(batch, skip) += SkipWeight0(skip, r_in) * NonLin0(batch, r_in)
            where r_in in 0:IN # necessary because r_in gets into unresolved min/max
       Res0(batch, out)   =   ResBias0(  out)
            where batch in 0:BATCH
       Res0(batch, out)  += ResWeight0(  out, r_in) * NonLin0(batch, r_in)
            where r_in in 0:IN # necessary because r_in gets into unresolved min/max
       Res0(batch, out)   =       Res0(batch,  out) + NonLin0(batch, out)
            where out in 0:IN # necessary because r_in gets into unresolved min/max

    Dilate1(batch, out)   =   Bias1(out)
            where batch in 0:BATCH
    Dilate1(batch, out)  += Weight1(out, r_in, r_kern) * Data1(batch, r_in, r_kern)
    NonLin1(batch, out)   = 1 / (1 + exp(-1*(Dilate1(batch, out))))
    NonLin1(batch, out)  *= tanh(Dilate1(batch, out + 64))
      Skip1(batch, skip)  =   SkipBias1(skip)
            where batch in 0:BATCH
      Skip1(batch, skip) += SkipWeight1(skip, r_in) * NonLin1(batch, r_in)
            where r_in in 0:IN # necessary because r_in gets into unresolved min/max
       Res1(batch, out)   =   ResBias1(  out)
            where batch in 0:BATCH
       Res1(batch, out)  += ResWeight1(  out, r_in) * NonLin1(batch, r_in)
            where r_in in 0:IN # necessary because r_in gets into unresolved min/max
       Res1(batch, out)   =       Res1(batch,  out) + NonLin1(batch, out)
            where out in 0:IN # necessary because r_in gets into unresolved min/max
}
  )TC";
  tc::ATenCompilationUnit<tc::CudaTcExecutor> atCompl;
  atCompl.define(tc);

  // 2. Allocate tensors with random data.
  at::Tensor weight0 = at::CUDA(at::kFloat).rand({128, 64, 2});
  at::Tensor bias0 = at::CUDA(at::kFloat).rand({128});
  at::Tensor data0 = at::CUDA(at::kFloat).rand({1, 64, 2});
  at::Tensor res_weight0 = at::CUDA(at::kFloat).rand({64, 64});
  at::Tensor res_bias0 = at::CUDA(at::kFloat).rand({64});
  at::Tensor skip_weight0 = at::CUDA(at::kFloat).rand({256, 64});
  at::Tensor skip_bias0 = at::CUDA(at::kFloat).rand({256});

  at::Tensor weight1 = at::CUDA(at::kFloat).rand({128, 64, 2});
  at::Tensor bias1 = at::CUDA(at::kFloat).rand({128});
  at::Tensor data1 = at::CUDA(at::kFloat).rand({1, 64, 2});
  at::Tensor res_weight1 = at::CUDA(at::kFloat).rand({64, 64});
  at::Tensor res_bias1 = at::CUDA(at::kFloat).rand({64});
  at::Tensor skip_weight1 = at::CUDA(at::kFloat).rand({256, 64});
  at::Tensor skip_bias1 = at::CUDA(at::kFloat).rand({256});

  // 3. Run autotuning with evolutionary search starting from a naive option.
  auto naiveOptions = tc::CudaMappingOptions::makeNaiveCudaMappingOptions();
  tc::autotune::GeneticAutotunerATen geneticAutotuneATen(tc);
  auto bestOption = geneticAutotuneATen.tune(
      FLAGS_tuner_proto,
      "wavenet2layers",
      {weight0,
       bias0,
       data0,
       res_weight0,
       res_bias0,
       skip_weight0,
       skip_bias0,
       weight1,
       bias1,
       data1,
       res_weight1,
       res_bias1,
       skip_weight1,
       skip_bias1},
      naiveOptions);

  // 4. Compile and run the TC with the best option.
  // Outputs get allocated; could also be pre-allocated and passed.
  auto handle = atCompl.compile(
      "wavenet2layers",
      {weight0,
       bias0,
       data0,
       res_weight0,
       res_bias0,
       skip_weight0,
       skip_bias0,
       weight1,
       bias1,
       data1,
       res_weight1,
       res_bias1,
       skip_weight1,
       skip_bias1},
      bestOption.getValue());
  std::vector<at::Tensor> outputs;
  auto duration = atCompl.run(
      "wavenet2layers",
      {weight0,
       bias0,
       data0,
       res_weight0,
       res_bias0,
       skip_weight0,
       skip_bias0,
       weight1,
       bias1,
       data1,
       res_weight1,
       res_bias1,
       skip_weight1,
       skip_bias1},
      outputs,
      handle,
      true);
  std::cout
      << "wavenet2layers size weight0: " << weight0.sizes() << " ran in: "
      << std::chrono::duration_cast<std::chrono::microseconds>(duration).count()
      << "us\n";

  // 5. TODO: the following represent reasonable initialization
  // operations. Port them from PyTorch if we want to look at precision in the
  // future.
  //   weight0 = 5*(torch.cuda.FloatTensor(128,64,2).uniform_() - 0.5)
  //   bias0 = 2*(torch.cuda.FloatTensor(128).uniform_() - 0.5)
  //   data0 = 2*(torch.cuda.FloatTensor(1,64,2).uniform_() - 0.5)
  //   res_weight0 = 2*(torch.cuda.FloatTensor(64,64).uniform_() - 0.5)
  //   res_bias0 = 2*(torch.cuda.FloatTensor(64).uniform_() - 0.5)
  //   skip_weight0 = 2*(torch.cuda.FloatTensor(256,64).uniform_() - 0.5)
  //   skip_bias0 = 2*(torch.cuda.FloatTensor(256).uniform_() - 0.5)
  //
  //   weight1 = 5*(torch.cuda.FloatTensor(128,64,2).uniform_() - 1.5)
  //   bias1 = 2*(torch.cuda.FloatTensor(128).uniform_() - 1.5)
  //   data1 = 2*(torch.cuda.FloatTensor(1,64,2).uniform_() - 1.5)
  //   res_weight1 = 2*(torch.cuda.FloatTensor(64,64).uniform_() - 1.5)
  //   res_bias1 = 2*(torch.cuda.FloatTensor(64).uniform_() - 1.5)
  //   skip_weight1 = 2*(torch.cuda.FloatTensor(256,64).uniform_() - 1.5)
  //   skip_bias1 = 2*(torch.cuda.FloatTensor(256).uniform_() - 1.5)

  // 6. Run unchecked 1000 times, to put GPU in high usage mode, use with:
  //   nvprof --profile-from-start off executable --use_nvprof=1
  {
    tc::CudaProfiler cp;
    for (int i = 0; i < 1000; ++i) {
      atCompl.uncheckedRun(
          {weight0,
           bias0,
           data0,
           res_weight0,
           res_bias0,
           skip_weight0,
           skip_bias0,
           weight1,
           bias1,
           data1,
           res_weight1,
           res_bias1,
           skip_weight1,
           skip_bias1},
          outputs,
          handle);
    }
  }
}

// From root, run with:
//   ./build/examples/wavenet --tuner_threads=10 --tuner_gen_pop_size=10
//   --tuner_gen_generations=3 --tuner_gen_number_elites=4
//   --tuner_proto="/tmp/wavenet"
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  setAtenSeed(tc::initRandomSeed(), at::Backend::CUDA);
  return RUN_ALL_TESTS();
}
