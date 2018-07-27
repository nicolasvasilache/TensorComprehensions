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
#include "tc/core/cpu/cpu_tc_executor.h"

#include "tc/core/cpu/cpu_mapping_options.h"
#include "tc/core/cpu/cpu_mapping_options_cpp_printer.h"
#include "tc/core/halide_utils.h"
#include "tc/core/polyhedral/cpu/mapped_scop.h"
#include "tc/core/tc2halide.h"
#include "tc/core/tensor.h"
#include "tc/lang/parser.h"
#include "tc/lang/sema.h"
#include "version.h"

namespace tc {
CpuTcExecutor::CpuTcExecutor(
    const std::vector<TensorInfo>& inputsInfo,
    const std::vector<TensorInfo>& outputsInfo,
    const tc2halide::HalideComponents& halideComponents,
    const typename CpuBackend::CompilationResultType& compilationResult)
    : TcExecutor<CpuBackend>(
          inputsInfo,
          outputsInfo,
          halideComponents,
          compilationResult) {
  LOG(ERROR) << "NYI: CpuTcExecutor::CpuTcExecutor setup RTC";
}

CpuCompilationResult CpuBackend::compileWithTcMapper(
    const std::string& tcName,
    tc2halide::HalideComponents halideComponents,
    const std::vector<const DLConstTensor*>& inputs,
    /* TODO: in the future also pass outputs for stride and alignment info */
    const CpuMappingOptions& options) {
  auto scop = makeScop<CpuBackend>(tcName, halideComponents, inputs, options);
  auto mappedScop = polyhedral::cpu::MappedScop::makeSequential(
    std::move(scop), options);
  LOG_IF(INFO, FLAGS_debug_tc_mapper) << "Mapped schedule:" << std::endl
                                      << *(mappedScop->schedule());

  auto parameters = mappedScop->scop().getParameterValues();
  auto specializedName = specializeKernelName(tcName, parameters);
  auto pJit = mappedScop->codegen(specializedName);
  // auto fptr =
  //     reinterpret_cast<void (*)(float*, float*, float*, int, int, int, int)>(
  //         pJit->getSymbolAddress(specializedName));

  return CpuCompilationResult{
    std::string("source"), specializedName, parameters};
}

void CpuTcExecutor::uncheckedRun(
    const std::vector<const void*>& inputs,
    const std::vector<void*>& outputs,
    typename CpuBackend::RuntimeInformation info) const {
  LOG(ERROR) << "NYI: CpuTcExecutor::uncheckedRun";
}

ProfilingInfo CpuTcExecutor::profileUnchecked(
    const std::vector<const void*>& inputs,
    const std::vector<void*>& outputs) const {
  LOG(ERROR) << "NYI: CpuTcExecutor::profileUnchecked";
  return ProfilingInfo{Duration::max(), Duration::max()};
}
} // namespace tc
