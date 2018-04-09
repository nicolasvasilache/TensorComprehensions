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

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <vector>

#include "tc/core/utils/math.h"

namespace tc {
namespace autotune {

namespace detail {
template <typename Vector>
void mergeVectorsImpl(Vector&) {}

template <typename Vector, typename... Vectors>
void mergeVectorsImpl(Vector& sink, Vector&& v, Vectors&&... vs) {
  mergeVectorsImpl(sink, std::forward<Vectors>(vs)...);

  if (std::is_rvalue_reference<decltype(v)>::value) {
    sink.reserve(sink.size() + v.size());
    std::move(v.begin(), v.end(), std::back_inserter(sink));
  } else {
    sink.insert(sink.end(), v.begin(), v.end());
  }
}

} // namespace detail

inline std::vector<std::size_t> powers2andCeilDivisors(std::size_t val) {
  auto numPowers = static_cast<std::size_t>(std::ceil(std::log2(val)));
  // 1. generate `numPowers' powers of 2
  std::vector<std::size_t> res(numPowers + 1);
  std::size_t p = 1;
  std::generate(res.begin(), res.end(), [p]() mutable {
    auto old_p = p;
    p *= 2;
    return old_p;
  });
  // 2. additionally insert ceil(val / powers2)
  res.reserve(res.size() * 2);
  for (std::size_t i = 0, s = res.size(); i < s; ++i) {
    if (res[i] > val) {
      continue;
    }
    res.push_back(std::ceil(static_cast<double>(val) / res[i]));
  }
  std::sort(res.begin(), res.end());
  res.erase(std::unique(res.begin(), res.end()), res.end());
  return res;
}
template <typename Vector, typename... Vectors>
Vector mergeVectors(Vector&& v, Vectors&&... vs) {
  Vector merged;
  detail::mergeVectorsImpl(
      merged, std::forward<Vector>(v), std::forward<Vectors>(vs)...);
  std::sort(merged.begin(), merged.end());
  merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
  return merged;
}

template <typename OptionsCacheType>
std::vector<typename OptionsCacheType::MappingOptionsType> restoreCandidates(
    const lang::CanonicalTcString& tc,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs,
    const std::string& deviceStr) {
  auto candidates = getOptionsAndMedianRuntimes<OptionsCacheType>(
      tc, inputs, outputs, deviceStr);
  LOG_IF(INFO, candidates.size() < FLAGS_tuner_gen_restore_number)
      << "Requested " << FLAGS_tuner_gen_restore_number
      << " candidates but there are only " << candidates.size() << " in cache.";
  auto restoreNumber =
      std::min(candidates.size(), size_t(FLAGS_tuner_gen_restore_number));
  using MappingOptions = typename OptionsCacheType::MappingOptionsType;
  std::sort(
      candidates.begin(),
      candidates.end(),
      [](const OptionsWithMedianTime<MappingOptions>& a,
         const OptionsWithMedianTime<MappingOptions>& b) {
        return a.medianRuntime < b.medianRuntime;
      });
  std::vector<CudaMappingOptions> res;
  res.reserve(restoreNumber);
  std::transform(
      candidates.begin(),
      candidates.begin() + restoreNumber,
      std::back_inserter(res),
      [](const OptionsWithMedianTime<MappingOptions>& rr) {
        return rr.options;
      });
  return res;
}

template <typename OptionsCacheType>
llvm::Optional<typename OptionsCacheType::MappingOptionsType> getBestOptions(
    const lang::CanonicalTcString& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs,
    const std::string& deviceStr) {
  auto bestOptions = OptionsCacheType::getCache()->retrieveBestOptions(
      id, inputs, outputs, deviceStr);
  if (bestOptions) {
    return *bestOptions;
  }
  return llvm::Optional<CudaMappingOptions>{};
}

template <typename OptionsCacheType>
std::vector<
    OptionsWithMedianTime<typename OptionsCacheType::MappingOptionsType>>
getOptionsAndMedianRuntimes(
    const lang::CanonicalTcString& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs,
    const std::string& deviceStr) {
  auto candidates = OptionsCacheType::getCache()->retrieveOptionsAndRuntimes(
      id, inputs, outputs, deviceStr);

  using MappingOptions = typename OptionsCacheType::MappingOptionsType;
  std::vector<OptionsWithMedianTime<MappingOptions>> c;
  c.reserve(candidates.size());
  std::transform(
      candidates.begin(),
      candidates.end(),
      std::back_inserter(c),
      [](const typename OptionsCacheType::RetrievalResult& rr)
          -> OptionsWithMedianTime<MappingOptions> {
        return {std::move(rr.options), median(rr.recordedRuntimes)};
      });
  return c;
}
} // namespace autotune
} // namespace tc
