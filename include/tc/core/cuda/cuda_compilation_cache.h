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

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <dlpack/dlpack.h>

#include <compcache.pb.h>

#include "tc/core/compilation_cache.h"
#include "tc/core/cuda/cuda.h"
#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/cuda/cuda_rtc.h"
#include "tc/core/utils/time.h"

namespace tc {

////////////////////////////////////////////////////////////////////////////////
// CudaCache
////////////////////////////////////////////////////////////////////////////////
struct CudaCachedEntry {
  using MappingOptionsType = CudaMappingOptions;

  CudaCachedEntry(
      const std::string& id,
      const std::string& kernelSpecializedName,
      const std::vector<int>& kernelParameters,
      const Grid& grid,
      const Block& block,
      const CudaMappingOptions& mappingOptions,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs,
      const std::string& cudaSource,
      const std::string& deviceStr);

  CudaCachedEntry(const CudaCacheEntryProto& buf);
  CudaCacheEntryProto toProtobuf() const;

  struct Key {
    std::string id;
    CudaMappingOptions mappingOptions;
    std::vector<detail::TensorInfo> inputs;
    std::vector<detail::TensorInfo> outputs;
    std::string deviceStr;
    std::string gitVersion;
  };

  struct Values {
    std::string cudaSource;
    std::string kernelSpecializedName;
    std::vector<int> kernelParameters;
    Grid grid;
    Block block;
  };
  Key key;
  Values values;
};

struct CudaCacheRetrievalResult {
  std::string source;
  std::string specializedName;
  std::vector<int> parameters;
  Grid grid;
  Block block;
};

using CudaOptionsCache = OptionsCache<OptionsCachedEntry<CudaMappingOptions>>;

/**
 * CudaCache stores the Cuda source of optimized kernels
 * A CudaCache holds multiple CudaCachedEntry's.
 * Each CudaCachedEntry is split to two conceptual parts the key and the values.
 * The values are:
 *                  the specialized (wrt inputs) Cuda source code,
 *                  the kernel's specialized name,
 *                  the kernel parameters,
 *                  the Cuda block and grid dimensions
 * The key is:
 *                  the kernel/op's unique id (string),
 *                  the specialized input dimensions,
 *                  the isl options when the kernel was optimized,
 *                  the target architecture (string),
 *                  tc's version (string),
 */
class CudaCache : public Cache<CudaCache, CudaCachedEntry> {
 public:
  typedef CudaCacheProto ProtobufType;

  CudaCache() = default;
  CudaCache(const CudaCacheProto& buf);
  CudaCacheProto toProtobuf() const;

  /**
   * If op was previously cached and the inputs' shape, isl options, and the
   * target device are the same then this is a noop
   * Else (cudaSource, grid, block) is stored in the cache
   */
  void cacheKernel(CudaCachedEntry&& entry);

  /**
   * Returns the cache entry that matches op (id, isl options, target device)
   * and inputs' shapes.
   */
  std::unique_ptr<CudaCacheRetrievalResult> retrieveKernel(
      const std::string& id,
      const CudaMappingOptions& options,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs) const;

  void removeEntriesNotInOptionsCache(const CudaOptionsCache& oc);
};

////////////////////////////////////////////////////////////////////////////////
// ManualCudaCache
////////////////////////////////////////////////////////////////////////////////
/*
 * ManualCudaCache stores the manually injected source of Cuda kernels
 * It is just a CUDA cache with an ignored CudaMappingOptions
 */
class ManualCudaCache : public Cache<ManualCudaCache, CudaCachedEntry> {
 public:
  using ProtobufType = ManualCudaCacheProto;
  using CachedEntry = CudaCachedEntry;
  using RetrievalResult = CudaCacheRetrievalResult;

  ManualCudaCache() = default;
  ManualCudaCache(const ProtobufType& buf);
  ProtobufType toProtobuf() const;

  /*
   * Stores:
   *   (cudaSource, grid, block, specializedName, parameters)
   * in the cache with key:
   *   (id, input shapes, output shapes, target device).
   * If the key already exist in the cache, the values are replaced.
   */
  void cacheKernel(CachedEntry&& entry);

  std::unique_ptr<RetrievalResult> retrieveKernel(
      const std::string& id,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs) const;
};

////////////////////////////////////////////////////////////////////////////////
// Free functions
////////////////////////////////////////////////////////////////////////////////
inline void removeFromCudaCacheEntriesNotInOptionsCache(
    CudaCache& cc,
    const CudaOptionsCache& oc) {
  cc.removeEntriesNotInOptionsCache(oc);
}
} // namespace tc
