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
#include "tc/core/cuda/cuda_compilation_cache.h"

#include <version.h>

#include <cstdint>
#include <fstream>
#include <numeric>
#include <tuple>

#include "tc/core/compilation_cache.h"
#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/utils/math.h"

namespace tc {

namespace {
template <typename Array, typename Buf>
void WriteProtobufArray(const Array& arr, Buf* buf) {
  google::protobuf::RepeatedField<typename Array::value_type> data(
      arr.begin(), arr.end());
  buf->Swap(&data);
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
// CudaCache
////////////////////////////////////////////////////////////////////////////////
std::shared_ptr<CudaCache>& CudaCache::getGlobalSharedCache() {
  static std::shared_ptr<CudaCache> cudaCache_;
  return cudaCache_;
}

CudaCachedEntry::CudaCachedEntry(
    const std::string& id,
    const std::string& kernelSpecializedName,
    const std::vector<int>& kernelParameters,
    const Grid& grid,
    const Block& block,
    const CudaMappingOptions& mappingOptions,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs,
    const std::string& cudaSource,
    const std::string& deviceStr)
    : key{id,
          mappingOptions,
          DLTensorToTensorInfoVector(inputs),
          DLTensorToTensorInfoVector(outputs),
          deviceStr,
          git_version},
      values{cudaSource, kernelSpecializedName, kernelParameters, grid, block} {
}

CudaCachedEntry::CudaCachedEntry(const CudaCacheEntryProto& buf)
    : key{buf.id(),
          CudaMappingOptions{buf.kernel_options()},
          ProtoToTensorInfoVector(buf.inputs()),
          ProtoToTensorInfoVector(buf.outputs()),
          buf.device_str(),
          buf.git_version()},
      values{buf.cuda_source(),
             buf.specialized_name(),
             std::vector<int>{buf.parameters().begin(), buf.parameters().end()},
             Grid(buf.grid_dims()),
             Block(buf.block_dims())} {}

CudaCache::CudaCache(const CudaCacheProto& buf) {
  entries_.reserve(buf.entries_size());
  for (const auto& entry_buf : buf.entries())
    entries_.emplace_back(entry_buf);
}

void CudaCache::cacheKernel(CudaCachedEntry&& entry) {
  std::lock_guard<std::mutex> lock(mtx_);
  ++numberCacheAttemps;
  auto retrievedEntry = searchKernel(
      entries_,
      entry.key.id,
      entry.key.mappingOptions,
      entry.key.inputs,
      entry.key.outputs,
      CudaGPUInfo::GPUInfo().GetCudaDeviceStr());
  if (retrievedEntry) {
    if (retrievedEntry->values.cudaSource == entry.values.cudaSource or
        retrievedEntry->values.grid == entry.values.grid or
        retrievedEntry->values.block == entry.values.block) {
      throw CacheEntrySameKeyDifferentValue(
          "CudaCache::CacheKernel: a kernel matching the id, options and "
          "inputs was previously cached with different cuda source or block "
          "or grid dimensions.");
    }
    return;
  }
  entries_.emplace_back(entry);
}

std::unique_ptr<CudaCacheRetrievalResult> CudaCache::retrieveKernel(
    const std::string& id,
    const CudaMappingOptions& options,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) const {
  std::lock_guard<std::mutex> lock(mtx_);
  ++numberAttemptedRetrievals;
  auto entry = detail::searchKernel(
      entries_,
      id,
      options,
      inputs,
      outputs,
      CudaGPUInfo::GPUInfo().GetCudaDeviceStr());
  if (not entry) {
    return nullptr;
  }
  ++numberSuccessfulRetrievals;
  return std::unique_ptr<CudaCacheRetrievalResult>(
      new CudaCacheRetrievalResult{entry->values.cudaSource,
                                   entry->values.kernelSpecializedName,
                                   entry->values.kernelParameters,
                                   entry->values.grid,
                                   entry->values.block});
}

void CudaCache::removeEntriesNotInOptionsCache(const CudaOptionsCache& oc) {
  std::vector<CudaCachedEntry> newEntries;
  for (const auto& entry : oc) {
    for (const auto& options : entry.values) {
      auto cudaEntry = detail::searchKernel(
          entries_,
          entry.key.id,
          options.mappingOptions,
          entry.key.inputs,
          entry.key.outputs,
          CudaGPUInfo::GPUInfo().GetCudaDeviceStr());
      if (cudaEntry) {
        newEntries.push_back(std::move(*cudaEntry));
      }
    }
  }
  entries_ = std::move(newEntries);
}

CudaCacheProto CudaCache::toProtobuf() const {
  CudaCacheProto buf;
  auto* entriesBuf = buf.mutable_entries();
  entriesBuf->Reserve(entries_.size());
  std::transform(
      entries_.begin(),
      entries_.end(),
      google::protobuf::RepeatedPtrFieldBackInserter(entriesBuf),
      [](const CudaCachedEntry& entry) { return entry.toProtobuf(); });
  return buf;
}

CudaCacheEntryProto CudaCachedEntry::toProtobuf() const {
  CudaCacheEntryProto buf;
  buf.set_id(key.id);
  *buf.mutable_kernel_options() = key.mappingOptions.proto();
  std::transform(
      key.inputs.begin(),
      key.inputs.end(),
      google::protobuf::RepeatedPtrFieldBackInserter(buf.mutable_inputs()),
      [](const detail::TensorInfo& input) { return input.toProtobuf(); });
  std::transform(
      key.outputs.begin(),
      key.outputs.end(),
      google::protobuf::RepeatedPtrFieldBackInserter(buf.mutable_outputs()),
      [](const detail::TensorInfo& output) { return output.toProtobuf(); });
  buf.set_device_str(key.deviceStr);
  buf.set_git_version(key.gitVersion);

  buf.set_cuda_source(values.cudaSource);
  *buf.mutable_grid_dims() = values.grid.view.proto;
  *buf.mutable_block_dims() = values.block.view.proto;
  buf.set_specialized_name(values.kernelSpecializedName);
  WriteProtobufArray(values.kernelParameters, buf.mutable_parameters());

  return buf;
}

////////////////////////////////////////////////////////////////////////////////
// ManualCudaCache
////////////////////////////////////////////////////////////////////////////////
std::shared_ptr<ManualCudaCache>& ManualCudaCache::getGlobalSharedCache() {
  static std::shared_ptr<ManualCudaCache> manualCudaCache_;
  return manualCudaCache_;
}

ManualCudaCachedEntry::ManualCudaCachedEntry(
    const std::string& id,
    const std::string& kernelSpecializedName,
    const std::vector<int>& kernelParameters,
    const Grid& grid,
    const Block& block,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs,
    const std::string& cudaSource,
    const std::string& deviceStr)
    : key{id,
          DLTensorToTensorInfoVector(inputs),
          DLTensorToTensorInfoVector(outputs),
          deviceStr,
          git_version},
      values{cudaSource, kernelSpecializedName, kernelParameters, grid, block} {
}

void ManualCudaCache::cacheKernel(ManualCudaCachedEntry&& entry) {
  std::lock_guard<std::mutex> lock(mtx_);
  ++numberCacheAttemps;
  auto retrievedEntry = detail::searchKernel(
      entries_,
      entry.key.id,
      entry.key.inputs,
      entry.key.outputs,
      CudaGPUInfo::GPUInfo().GetCudaDeviceStr());
  if (retrievedEntry) {
    retrievedEntry->values.grid = entry.values.grid;
    retrievedEntry->values.block = entry.values.block;
    retrievedEntry->values.cudaSource = entry.values.cudaSource;
    retrievedEntry->values.kernelSpecializedName =
        entry.values.kernelSpecializedName;
    retrievedEntry->values.kernelParameters = entry.values.kernelParameters;
    return;
  }
  entries_.emplace_back(entry);
}

std::unique_ptr<ManualCudaCacheRetrievalResult> ManualCudaCache::retrieveKernel(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs) const {
  std::lock_guard<std::mutex> lock(mtx_);
  ++numberAttemptedRetrievals;
  auto entry = detail::searchKernel(
      entries_, id, inputs, outputs, CudaGPUInfo::GPUInfo().GetCudaDeviceStr());
  if (not entry) {
    return nullptr;
  }
  ++numberSuccessfulRetrievals;
  return std::unique_ptr<ManualCudaCacheRetrievalResult>(
      new ManualCudaCacheRetrievalResult{entry->values.cudaSource,
                                         entry->values.kernelSpecializedName,
                                         entry->values.kernelParameters,
                                         entry->values.grid,
                                         entry->values.block});
}
} // namespace tc
