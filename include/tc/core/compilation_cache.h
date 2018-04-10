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
#include <version.h>

#include "tc/core/utils/time.h"

namespace tc {

namespace detail {
template <typename CachedEntryType, typename TensorType>
const CachedEntryType* searchKernel(
    const std::vector<CachedEntryType>& entries,
    const std::string& id,
    const std::vector<TensorType>& inputs,
    const std::vector<TensorType>& outputs,
    const std::string& deviceStr);
template <typename CachedEntryType, typename TensorType>
CachedEntryType* searchKernel(
    std::vector<CachedEntryType>& entries,
    const std::string& id,
    const std::vector<TensorType>& inputs,
    const std::vector<TensorType>& outputs,
    const std::string& deviceStr);
template <typename CachedEntryType, typename TensorType>
const CachedEntryType* searchKernel(
    const std::vector<CachedEntryType>& entries,
    const std::string& id,
    const typename CachedEntryType::MappingOptionsType& options,
    const std::vector<TensorType>& inputs,
    const std::vector<TensorType>& outputs,
    const std::string& deviceStr);
template <typename CachedEntryType, typename TensorType>
CachedEntryType* searchKernel(
    std::vector<CachedEntryType>& entries,
    const std::string& id,
    const typename CachedEntryType::MappingOptionsType& options,
    const std::vector<TensorType>& inputs,
    const std::vector<TensorType>& outputs,
    const std::string& deviceStr);

/**
 * TensorInfo wraps the necessary bits of DLTensor that are used as part of the
 * CompilationCache's entry keys.
 *
 * It is serializable to protobuf and stored directly in the cache.
 */
struct TensorInfo {
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  uint64_t alignment;
  DLDataType dType;

  TensorInfo(const DLTensor* t);
  TensorInfo(const TensorInfoProto& buf);

  bool operator==(const DLTensor* t) const;
  bool operator==(const TensorInfo& t) const;
  bool operator<(const TensorInfo& t) const;
  TensorInfoProto toProtobuf() const;
};
} // namespace detail

template <typename CC, typename CachedEntryType>
class Cache {
 public:
  static std::shared_ptr<CC>& getGlobalSharedCache();

  static void enableCache();
  static void disableCache();
  static void dumpCacheToProtobuf(const std::string& filename);
  static void loadCacheFromProtobuf(const std::string& filename);
  template <typename Protobuf>
  static void loadCacheFromProtobuf(const Protobuf& buf);
  static std::shared_ptr<CC> getCache();
  static bool cacheEnabled();

  typename std::vector<CachedEntryType>::const_iterator begin() const {
    return entries_.begin();
  }
  typename std::vector<CachedEntryType>::const_iterator end() const {
    return entries_.end();
  }
  size_t size() const;
  void clear();

  mutable int numberAttemptedRetrievals = 0;
  mutable int numberSuccessfulRetrievals = 0;
  mutable int numberCacheAttemps = 0;

 protected:
  // XXX:this should be a std or boost shared_mutex
  mutable std::mutex mtx_;

  std::vector<CachedEntryType> entries_;
};

std::vector<detail::TensorInfo> DLTensorToTensorInfoVector(
    const std::vector<const DLTensor*>& ts);
std::vector<detail::TensorInfo> ProtoToTensorInfoVector(
    const google::protobuf::RepeatedPtrField<TensorInfoProto>& buf);

////////////////////////////////////////////////////////////////////////////////
// OptionsCache
////////////////////////////////////////////////////////////////////////////////
/**
 * An OptionsCache holds multiple OptionsCachedEntry's.
 * Each OptionsCachedEntry is split to two conceptual parts the key and the
 * values. The key is: the kernel/op's unique id (string), the specialized input
 * dimensions, the target architecture (string), tc's version
 * (string), The values are a vector of: the isl options used
 * when the kernel was optimized, profiling information
 */
template <typename MappingOptionsT>
struct OptionsCachedEntry {
  using MappingOptionsType = MappingOptionsT;

  OptionsCachedEntry(
      const std::string& id,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs,
      const std::string& deviceStr,
      const MappingOptionsType& options,
      Duration runtime);
  OptionsCachedEntry(const OptionsCacheEntryProto& buf);
  OptionsCacheEntryProto toProtobuf() const;

  struct Key {
    Key(const std::string& id,
        const std::vector<const DLTensor*>& inputs,
        const std::vector<const DLTensor*>& outputs,
        const std::string& deviceStr,
        const std::string& gitVersion);

    Key(const std::string& id,
        std::vector<detail::TensorInfo>&& inputs,
        std::vector<detail::TensorInfo>&& outputs,
        const std::string& deviceStr,
        const std::string& gitVersion);

    std::string id;
    std::vector<detail::TensorInfo> inputs;
    std::vector<detail::TensorInfo> outputs;
    std::string deviceStr;
    std::string gitVersion;
  };

  struct Values {
    Values(const MappingOptionsType& options, Duration runtime);
    Values(const MappingOptionsType& options, std::vector<Duration>&& runtimes);
    MappingOptionsType mappingOptions;
    std::vector<Duration> recordedRuntimes;
  };
  Key key;
  std::vector<Values> values;
};

template <typename MappingOptionsType>
struct OptionsCacheRetrievalResult {
  MappingOptionsType options;
  std::vector<Duration> recordedRuntimes;
};

template <typename OptionsCachedEntryType>
class OptionsCache : public Cache<
                         OptionsCache<OptionsCachedEntryType>,
                         OptionsCachedEntryType> {
 public:
  using ProtobufType = OptionsCacheProto;
  using CachedEntryType = OptionsCachedEntryType;
  using MappingOptionsType = typename CachedEntryType::MappingOptionsType;
  using RetrievalResult = OptionsCacheRetrievalResult<MappingOptionsType>;

  OptionsCache() = default;
  OptionsCache(const OptionsCacheProto& buf);

  OptionsCacheProto toProtobuf() const;

  // returns the sum of cache entry sizes (that is a single cache entry can have
  // multiple options and profiling information associated with it)
  size_t totalSize() const;

  void recordRuntime(
      const std::string& id,
      const MappingOptionsType& options,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs,
      const std::string& deviceStr,
      Duration runtime);

  std::vector<RetrievalResult> retrieveOptionsAndRuntimes(
      const std::string& id,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs,
      const std::string& deviceStr) const;

  std::unique_ptr<MappingOptionsType> retrieveBestOptions(
      const std::string& id,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs,
      const std::string& deviceStr) const;

  std::vector<MappingOptionsType> retrieveTopKOptions(
      const std::string& id,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<const DLTensor*>& outputs,
      const std::string& deviceStr,
      size_t k) const;

  // Only (up to) numberToKeep entries per operation (combination of id and
  // input info) are kept in the cache. The best performing versions are kept
  void keepOnlyBestCandidates(size_t numberToKeep);
};

class CacheEntrySameKeyDifferentValue : public std::invalid_argument {
 public:
  explicit CacheEntrySameKeyDifferentValue(const std::string& what_arg)
      : invalid_argument(what_arg) {}
  explicit CacheEntrySameKeyDifferentValue(const char* what_arg)
      : invalid_argument(what_arg) {}
};

bool operator==(
    const std::vector<const DLTensor*>& inputsTensor,
    const std::vector<detail::TensorInfo>& inputsInfo);

inline std::string makeOptionsFilename(const std::string& filename) {
  return filename + ".options";
}

inline std::string makeCudaFilename(const std::string& filename) {
  return filename + ".cuda";
}
} // namespace tc

#include "tc/core/compilation_cache-inl.h"
