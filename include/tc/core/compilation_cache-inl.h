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
#include <iostream>

#include "tc/core/utils/math.h"

namespace tc {

namespace detail {
template <typename CachedEntryType, typename TensorType>
const CachedEntryType* searchKernel(
    const std::vector<CachedEntryType>& entries,
    const std::string& id,
    const std::vector<TensorType>& inputs,
    const std::vector<TensorType>& outputs,
    const std::string& deviceStr) {
  auto it = std::find_if(
      entries.begin(), entries.end(), [&](const CachedEntryType& c) {
        using tc::operator==;
        return id == c.key.id && inputs == c.key.inputs &&
            outputs == c.key.outputs && deviceStr == c.key.deviceStr;
      });
  if (it != entries.end()) {
    if (it->key.gitVersion != tc::git_version) {
      std::cerr << "[WARNING] Proto version doesn't match. TC git version is: "
                << tc::git_version
                << " and Proto version is: " << it->key.gitVersion
                << " .This proto might be incompatible"
                << " with your TC binary and can break. Please autotune"
                << " against the correct TC version." << std::endl;
    }
    return &*it;
  }
  return nullptr;
}

template <typename CachedEntryType, typename TensorType>
CachedEntryType* searchKernel(
    std::vector<CachedEntryType>& entries,
    const std::string& id,
    const std::vector<TensorType>& inputs,
    const std::vector<TensorType>& outputs,
    const std::string& deviceStr) {
  return const_cast<CachedEntryType*>(searchKernel(
      static_cast<const std::vector<CachedEntryType>&>(entries),
      id,
      inputs,
      outputs,
      deviceStr));
}

template <typename CachedEntryType, typename TensorType>
const CachedEntryType* searchKernel(
    const std::vector<CachedEntryType>& entries,
    const std::string& id,
    const typename CachedEntryType::MappingOptionsType& options,
    const std::vector<TensorType>& inputs,
    const std::vector<TensorType>& outputs,
    const std::string& deviceStr) {
  auto it = std::find_if(
      entries.begin(), entries.end(), [&](const CachedEntryType& c) {
        using tc::operator==;
        return id == c.key.id && options == c.key.mappingOptions &&
            inputs == c.key.inputs && outputs == c.key.outputs &&
            deviceStr == c.key.deviceStr;
      });
  if (it != entries.end()) {
    if (it->key.gitVersion != tc::git_version) {
      std::cerr << "[WARNING] Proto version doesn't match. TC git version is: "
                << tc::git_version
                << " and Proto version is: " << it->key.gitVersion
                << " .This proto might be incompatible"
                << " with your TC binary and can break. Please autotune"
                << " against the correct TC version." << std::endl;
    }
    return &*it;
  }
  return nullptr;
}

template <typename CachedEntryType, typename TensorType>
CachedEntryType* searchKernel(
    std::vector<CachedEntryType>& entries,
    const std::string& id,
    const typename CachedEntryType::MappingOptionsType& options,
    const std::vector<TensorType>& inputs,
    const std::vector<TensorType>& outputs,
    const std::string& deviceStr) {
  return const_cast<CachedEntryType*>(searchKernel(
      static_cast<const std::vector<CachedEntryType>&>(entries),
      id,
      options,
      inputs,
      outputs,
      deviceStr));
}
} // namespace detail

////////////////////////////////////////////////////////////////////////////////
// OptionsCache
////////////////////////////////////////////////////////////////////////////////
template <typename MappingOptionsType>
OptionsCachedEntry<MappingOptionsType>::OptionsCachedEntry(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs,
    const std::string& deviceStr,
    const MappingOptionsType& options,
    Duration runtime)
    : key(id, inputs, outputs, deviceStr, git_version) {
  values.emplace_back(options, runtime);
}

template <typename MappingOptionsType>
OptionsCachedEntry<MappingOptionsType>::Key::Key(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs_,
    const std::vector<const DLTensor*>& outputs_,
    const std::string& deviceStr,
    const std::string& gitVersion)
    : Key(id,
          DLTensorToTensorInfoVector(inputs_),
          DLTensorToTensorInfoVector(outputs_),
          deviceStr,
          gitVersion) {}

template <typename MappingOptionsType>
OptionsCachedEntry<MappingOptionsType>::Key::Key(
    const std::string& id,
    std::vector<detail::TensorInfo>&& inputs_,
    std::vector<detail::TensorInfo>&& outputs_,
    const std::string& deviceStr,
    const std::string& gitVersion)
    : id(id),
      inputs(std::move(inputs_)),
      outputs(std::move(outputs_)),
      deviceStr(deviceStr),
      gitVersion(gitVersion) {}

template <typename MappingOptionsType>
OptionsCachedEntry<MappingOptionsType>::Values::Values(
    const MappingOptionsType& options,
    Duration runtime)
    : mappingOptions(options), recordedRuntimes{runtime} {}

template <typename MappingOptionsType>
OptionsCachedEntry<MappingOptionsType>::Values::Values(
    const MappingOptionsType& options,
    std::vector<Duration>&& runtimes)
    : mappingOptions(options), recordedRuntimes(std::move(runtimes)) {}

template <typename MappingOptionsType>
OptionsCachedEntry<MappingOptionsType>::OptionsCachedEntry(
    const OptionsCacheEntryProto& buf)
    : key(buf.id(),
          ProtoToTensorInfoVector(buf.inputs()),
          ProtoToTensorInfoVector(buf.outputs()),
          buf.device_str(),
          buf.git_version()) {
  if (buf.values_size() == 0) {
    throw std::invalid_argument(
        "OptionsCachedEntry invalid protobuf: each entry should have "
        "at least one value field.");
  }

  for (const auto& value : buf.values()) {
    if (value.recorded_runtimes_size() == 0) {
      throw std::invalid_argument(
          "OptionsCachedEntry invalid protobuf: each entry value "
          "should have at least one recorded runtime.");
    }
    std::vector<Duration> runtimes;
    runtimes.reserve(value.recorded_runtimes_size());
    std::transform(
        value.recorded_runtimes().begin(),
        value.recorded_runtimes().end(),
        std::back_inserter(runtimes),
        [](int64_t us) { return std::chrono::microseconds(us); });
    values.emplace_back(
        MappingOptionsType(value.kernel_options()), std::move(runtimes));
  }
}

template <typename MappingOptionsType>
OptionsCacheEntryProto OptionsCachedEntry<MappingOptionsType>::toProtobuf()
    const {
  OptionsCacheEntryProto buf;
  buf.set_id(key.id);
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

  std::transform(
      values.begin(),
      values.end(),
      google::protobuf::RepeatedPtrFieldBackInserter(buf.mutable_values()),
      [](const Values& v) {
        OptionsCacheValuesProto buf;
        *buf.mutable_kernel_options() = v.mappingOptions.proto();
        for (const auto& r : v.recordedRuntimes) {
          buf.add_recorded_runtimes(
              std::chrono::duration_cast<std::chrono::microseconds>(r).count());
        }
        return buf;
      });
  return buf;
}

template <typename MappingOptionsType>
std::shared_ptr<OptionsCache<MappingOptionsType>>&
OptionsCache<MappingOptionsType>::getGlobalSharedCache() {
  static std::shared_ptr<OptionsCache> optionsCache_;
  return optionsCache_;
}

template <typename MappingOptionsType>
OptionsCache<MappingOptionsType>::OptionsCache(const OptionsCacheProto& buf) {
  this->entries_.reserve(buf.entries_size());
  for (const auto& entry_buf : buf.entries())
    this->entries_.emplace_back(entry_buf);
}

template <typename MappingOptionsType>
OptionsCacheProto OptionsCache<MappingOptionsType>::toProtobuf() const {
  OptionsCacheProto buf;
  auto* entriesBuf = buf.mutable_entries();
  entriesBuf->Reserve(this->entries_.size());
  std::transform(
      this->entries_.begin(),
      this->entries_.end(),
      google::protobuf::RepeatedPtrFieldBackInserter(entriesBuf),
      [](const OptionsCachedEntry<MappingOptionsType>& entry) {
        return entry.toProtobuf();
      });
  return buf;
}

template <typename MappingOptionsType>
size_t OptionsCache<MappingOptionsType>::totalSize() const {
  std::lock_guard<std::mutex> lock(this->mtx_);
  return std::accumulate(
      this->entries_.begin(),
      this->entries_.end(),
      size_t(0),
      [](size_t sum, const OptionsCachedEntry<MappingOptionsType>& e) {
        return sum + e.values.size();
      });
}

template <typename MappingOptionsType>
void OptionsCache<MappingOptionsType>::recordRuntime(
    const std::string& id,
    const MappingOptionsType& options,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs,
    const std::string& deviceStr,
    Duration runtime) {
  std::lock_guard<std::mutex> lock(this->mtx_);
  ++this->numberCacheAttemps;
  auto kernel =
      detail::searchKernel(this->entries_, id, inputs, outputs, deviceStr);
  if (not kernel) {
    this->entries_.emplace_back(
        id, inputs, outputs, deviceStr, options, runtime);
    return;
  }
  auto v = std::find_if(
      kernel->values.begin(),
      kernel->values.end(),
      [&options](
          const typename OptionsCachedEntry<MappingOptionsType>::Values& v) {
        return v.mappingOptions == options;
      });
  if (v == kernel->values.end()) {
    kernel->values.emplace_back(options, runtime);
    return;
  }

  v->recordedRuntimes.push_back(runtime);
}

template <typename OptionsCacheEntryType>
std::vector<OptionsCacheRetrievalResult<
    typename OptionsCacheEntryType::MappingOptionsType>>
OptionsCache<OptionsCacheEntryType>::retrieveOptionsAndRuntimes(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs,
    const std::string& deviceStr) const {
  std::lock_guard<std::mutex> lock(this->mtx_);
  ++this->numberAttemptedRetrievals;
  auto ret =
      detail::searchKernel(this->entries_, id, inputs, outputs, deviceStr);
  if (not ret) {
    return {};
  }
  ++this->numberSuccessfulRetrievals;
  std::vector<RetrievalResult> res;
  res.reserve(ret->values.size());
  std::transform(
      ret->values.begin(),
      ret->values.end(),
      std::back_inserter(res),
      [](const typename CachedEntryType::Values& v) -> RetrievalResult {
        return {v.mappingOptions, v.recordedRuntimes};
      });
  return res;
}

template <typename OptionsCachedEntryType>
std::unique_ptr<typename OptionsCachedEntryType::MappingOptionsType>
OptionsCache<OptionsCachedEntryType>::retrieveBestOptions(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs,
    const std::string& deviceStr) const {
  auto ret = retrieveTopKOptions(id, inputs, outputs, deviceStr, 1);
  if (ret.empty()) {
    return nullptr;
  }
  return std::unique_ptr<MappingOptionsType>(
      new MappingOptionsType(ret.front()));
}

template <typename OptionsCachedEntryType>
std::vector<typename OptionsCachedEntryType::MappingOptionsType>
OptionsCache<OptionsCachedEntryType>::retrieveTopKOptions(
    const std::string& id,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<const DLTensor*>& outputs,
    const std::string& deviceStr,
    size_t k) const {
  auto candidates =
      detail::searchKernel(this->entries_, id, inputs, outputs, deviceStr);
  std::lock_guard<std::mutex> lock(this->mtx_);
  ++this->numberAttemptedRetrievals;
  if (not candidates) {
    return {};
  }

  struct OptionsWithMedian {
    const MappingOptionsType* options;
    Duration medianRuntime;
  };

  std::vector<OptionsWithMedian> candidatesMedian;
  candidatesMedian.reserve(candidates->values.size());
  std::transform(
      candidates->values.begin(),
      candidates->values.end(),
      std::back_inserter(candidatesMedian),
      [](const typename OptionsCachedEntry<MappingOptionsType>::Values& v) {
        if (v.recordedRuntimes.empty()) {
          throw std::runtime_error(
              "OptionsCache invariant violated: each cached option should "
              "have at least one associated recorded runtime.");
        }
        return OptionsWithMedian{&v.mappingOptions, median(v.recordedRuntimes)};
      });
  std::sort(
      candidatesMedian.begin(),
      candidatesMedian.end(),
      [](const OptionsWithMedian& a, const OptionsWithMedian& b) {
        return a.medianRuntime < b.medianRuntime;
      });
  if (k > candidatesMedian.size()) {
    k = candidatesMedian.size();
  }

  std::vector<MappingOptionsType> res;
  res.reserve(k);
  std::transform(
      candidatesMedian.begin(),
      candidatesMedian.begin() + k,
      std::back_inserter(res),
      [](const OptionsWithMedian& c) { return *c.options; });

  ++this->numberSuccessfulRetrievals;
  return res;
}

template <typename OptionsCachedEntryType>
void OptionsCache<OptionsCachedEntryType>::keepOnlyBestCandidates(
    size_t numberToKeep) {
  std::lock_guard<std::mutex> lock(this->mtx_);

  for (auto& entry : this->entries_) {
    std::sort(
        entry.values.begin(),
        entry.values.end(),
        [](const typename OptionsCachedEntry<MappingOptionsType>::Values& a,
           const typename OptionsCachedEntry<MappingOptionsType>::Values& b) {
          // XXX:this is stupid, medians should be precomputed
          return median(a.recordedRuntimes) < median(b.recordedRuntimes);
        });
    if (entry.values.size() > numberToKeep) {
      entry.values.erase(
          entry.values.begin() + numberToKeep, entry.values.end());
    }
  }
}
} // namespace tc
