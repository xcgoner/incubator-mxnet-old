/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/**
 * Copyright (c) 2018 by Contributors
 * @file   kvstore_dist_sync_allreduce.h
 * @brief  distributed implementation based on allreduce
 */
#ifndef MXNET_KVSTORE_KVSTORE_DIST_SYNC_ALLREDUCE_H_
#define MXNET_KVSTORE_KVSTORE_DIST_SYNC_ALLREDUCE_H_

#include <mpi.h>
#include <mxnet/kvstore.h>
#include <unordered_map>
#include <bitset>
#include <vector>
#include <string>
#include <utility>
#include <functional>
#include <algorithm>
#include "./comm.h"
#include "./kvstore_utils.h"

#if MXNET_USE_ALLREDUCE_DIST_KVSTORE
#include "collectives/include/collectives.h"

namespace mxnet {
namespace kvstore {

/**
 * \brief store data in local machine
 */
class KVStoreDistSyncAllReduce : public KVStoreLocal {
 public:
  explicit KVStoreDistSyncAllReduce(bool use_device_comm)
     : KVStoreLocal(use_device_comm) {
       LOG(INFO) << "Initializing kvstore";
    int ret = MXCOLLIBInit();
    if (ret != 0) {
      LOG(FATAL) << "kvstore with type [" << type_ << "] failed with collective library init";
    }
    // TODO: bigarray_bound_, naive load balancing
  }

  virtual ~KVStoreDistSyncAllReduce() {
    Engine::Get()->WaitForAll();
    MPI_Barrier(MPI_COMM_WORLD);
  }

  void InitImpl(const std::vector<int>& keys,
                const std::vector<NDArray>& values) override {
    CheckUnique(keys);
    for (size_t i = 0; i < keys.size(); ++i) {
      comm_->Init(keys[i], values[i].storage_type(), values[i].shape(), values[i].dtype());
    }
    if (get_rank() == 0) {
      // Push_(keys, values, 0, false);
      // // wait until the push is finished
      // for (const int key : keys) {
      //   comm_buf_[key].WaitToWrite();
      //   compr_buf_[key].WaitToWrite();
      // }
    } else {
      // do nothing
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  void PushImpl(const std::vector<int>& keys,
                const std::vector<NDArray>& values,
                int priority) override {
    LOG(WARNING) << "Not supported in KVStore with type " << type_ << ".";
  }

  void PullImpl(const std::vector<int>& keys,
                const std::vector<NDArray*>& values,
                int priority) override {
    LOG(WARNING) << "Not supported in KVStore with type " << type_ << ".";
  }

  void PullRowSparseImpl(const std::vector<int>& keys,
                         const std::vector<std::pair<NDArray*, NDArray>>& val_rowids,
                         int priority = 0) override {
    LOG(WARNING) << "Not supported in KVStore with type " << type_ << ".";
  }

  // void Push(const std::vector<int>& keys,
  //           const std::vector<NDArray>& values,
  //           int priority) override {
  //   LOG(WARNING) << "Not supported in KVStore with type " << type_ << ".";
  // }

  // void Pull(const std::vector<int>& keys,
  //           const std::vector<NDArray*>& values,
  //           int priority) override {
  //   LOG(WARNING) << "Not supported in KVStore with type " << type_ << ".";
  // }

  // void PullRowSparse(const std::vector<int>& keys,
  //                    const std::vector<std::pair<NDArray*, NDArray>>& val_rowids,
  //                    int priority = 0) override {
  //   LOG(WARNING) << "Not supported in KVStore with type " << type_ << ".";
  // }

  // void Push(const std::vector<std::string>& str_keys,
  //           const std::vector<NDArray>& values,
  //           int priority) override {
  //   LOG(WARNING) << "Not supported in KVStore with type " << type_ << ".";
  // }

  // void Pull(const std::vector<std::string>& str_keys,
  //           const std::vector<NDArray*>& values,
  //           int priority) override {
  //   LOG(WARNING) << "Not supported in KVStore with type " << type_ << ".";
  // }

  // void PullRowSparse(const std::vector<std::string>& str_keys,
  //                    const std::vector<std::pair<NDArray*, NDArray>>& val_rowids,
  //                    int priority = 0) override {
  //   LOG(WARNING) << "Not supported in KVStore with type " << type_ << ".";
  // }

  // void SetGradientCompression(const std::vector<std::pair<std::string, std::string> >
  //                             & kwargs) override {
  //   LOG(WARNING) << "Not supported in KVStore with type " << type_ << ".";
  // }

  // void PushPull(const std::vector<int> &keys,
  //               const std::vector<NDArray*> &in_values,
  //               const std::vector<NDArray*> &out_values,
  //               int priority) override {
  //   int ret = MXAllReduce(comm_, keys, in_values, out_values, priority);
  //   if (ret != 0) {
  //     LOG(FATAL) << "MXAllReduce is not successful. ret: " << ret;
  //   }
  // }

  // void PushPull(const std::vector<std::string> &str_keys,
  //               const std::vector<NDArray*> &in_values,
  //               const std::vector<NDArray*> &out_values,
  //               int priority) override {
  //   int ret = MXAllReduceEx(comm_, str_keys, in_values, out_values, priority);
  //   if (ret != 0) {
  //     LOG(FATAL) << "MXAllReduceEx is not successful. ret: " << ret;
  //   }
  // }

  // void Broadcast(const std::vector<int> &keys,
  //                const std::vector<NDArray*> &values,
  //                int root_rank,
  //                int priority) override {
  //   int ret = MXBroadcast(comm_, keys, values, root_rank, priority);
  //   if (ret != 0) {
  //     LOG(FATAL) << "MXBroadCast is not successful. ret: " << ret;
  //   }
  // }

  // void Broadcast(const std::vector<std::string> &str_keys,
  //                const std::vector<NDArray*> &values,
  //                int root_rank,
  //                int priority) override {
  //   int ret = MXBroadcastEx(comm_, str_keys, values, root_rank, priority);
  //   if (ret != 0) {
  //     LOG(FATAL) << "MXBroadCastEx is not successful. ret: " << ret;
  //   }
  // }

  int get_rank() const override {
    int ret, rank;
    ret = MXGetMpiRank(&rank);
    if (ret != 0) {
      LOG(FATAL) << "MXGetMpiRank is not successful. ret: " << ret;
      rank = -1;
    }
    return rank;
  }

  int get_group_size() const override {
    int ret, size;
    ret = MXGetMpiSize(&size);
    if (ret != 0) {
      LOG(FATAL) << "MXGetMpiSize is not successful. ret: " << ret;
      size = -1;
    }
    return size;
  }
 private:
  void PushPullImpl(const std::vector<int>& keys,
                const std::vector<NDArray>& invals,
                const std::vector<NDArray*>& outvals,
                int priority) override {
    // first aggregate the values over keys
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray> > grouped_invals;
    GroupKVPairsPush(keys, invals, &uniq_keys, &grouped_invals);
    std::vector<std::vector<NDArray*> > grouped_outvals;
    GroupKVPairsPull(keys, outvals, &uniq_keys, &grouped_outvals);

    size_t idx = 0;

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      // merge over devices
      int key = uniq_keys[i];
      auto& inval = grouped_invals[i];
      NDArray merged = comm_->Reduce(key, inval, priority);

      const auto storage_type = merged.storage_type();
      auto &comm_buf = comm_buf_[key];
      if (merged.ctx().dev_mask() == cpu::kDevMask) {
        // Start of a push doesn't guarantee that the previous pushes are completed.
        // This shouldn't affect training of networks though because training involves
        // a sequence of push, pull, then push. This imposes ordering that the
        // second push happens after the first pull, and the pull happens after first push.
        comm_buf = merged;  // avoid memory copy
      } else {
        if (comm_buf.is_none()) {
          if (storage_type == kDefaultStorage) {
            comm_buf = NDArray(merged.shape(), pinned_ctx_, true, merged.dtype());
          } else {
            comm_buf = NDArray(storage_type, merged.shape(), pinned_ctx_, true, merged.dtype());
          }
        }
        CopyFromTo(merged, &comm_buf);
      }
      // push to servers
      // TODO: compression
      CHECK(storage_type == kDefaultStorage);
      CHECK(gradient_compression_->get_type() == CompressionType::kNone);
      std::string new_key = MXGetMpiKey(keys, key, idx, false);
      idx++;
      MXAllReduce_(new_key, comm_buf, priority);
      comm_->Broadcast(key, comm_buf, grouped_outvals[i], priority);
    }
  }

  void BroadcastImpl(const std::vector<int>& keys,
                const std::vector<NDArray*>& values,
                int root_rank,
                int priority) override {
    // first aggregate the values over keys
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray*> > grouped_vals;
    GroupKVPairsPull(keys, values, &uniq_keys, &grouped_vals);

    size_t idx = 0;

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      // merge over devices
      int key = uniq_keys[i];
      auto& inval = grouped_vals[i];
      NDArray merged = *(inval[0]);

      const auto storage_type = merged.storage_type();
      auto &comm_buf = comm_buf_[key];
      if (merged.ctx().dev_mask() == cpu::kDevMask) {
        // Start of a push doesn't guarantee that the previous pushes are completed.
        // This shouldn't affect training of networks though because training involves
        // a sequence of push, pull, then push. This imposes ordering that the
        // second push happens after the first pull, and the pull happens after first push.
        comm_buf = merged;  // avoid memory copy
      } else {
        if (comm_buf.is_none()) {
          if (storage_type == kDefaultStorage) {
            comm_buf = NDArray(merged.shape(), pinned_ctx_, true, merged.dtype());
          } else {
            comm_buf = NDArray(storage_type, merged.shape(), pinned_ctx_, true, merged.dtype());
          }
        }
        CopyFromTo(merged, &comm_buf);
      }
      // push to servers
      // TODO: compression
      CHECK(storage_type == kDefaultStorage);
      CHECK(gradient_compression_->get_type() == CompressionType::kNone);
      std::string new_key =  MXGetMpiKey(keys, key, idx, true);
      idx++;
      MXBroadcast_(new_key, comm_buf, root_rank, priority);
      comm_->Broadcast(key, comm_buf, grouped_vals[i], priority);
    }
  }

  /**
   * \brief buffer for non-compressed data.
   * When gradient compression is active, this is used
   * for the data in pull and for original data in push
   */
  std::unordered_map<int, NDArray> comm_buf_;

};
}  // namespace kvstore
}  // namespace mxnet

#endif  // MXNET USE ALLREDUCE DIST KVSTORE
#endif  // MXNET_KVSTORE_KVSTORE_DIST_SYNC_ALLREDUCE_H_
