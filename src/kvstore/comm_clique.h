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
 */
#ifndef MXNET_KVSTORE_COMM_CLIQUE_H_
#define MXNET_KVSTORE_COMM_CLIQUE_H_
#include <dmlc/omp.h>
#include <string>
#include <algorithm>
#include <utility>
#include <limits>
#include <vector>
#include <tuple>
#include <set>
#include <map>
#include "mxnet/ndarray.h"
#include "gradient_compression.h"
#include "../ndarray/ndarray_function.h"
#include "../operator/tensor/sparse_retain-inl.h"
#include "./kvstore_utils.h"
#include "./gpu_topology.h"
namespace mxnet {
namespace kvstore {
/**
 * \brief an implementation of Comm that performs reduction on device
 * directly using tree.
 *
 * It is faster if the total device-to-device bandwidths is larger than
 * device-to-cpu, which is often true for 4 or 8 GPUs. But it uses more device
 * memory.
 */
class CommDeviceClique : public CommDevice {
 public:
  CommDeviceClique(bool is_dist) {
    inited_ = false;
    // gpuarray_bound_ = dmlc::GetEnv("MXNET_KVSTORE_GPUARRAY_BOUND", 10000000);
    is_dist_ = is_dist;
  }

  virtual ~CommDeviceClique() { }

  const NDArray& Reduce(int key, const std::vector<NDArray>& src,
                        int priority) override {
    // when this reduce is called from kvstore_dist, gc is not set
    // we don't do compression twice in dist_sync_device
    if ((gc_ != nullptr) && (gc_->get_type() != CompressionType::kNone)) {
      return ReduceCompressed(key, src, priority);
    }

    // avoid extra copy for single device, but it may bring problems for
    // abnormal usage of kvstore
    if (src.size() == 1) {
      return src[0];
    }

    InitBuffersAndComm(src);
    auto& buf = GetMergeBuf(key);

    const NDArrayStorageType stype = src[0].storage_type();
    NDArray& buf_merged = buf.merged_buf(stype);
    // normal dense reduce
    if (stype == kDefaultStorage) {

      if (buf.copy_buf.empty()) {
        // TODO(mli) this results in large device memory usage for huge ndarray,
        // such as the largest fullc in VGG. consider to do segment reduce with
        // NDArray.Slice or gpu direct memory access. for the latter, we need to
        // remove some ctx check, and also it reduces 20% perf
        buf.copy_buf.resize(src.size() + cliques_.size());
        for (size_t i = 0; i < src.size(); ++i) {
          // assume the is no duplicated devices in the ctx list
          auto buf_ctx = gpu_ctx_[GetConnected(key, buf_merged.ctx().dev_id, dev_clique_[gpus_[src[i].ctx().dev_id]])];
          buf.copy_buf[i] = NDArray(
            buf_merged.shape(), buf_ctx, false, buf_merged.dtype());
        }
        for (size_t i = 0; i < cliques_.size(); ++i) {
          if (is_dist_) {
            buf.copy_buf[src.size()+i] = NDArray(
              buf_merged.shape(), pinned_ctx_, false, buf_merged.dtype());
          }
          else {
            buf.copy_buf[src.size()+i] = NDArray(
              buf_merged.shape(), buf_merged.ctx(), false, buf_merged.dtype());
          }
          buf.copy_buf[src.size()+i] = NDArray(
              buf_merged.shape(), buf_merged.ctx(), false, buf_merged.dtype());
        }
      }
      
      CHECK(buf_merged.ctx().dev_mask() == gpu::kDevMask);

      std::vector< std::vector<NDArray> > sub_reduce(cliques_.size());
      std::vector< std::vector<NDArray> > src_sub_reduce(cliques_.size());
      for (size_t i = 0; i < src.size(); ++i) {
        int idx = dev_clique_[gpus_[src[i].ctx().dev_id]];
        // TODO: initialize once
        src_sub_reduce[idx].push_back(src[i]);
        sub_reduce[idx].push_back(buf.copy_buf[i]);
      }
      for (size_t i = 0; i < cliques_.size(); ++i) {
        for (size_t j = 0; j < sub_reduce[i].size(); ++j) {
          CopyFromTo(src_sub_reduce[i][j], &(sub_reduce[i][j]), priority);
        }
        ElementwiseSum(sub_reduce[i], &sub_reduce[i][0], priority);
      }

      if (cliques_.size() > 1) {
        std::vector<NDArray> reduce(cliques_.size());
        for (size_t i = 0; i < cliques_.size(); ++i) {
          if (sub_reduce[i][0].ctx().dev_id == buf.copy_buf[src.size()+i].ctx().dev_id) {
            reduce[i] = sub_reduce[i][0];
          }
          else {
            CopyFromTo(sub_reduce[i][0], &(buf.copy_buf[src.size()+i]), priority);
            reduce[i] = buf.copy_buf[src.size()+i];
          }
        }
        buf_merged = reduce[0];
        ElementwiseSum(reduce, &(buf_merged), priority);
      }
      else {
        buf_merged = sub_reduce[0][0];
      }
      return buf_merged;
    } else {
      // sparse reduce
      buf_merged = ReduceRowSparse(key, src, priority);
      return buf_merged;
    }
  }

  void Broadcast(int key, const NDArray& src,
                 const std::vector<NDArray*> dst, int priority) override {
    if (dst.size() == 1) {
      CopyFromTo(src, dst[0], priority);
      return;
    }
    if (!inited_) {
      // copy to a random device first
      int dev_id = key % dst.size();
      CopyFromTo(src, dst[dev_id], priority);
      for (size_t i = 0; i < dst.size(); ++i) {
        if (i != static_cast<size_t>(dev_id)) {
          CopyFromTo(*dst[dev_id], dst[i], priority);
        }
      }
    } else {
      NDArray src_tmp;
      if (is_dist_) {
        CHECK(src.ctx().dev_mask() == cpu::kDevMask);
        auto& buf = GetMergeBuf(key);
        auto& buf_merged = buf.merged_buf(src.storage_type());
        CopyFromTo(src, &buf_merged, priority);
        src_tmp = buf_merged;
      }
      else {
        src_tmp = src;                                                                        
      }
      std::vector< std::vector<NDArray*> > sub_broadcast(cliques_.size());
      int src_clique_idx = dev_clique_[gpus_[src_tmp.ctx().dev_id]];
      for (size_t i = 0; i < dst.size(); ++i) {
        int clique_idx = dev_clique_[gpus_[dst[i]->ctx().dev_id]];
        sub_broadcast[clique_idx].push_back(dst[i]);
      }
      for (size_t i = 0; i < cliques_.size(); ++i) {
        if (i != src_clique_idx) {
          size_t connect_id = GetConnected(key, src_tmp.ctx().dev_id, i);
          size_t sub_bcast_id;
          for (size_t j = 0; j < sub_broadcast[i].size(); ++j) {
            if (gpus_[sub_broadcast[i][j]->ctx().dev_id] == connect_id) {
              CopyFromTo(src_tmp, sub_broadcast[i][j], priority);
              sub_bcast_id = j;
              break;
            }
          }
          for (size_t j = 0; j < sub_broadcast[i].size(); ++j) {
            if (sub_bcast_id == j) continue;
            CopyFromTo(*(sub_broadcast[i][sub_bcast_id]), sub_broadcast[i][j], priority);         
          }
        }
        else {
          for (size_t j = 0; j < sub_broadcast[i].size(); ++j) {
            CopyFromTo(src_tmp, sub_broadcast[i][j], priority);
          }
        }
      }
    }
  }

  int KeyBufHash(int key, int size) {
    return key % size;
  }

  int GetConnected(int key, int dev_id, int clique_id) {
    int gpu_idx = gpus_[dev_id];
    if (gpu_clique_connect_[gpu_idx][clique_id] != -1)
      return gpu_clique_connect_[gpu_idx][clique_id];
    const auto& clique = cliques_[clique_id];
    return clique[key % clique.size()];
  }

  // void InitMergeBuffer(const std::vector<Context>& devs) {
  //   std::sort(sorted_key_attrs_.begin(), sorted_key_attrs_.end(), [](
  //             const KeyAttrs& a, const KeyAttrs& b) {
  //     return std::get<1>(a).Size() > std::get<1>(b).Size();
  //   });

  //   for (size_t i = 0; i < sorted_key_attrs_.size(); ++i) {
  //     const int key  = std::get<0>(sorted_key_attrs_[i]);
  //     const TShape& shape = std::get<1>(sorted_key_attrs_[i]);
  //     const int type = std::get<2>(sorted_key_attrs_[i]);
  //     auto& buf = GetMergeBuf(key);
  //     // Delayed allocation - as the dense merged buffer might not be used at all if push()
  //     // only sees sparse arrays
  //     bool delay_alloc = true;
  //     buf.merged = NDArray(shape, pinned_ctx_, delay_alloc, type);
  //   }
  //   inited_ = true;
  // }

 private:

  void BronKerbosch(std::set<int> r, std::set<int> p, std::set<int> x, const std::vector< std::vector<int> >& p2p, const std::set<int>& subgraph, std::vector< std::vector<int> >& cliques) {
    if (p.empty() && x.empty()) {
      // report maximal clique
      std::vector<int> clique;
      for (auto v : r) {
        clique.push_back(v);
      }
      std::sort(clique.begin(), clique.end());
      cliques.push_back(clique);
    }
    else {
      std::set<int> p_cpy = p;
      for (auto v : p_cpy) {
        std::set<int> r_new = r, p_new, x_new;
        r_new.insert(v);
        for (size_t j = 0; j < p2p[v].size(); ++j) {
          if (subgraph.count(j) == 0) continue;
          if (p2p[v][j] == 1) {
            // check all the neighbours of v
            if (p.count(j)) p_new.insert(j);
            if (x.count(j)) x_new.insert(j);
          }
        }
        BronKerbosch(r_new, p_new, x_new, p2p, subgraph, cliques);
        p.erase(v);
        x.insert(v);
      }
    }
  }

  void EnableP2P(const std::vector<Context>& devs) override {
#if MXNET_USE_CUDA
    std::vector<int> gpus;
    for (const auto& d : devs) {
      if (d.dev_mask() == gpu::kDevMask) {
        gpus.push_back(d.dev_id);
        int idx = gpus.size()-1;
        gpus_[d.dev_id] = idx;
        gpu_ctx_[idx] = d;
      }
    }
    int n = static_cast<int>(gpus.size());
    int enabled = 0;
    p2p_.resize( n, std::vector<int>(n, 0) );
    for (int i = 0; i < n; ++i) {
      cudaSetDevice(gpus[i]);
      for (int j = 0; j < n; j++) {
        int access;
        cudaDeviceCanAccessPeer(&access, gpus[i], gpus[j]);
        if (access) {
          cudaError_t e = cudaDeviceEnablePeerAccess(gpus[j], 0);
          if (e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled) {
            ++enabled;
            p2p_[i][j] = 1;
          }
        }
      }
    }
    if (enabled == n * (n-1)) {
      LOG(INFO) << "entire clique";
      std::vector<int> gpu_idx(n);
      int idx = 0;
      std::iota(gpu_idx.begin(), gpu_idx.end(), idx++);
      cliques_.push_back(gpu_idx);
      std::ostringstream ss;
      for (auto v : cliques_[0]) {
        ss << v << ", ";
      }
      LOG(INFO) << ss.str();
    }
    else {
      std::vector<std::vector<int>> cliques;
      // Bron–Kerbosch algorithm
      std::set<int> r, p, x;
      for (int i = 0; i < n; ++i) {
        p.insert(i);
      }
      LOG(INFO) << "BronKerbosch";
      BronKerbosch(r, p, x, p2p_, p, cliques);
      std::vector<int> clique_size(cliques.size());
      std::vector<int> clique_idx(cliques.size());
      for (size_t i = 0; i < cliques.size(); ++i) {
        clique_idx[i] = i;
        clique_size[i] = cliques[i].size();
      }
      std::sort( clique_idx.begin(), clique_idx.end(), [&](int i, int j){return clique_size[i]>clique_size[j];} );
      // greedy
      cliques_.push_back(cliques[clique_idx[0]]);
      std::set<int> v_set;
      for (auto v : cliques_[0]) {
        v_set.insert(v);
      }

      for (size_t i = 1; i < clique_idx.size(); ++i) {

        bool intersection_flag = false;
        int idx = clique_idx[i];
        for (const auto& v : cliques[idx]) {
          if (v_set.count(v)) {
            intersection_flag = true;
            break;
          }
        }

        if (!intersection_flag) {
          cliques_.push_back(cliques[idx]);
          for (const auto& v : cliques[idx]) {
            v_set.insert(v);
          }
        }

        if (v_set.size() == n) break;
      }

      // debug
      for (size_t i = 0; i < cliques_.size(); ++i) {
        std::ostringstream ss;
        for (auto v : cliques_[i]) {
          ss << v << ", ";
        }
        LOG(INFO) << ss.str();
      }
      for (size_t i = 0; i < cliques_.size(); ++i) {
        for (auto v : cliques_[i]) {
          dev_clique_[v] = i;
        }
      }
    }
    gpu_clique_connect_.resize( n, std::vector<int>(cliques_.size(), -1) );
    for (int i = 0; i < n; ++i) {
      for (size_t j = 0; j < cliques_.size(); ++j) {
        for (auto v : cliques_[j]) {
          if (p2p_[i][v] || i == v) {
            gpu_clique_connect_[i][j] = v;
            break;
          }
        }
      }
    }
#endif
  }

  std::vector< std::vector<int> > p2p_;
  // dev_id to index
  std::unordered_map<int, int> gpus_;
  // index to context
  std::unordered_map<int, Context> gpu_ctx_;
  std::vector<std::vector<int>> cliques_;
  // index to clique
  std::unordered_map<int, int> dev_clique_;
  std::vector< std::vector<int> > gpu_clique_connect_;
  int   gpuarray_bound_;
  // flag for distributed kvstore
  bool is_dist_;
};

}  // namespace kvstore
}  // namespace mxnet
#endif  // MXNET_KVSTORE_COMM_CLIQUE_H_
