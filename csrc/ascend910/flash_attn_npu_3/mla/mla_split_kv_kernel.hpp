/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Source partial orchestration and FD writeback: CANN ops-transformer
// ca18aee610729ca5270a44e2d84c751bc687fbf0, fia_kernel_nonquant_mla.h
// and fia_block_vec_nonquant_mla.h. See LICENSE.cann.
#pragma once
#include "mla_kernel.hpp"
#include "mla_metadata.h"
namespace flash_attn::mla::Kernel {
// Production uses allocation/used spans and the existing independent input
// layouts. The launch wrapper supplies one persistent fragment per iteration.
template <typename InputT, bool AbsolutePartialOffsets = false>
class MlaSplitKvUsedBsnd
    : private MlaTaskRing<InputT, /*HasKvLengths=*/true, /*IsTnd=*/false, /*MixedKv=*/false, /*HasUsed=*/true,
                          /*HasStrides=*/true, /*IsPaged=*/false, /*IsCausal=*/false, /*IsWindow=*/true,
                          /*IsPartial=*/true, AbsolutePartialOffsets> {
  public:
    __aicore__ inline void operator()(const MlaDenseKernelSchedule& s, const MlaSplitKvKernelParams& p, float scale,
                                      GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr, GM_ADDR workspace, TPipe* pipe,
                                      GM_ADDR spans, const MlaBsndInputLayouts& layouts, float softcap = 0.0f)
    {
        this->Run(s, scale, qc, qr, kv, kr, workspace, workspace, workspace, pipe, spans, 0, &layouts, false, nullptr,
                  nullptr, p.window, &p, softcap);
    }
};
// TND allocation starts and independent strides are already carried by v5
// spans. Total Q is only the output head stride; mask uses per-sample used.
template <typename InputT>
class MlaSplitKvUsedTnd
    : private MlaTaskRing<InputT, /*HasKvLengths=*/true, /*IsTnd=*/true, /*MixedKv=*/false, /*HasUsed=*/true,
                          /*HasStrides=*/true, /*IsPaged=*/false, /*IsCausal=*/false, /*IsWindow=*/true,
                          /*IsPartial=*/true, /*AbsolutePartialOffsets=*/true> {
  public:
    __aicore__ inline void operator()(const MlaDenseKernelSchedule& s, const MlaSplitKvKernelParams& p, float scale,
                                      GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr, GM_ADDR workspace, TPipe* pipe,
                                      GM_ADDR spans, const MlaBsndInputLayouts& layouts, uint32_t totalQ,
                                      bool kvIsTnd = true, float softcap = 0.0f)
    {
        this->Run(s, scale, qc, qr, kv, kr, workspace, workspace, workspace, pipe, spans, totalQ, &layouts, kvIsTnd,
                  nullptr, nullptr, p.window, &p, softcap);
    }
};
// PA keeps logical fragment endpoints and resolves physical pages in Copy.
// Absolute offsets own a persistent slot across every task wave.
template <typename InputT, bool IsTnd = false>
class MlaSplitKvPaged : private MlaTaskRing<InputT, /*HasKvLengths=*/true, IsTnd, /*MixedKv=*/false, /*HasUsed=*/false,
                                            /*HasStrides=*/true, /*IsPaged=*/true, /*IsCausal=*/false,
                                            /*IsWindow=*/true, /*IsPartial=*/true, /*AbsolutePartialOffsets=*/true> {
  public:
    __aicore__ inline void operator()(const MlaDenseKernelSchedule& s, const MlaSplitKvKernelParams& p, float scale,
                                      GM_ADDR qc, GM_ADDR qr, const MlaPagedCacheInputs& cache, GM_ADDR workspace,
                                      TPipe* pipe, GM_ADDR spans, const MlaBsndInputLayouts& layouts,
                                      uint32_t totalQ = 0, float softcap = 0.0f)
    {
        this->Run(s, scale, qc, qr, reinterpret_cast<GM_ADDR>(cache.v.address),
                  reinterpret_cast<GM_ADDR>(cache.k.address), workspace, workspace, workspace, pipe, spans, totalQ,
                  &layouts, false, reinterpret_cast<GM_ADDR>(cache.table.address), &cache.layout, p.window, &p,
                  softcap);
    }
};

// GQA changes validated head/length coordinates only. The existing partial
// task ring, Cube/Vector arithmetic, PA Copy and event ordering are reused.
template <typename InputT, bool IsTnd, bool Paged>
class MlaGqaPartial : private MlaTaskRing<InputT, true, IsTnd, false, !Paged, true, Paged, false, true, true, true> {
  public:
    __aicore__ inline void RunFragment(const MlaDenseKernelSchedule& s, const MlaSplitKvKernelParams& p,
                                       const MlaTaskLengths& lengths, float scale, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv,
                                       GM_ADDR kr, GM_ADDR workspace, TPipe* pipe, const MlaBsndInputLayouts& layouts,
                                       bool kvIsTnd, const MlaPagedCacheInputs& cache, float softcap)
    {
        this->Run(s, scale, qc, qr, kv, kr, workspace, workspace, workspace, pipe, nullptr, s.sq, &layouts, kvIsTnd,
                  reinterpret_cast<GM_ADDR>(cache.table.address), &cache.layout, p.window, &p, softcap, &lengths);
    }
};
} // namespace flash_attn::mla::Kernel
