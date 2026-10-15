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
// and fia_block_vec_nonquant_mla.h. See ../LICENSE.cann.
#pragma once
#include "mla_dense.hpp"
#include "../mla_paged_inputs.h"

namespace flash_attn::mla::Kernel {
// Independent contiguous BSND partial attention. The caller validates the
// M10a fragments and native workspace budget. One fragment per physical AIC;
// no reduction, public accumulator, metadata or production dispatch here.
template <typename InputT>
class MlaSplitKvBsnd : private MlaTaskRing<InputT, true, false, false, false, false, false, false, true, true> {
  public:
    __aicore__ inline void operator()(const MlaDenseKernelSchedule& schedule, const MlaSplitKvKernelParams& partial,
                                      float scale, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr, GM_ADDR workspace,
                                      TPipe* pipe, GM_ADDR lengths)
    {
        this->Run(schedule, scale, qc, qr, kv, kr, workspace, workspace, workspace, pipe, lengths, 0, nullptr, false,
                  nullptr, nullptr, partial.window, &partial);
    }
};
// Production uses allocation/used spans and the existing independent input
// layouts. The launch wrapper supplies one persistent fragment per iteration.
template <typename InputT, bool AbsolutePartialOffsets = false>
class MlaSplitKvUsedBsnd
    : private MlaTaskRing<InputT, true, false, false, true, true, false, false, true, true, AbsolutePartialOffsets> {
  public:
    __aicore__ inline void operator()(const MlaDenseKernelSchedule& s, const MlaSplitKvKernelParams& p, float scale,
                                      GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr, GM_ADDR workspace, TPipe* pipe,
                                      GM_ADDR spans, const MlaBsndInputLayouts& layouts)
    {
        this->Run(s, scale, qc, qr, kv, kr, workspace, workspace, workspace, pipe, spans, 0, &layouts, false, nullptr,
                  nullptr, p.window, &p);
    }
};
// TND allocation starts and independent strides are already carried by v5
// spans. Total Q is only the output head stride; mask uses per-sample used.
template <typename InputT>
class MlaSplitKvUsedTnd : private MlaTaskRing<InputT, true, true, false, true, true, false, false, true, true, true> {
  public:
    __aicore__ inline void operator()(const MlaDenseKernelSchedule& s, const MlaSplitKvKernelParams& p, float scale,
                                      GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr, GM_ADDR workspace, TPipe* pipe,
                                      GM_ADDR spans, const MlaBsndInputLayouts& layouts, uint32_t totalQ,
                                      bool kvIsTnd = true)
    {
        this->Run(s, scale, qc, qr, kv, kr, workspace, workspace, workspace, pipe, spans, totalQ, &layouts, kvIsTnd,
                  nullptr, nullptr, p.window, &p);
    }
};
// PA keeps logical fragment endpoints and resolves physical pages in Copy.
// Absolute offsets own a persistent slot across every task wave.
template <typename InputT, bool IsTnd = false>
class MlaSplitKvPaged : private MlaTaskRing<InputT, true, IsTnd, false, false, true, true, false, true, true, true> {
  public:
    __aicore__ inline void operator()(const MlaDenseKernelSchedule& s, const MlaSplitKvKernelParams& p, float scale,
                                      GM_ADDR qc, GM_ADDR qr, const MlaPagedCacheInputs& cache, GM_ADDR workspace,
                                      TPipe* pipe, GM_ADDR spans, const MlaBsndInputLayouts& layouts,
                                      uint32_t totalQ = 0)
    {
        this->Run(s, scale, qc, qr, reinterpret_cast<GM_ADDR>(cache.v.address),
                  reinterpret_cast<GM_ADDR>(cache.k.address), workspace, workspace, workspace, pipe, spans, totalQ,
                  &layouts, false, reinterpret_cast<GM_ADDR>(cache.table.address), &cache.layout, p.window, &p);
    }
};
template <typename InputT>
using MlaSplitKvPagedBsnd = MlaSplitKvPaged<InputT>;
} // namespace flash_attn::mla::Kernel
