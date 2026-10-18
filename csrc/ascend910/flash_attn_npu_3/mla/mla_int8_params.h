/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Source: CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0.
// attention/incre_flash_attention/op_host/incre_flash_attention_tiling_mla.h; license: LICENSE.cann.
// Only INT8 content / BF16 RoPE and output, Hkv=1, 512+64, ND PA.
#pragma once
#include <cstdint>

namespace flash_attn::mla::Int8 {
constexpr uint32_t MAX_AIC_CORE_NUM = 26;
struct alignas(8) IncreFlashAttentionBaseParamsMla {
    uint32_t batchSize;
    uint32_t seqSize;
    uint32_t qSeqSize;
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
    float scaleValue;
    uint32_t nNumOfQInOneGroup;
    uint32_t actualLenQDims;
    uint32_t actualLenDims;
    uint32_t antiquantMode;
    uint32_t attenMaskFlag;
    uint32_t attenMaskSize;
    uint32_t sparseMode;
    uint32_t outputLayout;
    uint32_t softmaxLseFlag;
    uint32_t tSeqSize;
};
struct alignas(8) IncreFlashAttentionCoreParamsMla {
    uint32_t coreBEnd[MAX_AIC_CORE_NUM];
    uint32_t coreSidxEnd[MAX_AIC_CORE_NUM];
    uint32_t coreS1OuterEnd[MAX_AIC_CORE_NUM];
    uint32_t coreS2End[MAX_AIC_CORE_NUM];
};
struct alignas(8) IncreFlashAttentionSingleCoreParamsMla {
    uint32_t singleProcessSInnerSize;
    uint32_t usedCoreNum;
    uint32_t groupSplitSize;
    uint32_t s1SplitSize;
};
struct alignas(8) IncreFlashAttentionSingleCoreTensorSizeMla {
    uint32_t mmResUbSize;
    uint32_t bmm2ResUbSize;
};
struct alignas(8) IncreFlashAttentionSplitKVParamsMla {
    uint32_t s2;
    uint32_t sInnerLoopSize;
    uint32_t accumOutSize;
    uint32_t logSumExpSize;
};
struct alignas(8) IncreFlashAttentionTNDSplitCoreParamsMla {
    uint32_t tndFDCoreArrLen;
    uint32_t reserve;
    uint32_t balanceFDCoreBArr[MAX_AIC_CORE_NUM];
    uint32_t balanceFDCoreS1Arr[MAX_AIC_CORE_NUM];
    uint32_t balanceFDCoreKVSplitArr[MAX_AIC_CORE_NUM];
    uint32_t balanceFDCoreStartKVSplitNum[MAX_AIC_CORE_NUM];
    uint32_t gS1IdxEndOfFdHead[MAX_AIC_CORE_NUM * 2];
    uint32_t gS1IdxEndOfFdHeadSplit[MAX_AIC_CORE_NUM * 2];
    uint32_t usedVecNumOfFd;
    uint32_t FdBalanceFlag;
    uint32_t gS1SplitNumOfFdHeadMla[MAX_AIC_CORE_NUM * 2];
    uint32_t gS1LastPartSizeOfFdHead[MAX_AIC_CORE_NUM * 2];
};
struct alignas(8) Int8TilingParams {
    IncreFlashAttentionBaseParamsMla baseParams;
    IncreFlashAttentionSplitKVParamsMla splitKVParams;
    IncreFlashAttentionTNDSplitCoreParamsMla tndSplitCoreParams;
    IncreFlashAttentionCoreParamsMla increFlashAttentionCoreParams;
    IncreFlashAttentionSingleCoreParamsMla increFlashAttentionSingleCoreParams;
    IncreFlashAttentionSingleCoreTensorSizeMla increFlashAttentionSingleCoreTensorSize;
};
} // namespace flash_attn::mla::Int8
