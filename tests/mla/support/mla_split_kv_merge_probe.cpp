// Independent pinned FD Block oracle and migrated Vector merge. Source header
// bytes and arithmetic are unmodified. The harness supplies canonical empty
// partials, BSND output coordinates and a single FD task; no attention rerun.
#include "kernel_operator.h"
#include "acl/acl.h"
#include "runtime/rt.h"
#include "mla/mla_split_kv_merge_params.h"
#include <cmath>
using namespace AscendC;
using flash_attn::mla::MlaSplitKvMergeParams;
#ifdef MLA_PREFILL
constexpr uint32_t OutputWidth = 128, MaxMBase = 512;
#else
constexpr uint32_t OutputWidth = 512, MaxMBase = 256;
#endif
static_assert(sizeof(MlaSplitKvMergeParams) + 7 * sizeof(void*) + sizeof(float) <= 1536);
#ifdef MLA_DENSE_BF16
using OutputT = bfloat16_t;
#else
using OutputT = half;
#endif
#include "mla_probe_batch_launch.h"
#ifdef MLA_DENSE_ORACLE
#define MERGE_ENTRY cann_split_kv_merge
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "oracle_tiling.h"
#include "fia_public_define.h"
using fa_base_vector::FLOAT_ZERO;
constexpr float FLOAT_INF = AttentionCommon::ConstInfo::FLOAT_INF;
constexpr uint32_t BUFFER_SIZE_BYTE_1K = AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_1K;
#include "fia_kernel_nonquant_mla.h"
#endif
#else
#define MERGE_ENTRY mla_split_kv_merge
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "mla/kernel/mla_split_kv_merge.hpp"
#endif
#endif

__global__ __aicore__ __mix__(1, 2) void MERGE_ENTRY(GM_ADDR partial, GM_ADDR sum, GM_ADDR maximum, GM_ADDR output,
                                                     GM_ADDR lse, GM_ADDR accum, GM_ADDR accumLse,
                                                     MlaSplitKvMergeParams p, float softcap)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    if ASCEND_IS_AIV {
        TPipe pipe;
#ifdef MLA_DENSE_ORACLE
        if (GetBlockIdx() != 0) {
            return;
        }
        using Types = AttentionCommon::FIAType<OutputT, OutputT, OutputT, OutputT, false, true>;
        FiaBlockVecFlashDecode<Types> op;
        AttentionCommon::ConstInfo info{};
        info.batchSize = p.batch;
        info.qSeqSize = p.sq;
        info.kvSeqSize = p.kvLength;
        info.gSize = info.qHeadNum = p.heads;
        info.kvHeadNum = 1;
        info.headDim = info.headDimAlign = OutputWidth;
        info.mBaseSize = p.mBase;
        info.batchContinuous = true;
        info.softmaxLseFlag = true;
        info.outputLayout = AttentionCommon::FIA_LAYOUT::BSH;
        info.preLoadNum = Min(2U, p.splits);
        op.InitParams(info);
        GlobalTensor<float> o, s, m, l;
        GlobalTensor<OutputT> dst;
        GlobalTensor<uint64_t> empty;
        o.SetGlobalBuffer((__gm__ float*)partial);
        s.SetGlobalBuffer((__gm__ float*)sum);
        m.SetGlobalBuffer((__gm__ float*)maximum);
        l.SetGlobalBuffer((__gm__ float*)lse);
        dst.SetGlobalBuffer((__gm__ OutputT*)output);
        op.InitGlobalTensor(m, s, o, dst, empty, empty, nullptr, nullptr, nullptr);
        op.InitSoftmaxLseGm(l);
        op.InitDecodeParams();
        op.InitBuffers(&pipe);
        const uint32_t tile = Max(1U, Min(8U, 192U / p.splits));
        uint32_t batch = p.batchIndex, mBlock = p.rowStart / p.mBase, splits = p.splits;
        uint32_t blocks = (p.rows + tile - 1) / tile, tail = p.rows - (blocks - 1) * tile;
        uint32_t endTask = 0, endBlock = blocks - 1;
        AttentionCommon::FDparams fd{&batch, &mBlock, &splits, &blocks, &tail, &endTask, &endBlock, 1, tile};
        op.AllocEventID();
        op.FlashDecode(fd);
        op.FreeEventID();
#else
        flash_attn::mla::Kernel::MlaSplitKvMerge<OutputT, OutputWidth> op;
        op(p, partial, sum, maximum, output, lse, accum, accumLse, &pipe, softcap);
#endif
    }
#endif
}

extern "C" int launch_merge_softcap(void* stream, void* partial, void* sum, void* maximum, void* output, void* lse,
                                    void* accum, void* accumLse, const MlaSplitKvMergeParams* p, float softcap)
{
    // Validated dimensions keep every output/accumulator byte offset in int64.
    // Guard allocation sizes remain the owner's responsibility for this probe.
    if (!std::isfinite(softcap) || softcap < 0 || !p || !partial || !sum || !maximum || !output || !lse ||
        p->batch == 0 || p->batch > 65536 || p->sq == 0 || p->sq > 1048576 || p->heads == 0 || p->heads > 65536 ||
        p->batchIndex >= p->batch || p->mBase < 1 || p->mBase > MaxMBase || p->rows == 0 || p->rows > p->mBase ||
        p->rowStart % p->mBase != 0 || p->splits == 0 || p->splits > 256 || p->kvLength > 16384 || p->maskKind > 2 ||
        p->adapt > 1 ||
        // M10a's dense row bound keeps source S1G and ceil-div fields uint32.
        uint64_t(p->sq) * p->heads > UINT32_MAX - 255 || uint64_t(p->rowStart) + p->rows > uint64_t(p->sq) * p->heads ||
        uint64_t(p->batch) * p->sq * p->heads > INT64_MAX / (uint64_t(OutputWidth) * 4 * p->splits) ||
        (p->maskKind == 2 && (p->left < 0 || p->right < 0 || p->left > INT32_MAX || p->right > INT32_MAX)) ||
        (p->adapt && p->splits > 1 && (!accum || !accumLse))) {
        return -1;
    }
    for (uint32_t i = 0; i < p->splits; ++i) {
        if (p->begin[i] > p->end[i] || p->end[i] > p->kvLength) {
            return -1;
        }
    }
#ifdef MLA_DENSE_ORACLE
    if (softcap != 0.0f)
        return -1;
    // Native 6 KiB statistics buffers cannot represent 256 splits even with
    // one row, and original vector repeats are uint8. Never overflow them.
    if (p->splits > 192 || p->adapt) {
        return -2;
    }
#endif
    auto error = aclrtSetOpExecuteTimeOut(15);
    if (error != ACL_SUCCESS) {
        return error;
    }
#if !defined(__NPU_ARCH__)
    auto before = batchLaunchCount;
#endif
    MERGE_ENTRY<<<1, nullptr, stream>>>((uint8_t*)partial, (uint8_t*)sum, (uint8_t*)maximum, (uint8_t*)output,
                                        (uint8_t*)lse, (uint8_t*)accum, (uint8_t*)accumLse, *p, softcap);
#if !defined(__NPU_ARCH__)
    return batchLaunchCount == before + 1 ? batchLaunchError : -3;
#else
    return 0;
#endif
}

extern "C" int launch_merge(void* stream, void* partial, void* sum, void* maximum, void* output, void* lse, void* accum,
                            void* accumLse, const MlaSplitKvMergeParams* p)
{
    return launch_merge_softcap(stream, partial, sum, maximum, output, lse, accum, accumLse, p, 0.0f);
}
