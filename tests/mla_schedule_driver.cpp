// CPU-only differential harness. SOURCE_ORACLE never includes or links the migrated scheduler.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

struct Input {
    int64_t batch, sq, heads, sk;
    uint32_t aic, aiv;
    uint64_t l1, ub, l0a, l0b, l0c, libapi;
    uint32_t blockDimOverride;
};

template <class T>
void Emit(T value)
{
    std::cout << ',' << value;
}

template <class T>
void EmitArray(const T& values, size_t capacity)
{
    for (size_t i = 0; i < capacity; ++i) {
        Emit(i < values.size() ? values[i] : 0);
    }
}

#ifdef SOURCE_ORACLE
#include "split_core.h"
#include "log/log.h"

// Minimal data/API shims for executing verbatim pinned host function bodies.
// All omitted features are fixed false; no tensors, gert context, or migrated formulas.
namespace optiling {
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint64_t PRE_LOAD_NUM_MLA = 2;
enum class TilingKeyLayout {
    BSH_BSND,
    TND
};
enum class ScheduleMode {
    NORMAL_MODE = 0,
    BATCH_MODE = 1
};
struct FiaInfo {
    uint32_t bSize, n2Size = 1, s1Size, gSize;
    int64_t s2Size;
    uint32_t qkHeadDim = 512;
    bool pageAttentionFlag = false;
    uint32_t blockSize = 0;
    TilingKeyLayout inputLayout = TilingKeyLayout::BSH_BSND;
    const Input* platformInfo;
};
struct WorkspaceParams {
    uint64_t mm1ResSize = 0, mm2ResSize = 0, fdAccumOutSize = 0, fdLogSumExpSize = 0;
    void set_mm1ResSize(uint64_t x)
    {
        mm1ResSize = x;
    }
    void set_mm2ResSize(uint64_t x)
    {
        mm2ResSize = x;
    }
    void set_fdAccumOutSize(uint64_t x)
    {
        fdAccumOutSize = x;
    }
    void set_fdLogSumExpSize(uint64_t x)
    {
        fdLogSumExpSize = x;
    }
};
struct TilingData {
    WorkspaceParams workspaceParams;
};
namespace platform_ascendc {
struct PlatformAscendC {
    const Input* info;
    explicit PlatformAscendC(const Input* p) : info(p) {}
    uint32_t CalcTschBlockDim(uint32_t slices, uint32_t aic, uint32_t aiv) const
    {
        // Assert the *source* CalcNumBlocks call, then return the supplied platform policy.
        if (slices != aiv || aiv != aic * (info->aiv / info->aic)) {
            std::abort();
        }
        return info->blockDimOverride ? info->blockDimOverride : aic;
    }
};
} // namespace platform_ascendc

class FiaTilingNonQuantMla {
  public:
    FiaInfo* fiaInfo_;
    uint32_t aicNum_, cvRatio_, coreNum_, headDimAlign_ = 512;
    uint64_t libapiSize_, workspaceSize_ = 0;
    uint32_t mBaseSize_ = 256, sInnerSize_ = 0, sInnerSizeAlign_ = 0;
    uint32_t sInnerLoopTimes_ = 0, sInnerSizeTail_ = 0;
    int64_t mm1ResSize_ = 0, mm2ResSize_ = 0;
    bool splitKVFlag_ = false;
    uint32_t numBlocks_ = 0;
    ScheduleMode scheduleMode_ = ScheduleMode::NORMAL_MODE;
    TilingData tilingData_;

    bool DealSameSeqEachBatch() const
    {
        return true;
    }
    void CalcInnerSize(uint32_t seqSize);
    void CalcMBaseSize();
    void CalcMmResSize();
    void FillTilingWorkspaceParams();
    uint64_t CalcNormalWorkspaceSize(uint32_t, int64_t, int64_t, uint32_t) const;
    uint32_t CalcFlashDecodeParamNums(uint32_t) const;
    uint64_t CalcFlashDecodeWorkspace(uint32_t) const;
    void CalcWorkspaceSize();
    void CalcNumBlocks(uint32_t);
    void CalcScheduleMode();
};

// Produced by the pytest fixture exclusively from git show <fixed-commit>:<path>.
#include "source_functions.inc"
} // namespace optiling

void Run(const Input& in)
{
    using namespace optiling;
    FiaInfo info{};
    info.bSize = in.batch;
    info.s1Size = in.sq;
    info.gSize = in.heads;
    info.s2Size = in.sk;
    info.platformInfo = &in;
    FiaTilingNonQuantMla host{};
    host.fiaInfo_ = &info;
    host.aicNum_ = host.coreNum_ = in.aic;
    host.cvRatio_ = in.aiv / in.aic;
    host.libapiSize_ = in.libapi;
    host.CalcInnerSize(in.sk);
    BaseInfo base{};
    base.bSize = in.batch;
    base.s1Size = in.sq;
    base.gSize = in.heads;
    base.s2Size = in.sk;
    base.n2Size = 1;
    SplitParam param{};
    param.mBaseSize = host.mBaseSize_;
    param.s2BaseSize = host.sInnerSize_;
    SplitResult split(in.aic, host.cvRatio_);
    SplitCore(in.aic, base, param, split);
    if (split.numOfFdHead && !std::getenv("MLA_ORACLE_AUTOMATIC_FD")) {
        // Independently map the pinned source's selected partition to complete
        // M blocks. Materialize source row costs instead of the production
        // dense prefix formula, then sum the rows assigned to each core.
        SplitContext context(base, param);
        CalcSplitInfo(context);
        const uint32_t rows = context.splitInfo.s1GBaseNum[0];
        const uint32_t columns = context.splitInfo.s2BaseNum[0];
        std::vector<int64_t> costs;
        for (uint32_t b = 0; b < in.batch; ++b) {
            BatchCache batch{};
            CalcBatchCache(b, context, batch);
            for (uint32_t m = 0; m < rows; ++m) {
                S1GCache row{};
                CalcS1GCache(m, context, batch, row);
                costs.push_back(row.s1GCost);
            }
        }
        std::vector<uint64_t> ends;
        for (uint32_t i = 0; i < split.usedCoreNum; ++i) {
            const uint64_t physical = (uint64_t(split.bN2End[i]) * rows + split.gS1End[i]) * columns + split.s2End[i];
            ends.push_back((physical + columns - 1) / columns);
        }
        ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
        SplitResult whole(in.aic, host.cvRatio_);
        uint64_t start = 0;
        for (auto end : ends) {
            int64_t cost = 0;
            for (auto m = start; m < end; ++m) {
                cost += costs.at(m);
            }
            auto i = whole.usedCoreNum++;
            whole.bN2End[i] = end / rows;
            whole.gS1End[i] = end % rows;
            whole.maxCost = std::max(whole.maxCost, cost);
            start = end;
        }
        split = std::move(whole);
    }
    host.splitKVFlag_ = split.numOfFdHead > 0;
    host.CalcMmResSize();
    host.FillTilingWorkspaceParams();
    host.CalcWorkspaceSize();
    host.CalcNumBlocks(split.usedCoreNum);
    host.CalcScheduleMode();
    auto& w = host.tilingData_.workspaceParams;
    std::cout << "[0";
    for (uint64_t x : {uint64_t(host.mBaseSize_), uint64_t(host.sInnerSize_), uint64_t(split.usedCoreNum),
                       uint64_t(split.maxCost), uint64_t(split.numOfFdHead), uint64_t(split.maxS2SplitNum),
                       uint64_t(split.usedVecNumOfFd), uint64_t(param.gS1BaseSizeOfFd), w.mm1ResSize, w.mm2ResSize,
                       w.fdAccumOutSize, w.fdLogSumExpSize, host.workspaceSize_ - host.libapiSize_, host.libapiSize_,
                       host.workspaceSize_, uint64_t(host.numBlocks_), uint64_t(host.scheduleMode_)}) {
        Emit(x);
    }
    EmitArray(split.bN2End, 26);
    EmitArray(split.gS1End, 26);
    EmitArray(split.s2End, 26);
    EmitArray(split.fdRes.bN2IdxOfFdHead, 26);
    EmitArray(split.fdRes.gS1IdxOfFdHead, 26);
    EmitArray(split.fdRes.s2SplitNumOfFdHead, 26);
    EmitArray(split.fdRes.s2SplitStartIdxOfCore, 26);
    EmitArray(split.fdRes.gS1SplitNumOfFdHead, 26);
    EmitArray(split.fdRes.gS1LastPartSizeOfFdHead, 26);
    EmitArray(split.fdRes.gS1IdxEndOfFdHead, 52);
    EmitArray(split.fdRes.gS1IdxEndOfFdHeadSplit, 52);
    std::cout << "]\n";
}

#else
#include "mla_schedule.h"
#include <cstring>
#include <type_traits>

static_assert(std::is_standard_layout_v<flash_attn::mla::MlaSchedule>);
static_assert(std::is_trivially_copyable_v<flash_attn::mla::MlaSchedule>);
static_assert(sizeof(flash_attn::mla::MlaSchedule::bSize) == 4);
static_assert(sizeof(flash_attn::mla::MlaWorkspace::mm1ResSize) == 8);
static_assert(sizeof(flash_attn::mla::MlaWorkspace::totalBytes) == 8);

void Run(const Input& in)
{
    using namespace flash_attn::mla;
    MlaPlatformInfo platform{};
    platform.aicNum = in.aic;
    platform.aivNum = in.aiv;
    platform.localMemory = {in.l1, in.ub, in.l0a, in.l0b, in.l0c};
    platform.libApiWorkspaceBytes = in.libapi;
    for (uint32_t i = 0; i < kMlaMaxAic; ++i) {
        platform.blockDims[i] = in.blockDimOverride ? in.blockDimOverride : i + 1;
    }
    MlaSchedule schedule;
    std::memset(&schedule, 0xa5, sizeof(schedule));
    std::array<unsigned char, sizeof(schedule)> before;
    std::memcpy(before.data(), &schedule, sizeof(schedule));
    auto status = BuildDenseSchedule({in.batch, in.sq, in.heads, in.sk}, platform, schedule);
    std::cout << '[' << static_cast<uint32_t>(status);
    if (status != MlaScheduleStatus::Success) {
        if (std::memcmp(before.data(), &schedule, sizeof(schedule)) != 0) {
            std::abort(); // A rejected schedule must not modify the caller's previous result.
        }
        std::cout << "]\n";
        return;
    }
    auto& s = schedule;
    auto& w = s.workspace;
    // Bounds derived from segment lengths, independently of the host budget expression.
    uint64_t cursor = 0;
    std::array<uint64_t, 4> offsets{w.qkOffsetBytes, w.pOffsetBytes, w.pvOffsetBytes, w.vec2OffsetBytes};
    std::array<uint64_t, 4> sizes{w.mm1ResSize * 4, w.mm1ResSize * 2, w.mm2ResSize * 4, w.mm2ResSize * 4};
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (offsets[i] != cursor || offsets[i] % 32 != 0) {
            std::abort();
        }
        cursor += sizes[i] * s.blockDim * 2;
    }
    if (cursor != w.kernelBytes || cursor > w.userBytes || s.bSize != in.batch || s.gSize != in.heads ||
        s.s1Size != in.sq || s.s2Size != in.sk || s.n2Size != 1 || s.headDim != 512 || s.headDimRope != 64 ||
        s.aicNum != in.aic || s.aivNum != in.aiv || s.cvRatio != in.aiv / in.aic ||
        s.requiredLocalMemory.l1Bytes != 516096 || s.requiredLocalMemory.l0aBytes != 65536 ||
        s.requiredLocalMemory.l0bBytes != 65536 || s.requiredLocalMemory.l0cBytes != 131072 ||
        s.requiredLocalMemory.ubBytes != (s.cvRatio == 1 ? 178176 : 174080)) {
        std::abort();
    }
    for (uint64_t x : {uint64_t(s.mBaseSize), uint64_t(s.s2BaseSize), uint64_t(s.usedCoreNum), uint64_t(s.maxCost),
                       uint64_t(s.fd.numOfFdHead), uint64_t(s.fd.maxS2SplitNum), uint64_t(s.fd.usedVecNumOfFd),
                       uint64_t(s.fd.gS1BaseSizeOfFd), w.mm1ResSize, w.mm2ResSize, w.fdAccumOutSize, w.fdLogSumExpSize,
                       w.userBytes, w.libApiBytes, w.totalBytes, uint64_t(s.blockDim), uint64_t(s.scheduleMode)}) {
        Emit(x);
    }
    EmitArray(s.bN2End, 26);
    EmitArray(s.gS1End, 26);
    EmitArray(s.s2End, 26);
    EmitArray(s.fd.bN2IdxOfFdHead, 26);
    EmitArray(s.fd.gS1IdxOfFdHead, 26);
    EmitArray(s.fd.s2SplitNumOfFdHead, 26);
    EmitArray(s.fd.s2SplitStartIdxOfCore, 26);
    EmitArray(s.fd.gS1SplitNumOfFdHead, 26);
    EmitArray(s.fd.gS1LastPartSizeOfFdHead, 26);
    EmitArray(s.fd.gS1IdxEndOfFdHead, 52);
    EmitArray(s.fd.gS1IdxEndOfFdHeadSplit, 52);
    std::cout << "]\n";
}
#endif

int main()
{
    Input in{};
    while (std::cin >> in.batch >> in.sq >> in.heads >> in.sk >> in.aic >> in.aiv >> in.l1 >> in.ub >> in.l0a >>
           in.l0b >> in.l0c >> in.libapi >> in.blockDimOverride) {
        Run(in);
    }
    return std::cin.eof() ? 0 : 1;
}
