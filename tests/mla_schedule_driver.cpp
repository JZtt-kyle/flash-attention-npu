// CPU-only differential harness. SOURCE_ORACLE never includes or links the migrated scheduler.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <functional>
#include <numeric>
#include <vector>

struct Input {
    int64_t batch, sq, heads, sk;
    uint32_t aic, aiv;
    uint64_t l1, ub, l0a, l0b, l0c, libapi;
    uint32_t blockDimOverride;
#ifdef BSND_KV_LENGTHS
    bool hasLengths = false;
    uint64_t lengthCount = 0;
    std::vector<int32_t> kvLengths;
#endif
#ifdef TND_LENGTHS
    int64_t totalQuery, totalKv;
    bool hasQuery = false;
    uint64_t queryCount = 0;
    std::vector<int32_t> cuQuery;
#endif
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
#ifdef BSND_KV_LENGTHS
struct LengthTensor {
    std::vector<int64_t> values;
    template <typename T>
    const T* GetData() const
    {
        return values.data();
    }
};
struct TensorSlot {
    const LengthTensor* tensor = nullptr;
};
#endif
#ifdef TND_LENGTHS
enum class FiaLayout {
    TND,
    NTD,
    BSND
};
enum class KvStorageMode {
    BATCH_CONTINUOUS,
    PAGE_ATTENTION
};
// Only the state read/written by the pinned ProcessKVActualSeqLoop body.
class FiaInfoParser {
  public:
    bool antiQuantFlag_ = false, needInit_ = false, isSameActualseq_ = true;
    FiaLayout qLayout_ = FiaLayout::TND;
    KvStorageMode kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0, maxActualseq_ = 0;
    void ProcessKVActualSeqLoop(const int64_t*, uint32_t);
};
#include "source_parser.inc"
#endif
struct FiaInfo {
    uint32_t bSize, n2Size = 1, s1Size, gSize;
    int64_t s2Size;
    uint32_t qkHeadDim = 512;
    bool pageAttentionFlag = false;
    uint32_t blockSize = 0;
    TilingKeyLayout inputLayout = TilingKeyLayout::BSH_BSND;
    const Input* platformInfo;
#ifdef BSND_KV_LENGTHS
    bool batchContinuousFlag = true, actualSeqLenFlag = false;
    bool isSameActualseq = true, isSameSeqAllKVTensor = true;
    bool isAccumQSeq = false, isAccumKVSeq = false, attenMaskFlag = false;
    uint32_t actualLenQDims = 0, actualLenKvDims = 0;
    int64_t preToken = 0, nextToken = 0;
    int32_t sparseMode = 0;
    struct {
        TensorSlot actualSeqLengthsQ, actualSeqLengths;
    } opParamInfo;
#endif
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

#ifdef BSND_KV_LENGTHS
    uint32_t mFdBaseSize_ = 8;
    bool DealSameSeqEachBatch() const;
    void CreateSplitInput(BaseInfo&, SplitParam&) const;
#else
    bool DealSameSeqEachBatch() const
    {
        return true;
    }
#endif
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
#ifdef BSND_KV_LENGTHS
    // Oracle input shim owns true int64 storage, as required by the unmodified
    // pinned CreateSplitInput. It does not share the migrated length parser.
    LengthTensor tensor{};
#ifdef TND_LENGTHS
    LengthTensor query{};
    // The oracle owns int64 *tails*, independently of ParseTndLengths.
    // GetS1/GetS2SeqSize below are the original cumulative-length functions.
    query.values.assign(in.cuQuery.begin() + 1, in.cuQuery.end());
#ifdef TND_BSND_KV
    // TND Q tails and BSND KV lengths have different accumulation semantics.
    tensor.values.assign(in.kvLengths.begin(), in.kvLengths.end());
#else
    tensor.values.assign(in.kvLengths.begin() + 1, in.kvLengths.end());
#endif
    FiaInfoParser parser;
    parser.ProcessKVActualSeqLoop(tensor.values.data(), in.batch);
    info.inputLayout = TilingKeyLayout::TND;
    info.isAccumQSeq = info.isAccumKVSeq = true;
#ifdef TND_BSND_KV
    info.isAccumKVSeq = false;
#endif
    info.actualSeqLenFlag = true;
    info.actualLenQDims = info.actualLenKvDims = in.batch;
    info.opParamInfo.actualSeqLengthsQ.tensor = &query;
    info.opParamInfo.actualSeqLengths.tensor = &tensor;
    info.isSameActualseq = parser.isSameActualseq_;
#else
    if (in.hasLengths) {
        tensor.values.assign(in.kvLengths.begin(), in.kvLengths.end());
        info.actualSeqLenFlag = true;
        info.actualLenKvDims = tensor.values.size();
        info.opParamInfo.actualSeqLengths.tensor = &tensor;
        info.isSameActualseq = std::adjacent_find(tensor.values.begin(), tensor.values.end(),
                                                  std::not_equal_to<int64_t>()) == tensor.values.end();
    }
#endif
#endif
    FiaTilingNonQuantMla host{};
    host.fiaInfo_ = &info;
    host.aicNum_ = host.coreNum_ = in.aic;
    host.cvRatio_ = in.aiv / in.aic;
    host.libapiSize_ = in.libapi;
    BaseInfo base{};
#ifdef TND_LENGTHS
    SplitParam initial{};
    host.CreateSplitInput(base, initial);
    info.s1Size = 0;
#ifndef TND_BSND_KV
    info.s2Size = 0;
#endif
    for (uint32_t b = 0; b < info.bSize; ++b) {
        info.s1Size = std::max(info.s1Size, GetS1SeqSize(b, base));
#ifndef TND_BSND_KV
        info.s2Size = std::max(info.s2Size, int64_t(GetS2SeqSize(b, base)));
#endif
    }
    base = {};
#endif
    host.CalcInnerSize(info.s2Size);
#ifndef BSND_KV_LENGTHS
    base.bSize = in.batch;
    base.s1Size = in.sq;
    base.gSize = in.heads;
    base.s2Size = in.sk;
    base.n2Size = 1;
#endif
    SplitParam param{};
    param.mBaseSize = host.mBaseSize_;
    param.s2BaseSize = host.sInnerSize_;
#ifdef BSND_KV_LENGTHS
    host.CreateSplitInput(base, param);
#ifndef TND_LENGTHS
    for (uint32_t b = 0; b < info.bSize; ++b) {
        if (GetS1SeqSize(b, base) != in.sq || GetS2SeqSize(b, base) != (in.hasLengths ? in.kvLengths.at(b) : in.sk)) {
            std::abort();
        }
    }
#endif
#endif
    SplitResult split(in.aic, host.cvRatio_);
    SplitCore(in.aic, base, param, split);
    if (split.numOfFdHead && !std::getenv("MLA_ORACLE_AUTOMATIC_FD")) {
        // Independently map the pinned source's selected partition to complete
        // M blocks. Materialize source row costs instead of the production
        // dense prefix formula, then sum the rows assigned to each core.
        SplitContext context(base, param);
        CalcSplitInfo(context);
#ifndef TND_LENGTHS
        const uint32_t rows = context.splitInfo.s1GBaseNum[0];
#endif
#ifndef BSND_KV_LENGTHS
        const uint32_t columns = context.splitInfo.s2BaseNum[0];
#endif
        std::vector<int64_t> costs;
#ifdef TND_LENGTHS
        // Explicit row coordinates/costs, unlike the migrated batch-prefix formula.
        std::vector<std::pair<uint32_t, uint32_t>> positions;
#endif
        for (uint32_t b = 0; b < in.batch; ++b) {
            BatchCache batch{};
            CalcBatchCache(b, context, batch);
#ifdef TND_LENGTHS
            const uint32_t rows = context.splitInfo.s1GBaseNum[b];
#endif
            for (uint32_t m = 0; m < rows; ++m) {
                S1GCache row{};
                CalcS1GCache(m, context, batch, row);
                costs.push_back(row.s1GCost);
#ifdef TND_LENGTHS
                positions.emplace_back(b, m);
#endif
            }
        }
        std::vector<uint64_t> ends;
#ifdef TND_LENGTHS
        std::vector<std::pair<uint32_t, uint32_t>> boundaries;
#endif
        for (uint32_t i = 0; i < split.usedCoreNum; ++i) {
#ifdef TND_LENGTHS
            auto boundary = std::make_pair(split.bN2End[i], split.gS1End[i]);
            if (split.s2End[i] != 0) {
                ++boundary.second;
            }
            if (boundary.first < in.batch && boundary.second != 0 &&
                boundary.second == context.splitInfo.s1GBaseNum[boundary.first]) {
                ++boundary.first;
                boundary.second = 0;
            }
            const uint64_t end = std::lower_bound(positions.begin(), positions.end(), boundary) - positions.begin();
            if (ends.empty() || ends.back() != end) {
                ends.push_back(end);
                boundaries.push_back(boundary);
            }
#elif defined(BSND_KV_LENGTHS)
            uint64_t end = uint64_t(split.bN2End[i]) * rows + split.gS1End[i];
            if (split.s2End[i] > 0) {
                ++end;
            }
            ends.push_back(end);
#else
            const uint64_t physical = (uint64_t(split.bN2End[i]) * rows + split.gS1End[i]) * columns + split.s2End[i];
            ends.push_back((physical + columns - 1) / columns);
#endif
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
#ifdef TND_LENGTHS
            // Preserve the source boundary across empty-query samples. Several
            // batch coordinates can have the same flat row offset in TND.
            auto boundary = boundaries.at(i);
            whole.bN2End[i] = boundary.first;
            whole.gS1End[i] = boundary.second;
#else
            whole.bN2End[i] = end / rows;
            whole.gS1End[i] = end % rows;
#endif
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
#ifdef TND_LENGTHS
    Emit(info.s1Size);
#ifdef TND_BSND_KV
    Emit(*std::max_element(tensor.values.begin(), tensor.values.end()));
#else
    Emit(info.s2Size);
#endif
    for (uint32_t b = 0; b < info.bSize; ++b) {
        Emit(b == 0 ? 0 : query.values[b - 1]);
#ifdef TND_BSND_KV
        Emit(uint64_t(b) * in.sk);
#else
        Emit(b == 0 ? 0 : tensor.values[b - 1]);
#endif
        Emit(GetS1SeqSize(b, base));
        Emit(GetS2SeqSize(b, base));
    }
#endif
    std::cout << "]\n";
}

#else
#include "mla_schedule.h"
#ifdef BSND_KV_LENGTHS
#include "mla_bsnd_schedule.h"
#endif
#ifdef TND_LENGTHS
#include "mla_tnd_schedule.h"
#ifdef TND_BSND_KV
#include "mla_tnd_bsnd_schedule.h"
#endif
#endif
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
#ifdef TND_LENGTHS
#ifdef TND_BSND_KV
    MlaTndBsndSchedule tnd;
#else
    MlaTndSchedule tnd;
#endif
    tnd.lengths.maxQueryLength = 99;
    tnd.lengths.maxKvLength = 77;
    tnd.lengths.sequences.push_back({11, 22, 33, 44});
    const auto beforeLengths = tnd.lengths;
    auto& schedule = tnd.schedule;
#else
    MlaSchedule schedule;
#endif
    std::memset(&schedule, 0xa5, sizeof(schedule));
    std::array<unsigned char, sizeof(schedule)> before;
    std::memcpy(before.data(), &schedule, sizeof(schedule));
#ifdef TND_LENGTHS
    const int32_t sentinel = 1234;
    const int32_t* cq = in.hasQuery ? (in.cuQuery.empty() ? &sentinel : in.cuQuery.data()) : nullptr;
    const int32_t* ck = in.hasLengths ? (in.kvLengths.empty() ? &sentinel : in.kvLengths.data()) : nullptr;
    const bool parseOnly = std::getenv("MLA_TND_PARSE_ONLY") != nullptr;
#ifdef TND_BSND_KV
    MlaTndBsndShape shape{in.batch, in.totalQuery, in.heads, in.sq, in.sk};
    auto status = parseOnly ? ParseTndBsndLengths(shape, cq, in.queryCount, ck, in.lengthCount, tnd.lengths)
                            : BuildTndBsndSchedule(shape, platform, cq, in.queryCount, ck, in.lengthCount, tnd);
#else
    MlaTndShape shape{in.batch, in.totalQuery, in.totalKv, in.heads, in.sq, in.sk};
    auto status = parseOnly ? ParseTndLengths(shape, cq, in.queryCount, ck, in.lengthCount, tnd.lengths)
                            : BuildTndSchedule(shape, platform, cq, in.queryCount, ck, in.lengthCount, tnd);
#endif
    if (status != MlaScheduleStatus::Success &&
        (tnd.lengths.maxQueryLength != beforeLengths.maxQueryLength ||
         tnd.lengths.maxKvLength != beforeLengths.maxKvLength ||
         tnd.lengths.sequences.size() != beforeLengths.sequences.size() || tnd.lengths.sequences[0].queryStart != 11 ||
         tnd.lengths.sequences[0].kvStart != 22 || tnd.lengths.sequences[0].queryLength != 33 ||
         tnd.lengths.sequences[0].kvLength != 44)) {
        std::abort();
    }
#elif defined(BSND_KV_LENGTHS)
    const int32_t emptyLength = 0;
    const int32_t* lengths = in.hasLengths ? (in.kvLengths.empty() ? &emptyLength : in.kvLengths.data()) : nullptr;
    auto status = std::getenv("MLA_BSND_USE_DENSE")
                      ? BuildDenseSchedule({in.batch, in.sq, in.heads, in.sk}, platform, schedule)
                      : BuildBsndKvLengthSchedule({in.batch, in.sq, in.heads, in.sk}, platform, lengths, in.lengthCount,
                                                  schedule);
#else
    auto status = BuildDenseSchedule({in.batch, in.sq, in.heads, in.sk}, platform, schedule);
#endif
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
#ifdef TND_LENGTHS
    const auto emitLengths = [&]() {
        Emit(tnd.lengths.maxQueryLength);
        Emit(tnd.lengths.maxKvLength);
        for (const auto& seq : tnd.lengths.sequences) {
            Emit(seq.queryStart);
            Emit(seq.kvStart);
            Emit(seq.queryLength);
            Emit(seq.kvLength);
        }
    };
    if (parseOnly) {
        if (std::memcmp(before.data(), &schedule, sizeof(schedule)) != 0) {
            std::abort();
        }
        emitLengths();
        std::cout << "]\n";
        return;
    }
    const uint32_t sq = tnd.lengths.maxQueryLength;
#ifdef TND_BSND_KV
    const uint32_t sk = in.sk;
#else
    const uint32_t sk = tnd.lengths.maxKvLength;
#endif
#else
    const int64_t sq = in.sq, sk = in.sk;
#endif
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
    const bool empty = in.batch == 0 || sq == 0 || sk == 0;
    if (cursor != w.kernelBytes || cursor > w.userBytes || s.bSize != in.batch || s.gSize != in.heads ||
        s.s1Size != sq || s.s2Size != sk || s.n2Size != 1 || s.headDim != 512 || s.headDimRope != 64 ||
        s.aicNum != in.aic || s.aivNum != in.aiv || s.cvRatio != in.aiv / in.aic ||
        (!empty && (s.requiredLocalMemory.l1Bytes != 516096 || s.requiredLocalMemory.l0aBytes != 65536 ||
                    s.requiredLocalMemory.l0bBytes != 65536 || s.requiredLocalMemory.l0cBytes != 131072 ||
                    s.requiredLocalMemory.ubBytes != (s.cvRatio == 1 ? 178176 : 174080)))) {
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
#ifdef TND_LENGTHS
    emitLengths();
#endif
    std::cout << "]\n";
}
#endif

int main()
{
    Input in{};
    while (std::cin >> in.batch >> in.sq >> in.heads >> in.sk >> in.aic >> in.aiv >> in.l1 >> in.ub >> in.l0a >>
           in.l0b >> in.l0c >> in.libapi >> in.blockDimOverride) {
#ifdef BSND_KV_LENGTHS
        size_t supplied = 0;
        if (!(std::cin >> in.hasLengths >> in.lengthCount >> supplied)) {
            return 1;
        }
        in.kvLengths = std::vector<int32_t>(supplied);
        for (auto& length : in.kvLengths) {
            if (!(std::cin >> length)) {
                return 1;
            }
        }
#endif
#ifdef TND_LENGTHS
        size_t querySupplied = 0;
        if (!(std::cin >> in.totalQuery >> in.totalKv >> in.hasQuery >> in.queryCount >> querySupplied)) {
            return 1;
        }
        in.cuQuery = std::vector<int32_t>(querySupplied);
        for (auto& value : in.cuQuery) {
            if (!(std::cin >> value)) {
                return 1;
            }
        }
#endif
        Run(in);
    }
    return std::cin.eof() ? 0 : 1;
}
