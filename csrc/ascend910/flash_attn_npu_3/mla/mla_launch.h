#pragma once

#include <cstdint>
#include "acl/acl.h"

namespace flash_attn::mla {

// Static dense schedule; M05 will add the device-side metadata protocol.
// No device lengths, packed inputs, or host-to-device schedule copy are needed.
constexpr uint32_t kHeadsPerTile = 16;
constexpr uint32_t kShortKvLimit = 128;
constexpr uint32_t kOnlineKvLimit = 1280;
constexpr uint32_t kScoreBytes = kHeadsPerTile * kShortKvLimit * sizeof(float);
constexpr uint32_t kProbabilityBytes = kHeadsPerTile * kShortKvLimit * 2;
constexpr uint32_t kOutputBytes = kHeadsPerTile * 512 * sizeof(float);
constexpr uint32_t kWorkspaceBytesPerCore = kScoreBytes + kProbabilityBytes + kOutputBytes;
constexpr uint32_t kOnlineWorkspaceBytesPerCore = 2 * (kScoreBytes + kProbabilityBytes + kOutputBytes);

struct DenseParams {
    uint64_t q;
    uint64_t k;
    uint64_t v;
    uint64_t qv;
    uint64_t out;
    uint64_t lse;
    uint64_t workspace;
    uint64_t tasks;
    uint32_t queryLength;
    uint32_t kvLength;
    uint32_t heads;
    uint32_t headsPerTask;
    float scale;
};

template <typename Element>
void LaunchShort(const DenseParams& params, uint32_t blocks, uint64_t syncAddress, aclrtStream stream);

template <typename Element>
void LaunchOnline(const DenseParams& params, uint32_t blocks, uint64_t syncAddress, aclrtStream stream);

} // namespace flash_attn::mla
