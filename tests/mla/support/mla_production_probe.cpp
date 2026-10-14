// Test-only C bridge to the actually linked production launchers. No device
// implementation is compiled here and no test entry is added to the extension.
#include "mla/mla_launch.h"
#include "runtime/rt_ffts.h"

extern "C" int launch_dense_metadata(void* stream, void* qc, void* qr, void* kv, void* kr, void*, void* output,
                                     void* lse, void* workspace, void* metadata, uint32_t words, void*,
                                     const flash_attn::mla::MlaDenseMetadataContext* context, float scale)
{
    using namespace flash_attn::mla;
    if (!context || words != kMlaDenseMetadataWords) {
        return -1;
    }
    MlaLaunchArgs args{};
    args.stream = stream;
    uint32_t length = 0;
    auto status = rtGetC2cCtrlAddr(&args.ffts, &length);
    if (status != RT_ERROR_NONE) {
        return status;
    }
    args.qc = static_cast<uint8_t*>(qc);
    args.qr = static_cast<uint8_t*>(qr);
    args.kv = static_cast<uint8_t*>(kv);
    args.kr = static_cast<uint8_t*>(kr);
    args.output = static_cast<uint8_t*>(output);
    args.lse = static_cast<uint8_t*>(lse);
    args.workspace = static_cast<uint8_t*>(workspace);
    args.metadata = static_cast<uint8_t*>(metadata);
    args.context = *context;
    args.scale = scale;
    return context->dtype == uint32_t(MlaMetadataDtype::Bf16) ? LaunchDenseBf16(args) : LaunchDenseFp16(args);
}
