// Test-only invalid storage metadata injection. Never resize/free NPU memory or
// enqueue device work. The Python fixture restores the size before destruction.
#include <c10/core/TensorImpl.h>
#include <cstdint>

extern "C" uint64_t mla_test_storage_bytes(uint64_t tensor, uint64_t bytes)
{
    const auto& storage = reinterpret_cast<c10::TensorImpl*>(tensor)->storage();
    const uint64_t previous = storage.nbytes();
    storage.unsafeGetStorageImpl()->set_nbytes(bytes);
    return previous;
}
