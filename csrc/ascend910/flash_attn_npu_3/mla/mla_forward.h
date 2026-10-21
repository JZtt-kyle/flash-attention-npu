#pragma once
#include <torch/extension.h>
namespace flash_attn::mla {
std::optional<std::vector<at::Tensor>>
TryForward(at::Tensor q, at::Tensor k, at::Tensor v, std::optional<at::Tensor> k_new_, std::optional<at::Tensor> v_new_,
           std::optional<at::Tensor> q_v_, std::optional<at::Tensor> out_, std::optional<at::Tensor> cu_seqlens_q_,
           std::optional<at::Tensor> cu_seqlens_k_, std::optional<at::Tensor> cu_seqlens_k_new_,
           std::optional<at::Tensor> seqused_q_, std::optional<at::Tensor> seqused_k_,
           std::optional<int64_t> max_seqlen_q_, std::optional<int64_t> max_seqlen_k_,
           std::optional<at::Tensor> page_table_, std::optional<at::Tensor> kv_batch_idx_,
           std::optional<at::Tensor> leftpad_k_, std::optional<at::Tensor> rotary_cos_,
           std::optional<at::Tensor> rotary_sin_, std::optional<at::Tensor> seqlens_rotary_,
           std::optional<at::Tensor> q_descale_, std::optional<at::Tensor> k_descale_,
           std::optional<at::Tensor> v_descale_, std::optional<double> softmax_scale_, bool is_causal,
           int64_t window_size_left, int64_t window_size_right, int64_t attention_chunk, double softcap_option,
           bool is_rotary_interleaved, std::optional<at::Tensor> scheduler_metadata_, int64_t num_splits,
           std::optional<bool> pack_gqa_, int64_t sm_margin);
at::Tensor SchedulerMetadataOp(int64_t batch_size, int64_t max_seqlen_q, int64_t max_seqlen_k, int64_t num_heads_q,
                               int64_t num_heads_kv, int64_t headdim, int64_t headdim_v, at::ScalarType qkv_dtype,
                               at::Tensor cache_seqlens, std::optional<at::Tensor> seqlens_q,
                               std::optional<at::Tensor> cu_seqlens_k, std::optional<at::Tensor> cu_seqlens_k_new,
                               std::optional<at::Tensor> cache_leftpad, std::optional<int64_t> page_size,
                               int64_t max_seqlen_k_new, bool causal, int64_t window_size_left,
                               int64_t window_size_right, int64_t attention_chunk, bool has_softcap, int64_t num_splits,
                               std::optional<bool> pack_gqa, int64_t sm_margin);

at::Tensor SchedulerMetadata(int64_t batch, int64_t sq, int64_t sk, int64_t heads, int64_t kvHeads, int64_t dim,
                             int64_t valueDim, at::ScalarType dtype, const at::Tensor& lengths,
                             const std::optional<at::Tensor>& cuQ, std::optional<int64_t> page, int64_t left,
                             int64_t right, bool causal, bool hasSoftcap, int64_t splits, std::optional<bool> pack,
                             int64_t margin);

// The raw fwd caller validates the common boundary and owns the device guard.
std::vector<at::Tensor> Forward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                const std::optional<at::Tensor>& qv, const std::optional<at::Tensor>& cuQ,
                                const std::optional<at::Tensor>& cuK, const std::optional<at::Tensor>& usedQ,
                                const std::optional<at::Tensor>& usedK, std::optional<int64_t> maxQ,
                                std::optional<int64_t> maxK, const std::optional<at::Tensor>& pageTable,
                                std::optional<float> softmaxScale, bool causal, int64_t left, int64_t right,
                                float softcap, const std::optional<at::Tensor>& metadata, int64_t splits,
                                std::optional<bool> pack, int64_t margin);
std::vector<at::Tensor> GqaForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                   const std::optional<at::Tensor>& qv, int64_t splits, int64_t left, int64_t right,
                                   bool causal, std::optional<double> scale, const std::optional<at::Tensor>& usedQ,
                                   const std::optional<at::Tensor>& usedK, const std::optional<at::Tensor>& metadata,
                                   const std::optional<at::Tensor>& cuQ, const std::optional<at::Tensor>& cuK,
                                   int64_t maxQ, int64_t maxK, const std::optional<at::Tensor>& pageTable,
                                   double softcap, std::optional<bool> pack, int64_t margin);
at::Tensor GqaMetadataOp(int64_t batch, int64_t sq, int64_t sk, int64_t heads, int64_t kvHeads, int64_t valueDim,
                         at::ScalarType dtype, const at::Tensor& lengths, const std::optional<at::Tensor>& cuQ,
                         int64_t splits, int64_t left, int64_t right, bool causal, int64_t pageSize, bool hasSoftcap,
                         std::optional<bool> pack, int64_t margin,
                         const std::optional<at::Tensor>& usedQ = std::nullopt);
at::Tensor PrefillMixedMetadataOp(int64_t batch, int64_t maxQ, int64_t sk, int64_t heads, int64_t kvHeads,
                                  at::ScalarType dtype, const at::Tensor& lengths, const at::Tensor& cuQ,
                                  int64_t splits, int64_t left, int64_t right, bool causal,
                                  const std::optional<at::Tensor>& usedQ = std::nullopt, int64_t pageSize = 0,
                                  bool hasSoftcap = false);
at::Tensor PrefillBsndMetadataOp(int64_t batch, int64_t sq, int64_t sk, int64_t heads, int64_t kvHeads,
                                 at::ScalarType dtype, const at::Tensor& lengths, int64_t splits, int64_t left,
                                 int64_t right, bool causal, int64_t pageSize = 0, bool hasSoftcap = false);
std::vector<at::Tensor>
PrefillBsndForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, float scale, int64_t splits,
                   int64_t left, int64_t right, bool causal, const std::optional<at::Tensor>& usedQ,
                   const std::optional<at::Tensor>& usedK, const std::optional<at::Tensor>& metadata,
                   const std::optional<at::Tensor>& cuQ = std::nullopt,
                   const std::optional<at::Tensor>& cuK = std::nullopt, int64_t maxQ = 0, int64_t maxK = 0,
                   const std::optional<at::Tensor>& pageTable = std::nullopt, float softcap = 0.0f);
// TND explicit/native automatic split (v19/v20). Shared host adapters validate
// allocation/used lengths and optional records before launching the kernels.
std::vector<at::Tensor> SplitTndForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                        const at::Tensor& qv, const at::Tensor& cuQ, const at::Tensor& cuK,
                                        int64_t maxQ, int64_t maxK, float scale, int64_t left, int64_t right,
                                        bool causal, int64_t splits, const std::optional<at::Tensor>& usedQ,
                                        const std::optional<at::Tensor>& usedK,
                                        const std::optional<at::Tensor>& metadata, float softcap = 0.0f);
std::vector<at::Tensor> SplitBsndForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                         const at::Tensor& qv, float scale, int64_t windowLeft, int64_t windowRight,
                                         bool causal, int64_t splits, const std::optional<at::Tensor>& cacheSeqlens,
                                         const std::optional<at::Tensor>& usedQuery,
                                         const std::optional<at::Tensor>& schedulerMetadata, float softcap = 0.0f);
std::vector<at::Tensor> WindowBsndForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                          const at::Tensor& qv, float scale, int64_t windowLeft, int64_t windowRight,
                                          bool causal, const std::optional<at::Tensor>& cacheSeqlens,
                                          const std::optional<at::Tensor>& usedQuery,
                                          const std::optional<at::Tensor>& schedulerMetadata);

std::vector<at::Tensor> CausalBsndForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                          const at::Tensor& qv, float scale,
                                          const std::optional<at::Tensor>& cacheSeqlens,
                                          const std::optional<at::Tensor>& usedQuery,
                                          const std::optional<at::Tensor>& schedulerMetadata);
at::Tensor SplitBsndMetadataOp(int64_t batch, int64_t sq, int64_t sk, int64_t heads, at::ScalarType dtype,
                               const at::Tensor& cacheSeqlens, int64_t windowLeft, int64_t windowRight, bool causal,
                               int64_t splits, bool hasSoftcap = false);

at::Tensor WindowBsndMetadataOp(int64_t batch, int64_t sq, int64_t sk, int64_t heads, at::ScalarType dtype,
                                const at::Tensor& cacheSeqlens, int64_t windowLeft, int64_t windowRight, bool causal);

at::Tensor CausalBsndMetadataOp(int64_t batch, int64_t sq, int64_t sk, int64_t heads, at::ScalarType dtype,
                                const at::Tensor& cacheSeqlens);
std::vector<at::Tensor> SplitMixedForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                          const at::Tensor& qv, const at::Tensor& cuQ, int64_t maxQ, float scale,
                                          int64_t windowLeft, int64_t windowRight, bool causal, int64_t splits,
                                          const std::optional<at::Tensor>& cacheSeqlens,
                                          const std::optional<at::Tensor>& usedQuery,
                                          const std::optional<at::Tensor>& schedulerMetadata, float softcap = 0.0f);
at::Tensor SplitMixedMetadataOp(int64_t batch, int64_t maxQ, int64_t sk, int64_t heads, at::ScalarType dtype,
                                const at::Tensor& cacheSeqlens, const at::Tensor& cuQ, int64_t windowLeft,
                                int64_t windowRight, bool causal, int64_t splits, bool hasSoftcap = false);
std::vector<at::Tensor> WindowMixedForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                           const at::Tensor& qv, const at::Tensor& cuQ, int64_t maxQ, float scale,
                                           int64_t windowLeft, int64_t windowRight, bool causal,
                                           const std::optional<at::Tensor>& cacheSeqlens,
                                           const std::optional<at::Tensor>& usedQuery,
                                           const std::optional<at::Tensor>& schedulerMetadata);
at::Tensor WindowMixedMetadataOp(int64_t batch, int64_t maxQ, int64_t sk, int64_t heads, at::ScalarType dtype,
                                 const at::Tensor& cacheSeqlens, const at::Tensor& cuQ, int64_t windowLeft,
                                 int64_t windowRight, bool causal);
std::vector<at::Tensor> CausalMixedForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                           const at::Tensor& qv, const at::Tensor& cuQ, int64_t maxQ, float scale,
                                           const std::optional<at::Tensor>& cacheSeqlens,
                                           const std::optional<at::Tensor>& usedQuery,
                                           const std::optional<at::Tensor>& schedulerMetadata);
at::Tensor CausalMixedMetadataOp(int64_t batch, int64_t maxQ, int64_t sk, int64_t heads, at::ScalarType dtype,
                                 const at::Tensor& cacheSeqlens, const at::Tensor& cuQ);
// Call only after the common Python/C++ dense boundary has validated inputs.
std::vector<at::Tensor> DenseForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                     const at::Tensor& qv, float scale,
                                     const std::optional<at::Tensor>& cacheSeqlens = std::nullopt,
                                     const std::optional<at::Tensor>& schedulerMetadata = std::nullopt);
at::Tensor DenseMetadata(int64_t batch, int64_t sq, int64_t heads, int64_t sk, at::ScalarType dtype,
                         const at::Tensor& cacheSeqlens);
at::Tensor TndBsndMetadataOp(int64_t batch, int64_t maxQ, int64_t sk, int64_t heads, at::ScalarType dtype,
                             const at::Tensor& cacheSeqlens, const at::Tensor& cuQ);
at::Tensor PagedMetadataOp(int64_t batch, int64_t maxQ, int64_t capacity, int64_t heads, int64_t pageSize,
                           at::ScalarType dtype, const at::Tensor& cacheSeqlens, const std::optional<at::Tensor>& cuQ,
                           bool causal = false, int64_t windowLeft = -1, int64_t windowRight = -1, int64_t splits = 1,
                           bool hasSoftcap = false);
// BSND PA shared by public kvcache and internal dispatch. Performs its own
// tensor/storage validation. Internal Q-used always generates a fresh record.
std::vector<at::Tensor> PagedBsndForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                         const at::Tensor& qv, const at::Tensor& pageTable, float scale,
                                         const std::optional<at::Tensor>& cacheSeqlens,
                                         const std::optional<at::Tensor>& usedQuery,
                                         const std::optional<at::Tensor>& schedulerMetadata, bool causal = false,
                                         int64_t windowLeft = -1, int64_t windowRight = -1, int64_t splits = 1,
                                         float softcap = 0.0f);
// TND PA, including causal v11 and window v16. Allocation prefixes determine sample starts
// independently of effective Q-used lengths.
std::vector<at::Tensor> PagedTndForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                        const at::Tensor& qv, const at::Tensor& pageTable, const at::Tensor& cuQ,
                                        int64_t maxQ, float scale, const std::optional<at::Tensor>& cacheSeqlens,
                                        const std::optional<at::Tensor>& usedQuery,
                                        const std::optional<at::Tensor>& schedulerMetadata, bool causal = false,
                                        int64_t windowLeft = -1, int64_t windowRight = -1, int64_t splits = 1,
                                        float softcap = 0.0f);
std::vector<at::Tensor> WindowTndForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                         const at::Tensor& qv, const at::Tensor& cuQ, const at::Tensor& cuK,
                                         int64_t maxQ, int64_t maxK, float scale, int64_t windowLeft,
                                         int64_t windowRight, bool causal, const std::optional<at::Tensor>& usedQuery,
                                         const std::optional<at::Tensor>& usedKv);
std::vector<at::Tensor> TndForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& qv,
                                   const at::Tensor& cuQ, const at::Tensor& cuK, int64_t maxQ, int64_t maxK,
                                   float scale, const std::optional<at::Tensor>& usedQuery = std::nullopt,
                                   const std::optional<at::Tensor>& usedKv = std::nullopt, bool causal = false);
std::vector<at::Tensor> TndBsndForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                       const at::Tensor& qv, const at::Tensor& cuQ, int64_t maxQ, float scale,
                                       const std::optional<at::Tensor>& cacheSeqlens,
                                       const std::optional<at::Tensor>& schedulerMetadata);
std::vector<at::Tensor> BsndUsedForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                        const at::Tensor& qv, const at::Tensor& usedQuery,
                                        const std::optional<at::Tensor>& usedKv, float scale);
std::vector<at::Tensor> TndBsndUsedForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                           const at::Tensor& qv, const at::Tensor& cuQ, int64_t maxQ,
                                           const at::Tensor& usedQuery, const std::optional<at::Tensor>& usedKv,
                                           float scale);
} // namespace flash_attn::mla
