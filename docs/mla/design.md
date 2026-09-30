# CANN FIA MLA → FA3 NPU：适配方案

状态：方案已确认，按小步提交推进。本文描述目标契约，不表示这些能力已经实现。
当前进展见 [status.md](status.md)，迭代顺序见 [plan.md](plan.md)。

## 1. 范围与固定依据

迁移集合是 **910B/C 上 CANN FIA 已有 MLA 前向能力，与官方 FA3 可表达能力的交集**，再加接入所必需的 metadata 支持。接口冲突按 FA3 调整；不搬迁普通 MHA/GQA、量化大框架、950 专用模板或反向算子。MLA prefill 中使用 GQA 的分组语义仍在范围内。

**迁移原则：保留固定 CANN 实现的行为，只做 CATLASS 风格的结构重构和必要的 FA3 接入。** 以源实现的验证积累为基础，保持数值路径与设备流水，具体边界见第 6 节。

| 依据 | 固定版本 | 用途 |
| --- | --- | --- |
| 本仓 | `cab322eda3112fdbfefd99c8632b6393273562f3` | 迁移前基线 |
| CANN ops-transformer | `ca18aee610729ca5270a44e2d84c751bc687fbf0` | MLA 功能与性能来源 |
| CATLASS 子模块 | `769cd40a8716b28650b6bebb08db4834eea4462f` | 组件、分层与编程风格 |
| 官方 Dao-AILab/flash-attention | `d15f1531a460ba456f41b01a774f33ab2db8febf`，2026-09-22 | 取该提交的 `hopper/` FA3 实现；不混用 FA2、FA4 或 CuTe MLA 接口 |

官方一手依据：

- [Python 入口与 metadata 签名](https://github.com/Dao-AILab/flash-attention/blob/d15f1531a460ba456f41b01a774f33ab2db8febf/hopper/flash_attn_interface.py)：`flash_attn_func`、`flash_attn_varlen_func`、`flash_attn_with_kvcache`、`get_scheduler_metadata`。
- [C++ 校验与调度](https://github.com/Dao-AILab/flash-attention/blob/d15f1531a460ba456f41b01a774f33ab2db8febf/hopper/flash_api.cpp)：`mha_fwd`、`mha_fwd_get_scheduler_metadata`；以实际分支为准，不仅看 docstring。
- [官方测试](https://github.com/Dao-AILab/flash-attention/blob/d15f1531a460ba456f41b01a774f33ab2db8febf/hopper/test_flash_attn.py)：`has_qv`、异维 Q/V、varlen、kvcache、metadata 的组合。
- [split 合并](https://github.com/Dao-AILab/flash-attention/blob/d15f1531a460ba456f41b01a774f33ab2db8febf/hopper/flash_fwd_combine_kernel.h)：部分结果与 LSE 的合并行为。
- [softmax](https://github.com/Dao-AILab/flash-attention/blob/d15f1531a460ba456f41b01a774f33ab2db8febf/hopper/softmax.h) 的 `finalize` 与 [写回](https://github.com/Dao-AILab/flash-attention/blob/d15f1531a460ba456f41b01a774f33ab2db8febf/hopper/epilogue_fwd.hpp) 的 `store` / `store_zero`：全遮罩行及 kernel 内空 KV 的 O/LSE；不能只参考 host 的全局空张量分支。

CANN 根目录：`/home/zhangqijia/repo/ops-transformer/attention/fused_infer_attention_score`。下列路径均相对于此目录：

| 来源 | 已核实的事实 |
| --- | --- |
| `op_host/arch22/fia_tiling_nonquant_mla.cpp`，`IsCapable` | 专用模板为 FP16/BF16，内容维 512、RoPE 64、KV 头数 1；含 PA、FlashDecode、mask 等分支 |
| `op_kernel/arch22/fia_kernel_nonquant_mla.h`、`fia_block_cube_nonquant_mla.h`、`fia_block_vec_nonquant_mla.h`、`fia_block_vec_flashdecode_arch22.h` | 解码流水、两个矩阵乘、softmax、输出更新及 split 合并 |
| `op_host/arch22/fia_tiling_nonquant.cpp`，`IsHighPerformanceTemplate`；`op_kernel/arch22/fia_kernel_nonquant.h` 及关联 Block | prefill 的内容维 128 + RoPE 64、V 维 128 在通用非量化路径中，不能按文件名只取带 `mla` 的文件 |
| `op_host/arch22/fused_infer_attention_score_tiling_check_feature.cpp` | 维度、布局、mask、页大小和特性交叉约束；`CheckFeatureHeadDim`、`CheckFeatureAxisInfo`、`CheckFeatureBlockSize` 等 |
| `tests/pytest/test_fia_mla_d512_pa_dim0_noncontiguous.py` | PA 首维 stride 回归；golden 分别使用独立 K 和 V，不能假定源代码总是忽略 V |

源码与文档可能不一致。例如 V5 文档对 decode 的 K/V 复用描述不能替代上述 kernel 与测试；TensorList 的文档描述也必须经过 arch22 host 校验核实。实现时将实际可运行组合固定为用例，未证实组合标为待核实，不能静默删减。

## 2. FA3 接入要求

- [Python FA3 入口](../../flash_attn_npu_3/flash_attn_npu_interface.py) 与 [910 C++ 前向](../../csrc/ascend910/flash_attn_npu_3/flash_api.cpp) 按 `qv` 和维度识别两种 MLA 表示，透传内容与 RoPE 分量，明确校验支持边界。
- 输出按 V 维度分配，真实路径与 fake 对齐 MLA 64/512、192/128 的 O/LSE shape。
- dense/varlen metadata 调用传入实际 `headdim_v=v.shape[-1]`，使入口、调度和设备执行使用同一组维度。
- 复用 [metadata 参数](../../csrc/ascend910/flash_attn_npu_3/fa_metadata_args.h) 和 [AICPU 实现](../../csrc/ascend910/flash_attn_npu_3/fa_metadata.aicpu) 的启动与流同步机制，MLA 使用与源实现对应的调度字段。
- 普通 FA 保持现有 `uint8` metadata 协议兼容；MLA 提供设备 `int32` 调度 tensor，公开使用 `has_softcap`，计算 scale/softcap 数值由前向传入。
- 窗口归一化、split 范围及未支持参数按固定官方 FA3 分支逐项核对。

## 3. 两种 MLA 的接口映射

### 3.1 压缩 MLA：512 + 64

CANN 一般形式为 `scores = (Q_content K_contentᵀ + Q_rope K_ropeᵀ) * scale`，再以 V 加权求和。
FA3 的 `qv` 形式为 `scores = (q kᵀ + qv vᵀ) * scale`，输出为 `softmax(scores) v`。

| CANN 含义 | FA3 参数 | 典型 dense shape |
| --- | --- | --- |
| queryRope | `q` | `[B, Sq, Hq, 64]` |
| keyRope | `k` / `k_cache` | `[B, Sk, 1, 64]` |
| query 的内容部分 | `qv` | `[B, Sq, Hq, 512]` |
| 共享的压缩 KV | `v` / `v_cache` | `[B, Sk, 1, 512]` |
| attentionOut | `out` | `[B, Sq, Hq, 512]` |

内部可将 `qv → CANN query`、`q → queryRope`、`k → keyRope`、`v → key 和 value`，直接重用同一压缩 KV 数据，不拼接 576 维张量、不展开所有头。

**边界必须明确：** CANN 可计算的独立 `K_content != V` 的 512 维场景，FA3 `qv` 接口不能等价表达；拼成 Q/K 576 维又超过该 FA3 路径的维度范围。这一组合按“TriDao 不支持特性除外”排除，不能通过偷偷替换 V 宣称支持。源测试中这类数据要重新构造为共享压缩 KV，并保留该排除说明。

### 3.2 展开 MLA prefill：192 / 128

- FA3 输入 `q=[Q_content(128), Q_rope(64)]`、`k=[K_content(128), K_rope(64)]`，`v` 为独立 128 维，`qv=None`；输出最后一维为 128。
- CANN RoPE 合并和分离两种形式归一到这一入口。FA3 的 `qv` 校验要求 V 维度至少为 256，不能把 128 维 prefill 内容塞进 `qv`。
- kernel 内可读取 192 维完整行，或用带真实 stride 的两个视图读取内容/RoPE；slice 后行 stride 仍可能是 192，不能当作紧凑 128 或 64。
- “prefill”和“decode”表示这里的数据表示方式，不按 `Sq==1` 分派。512 维路径也必须覆盖多 query token 和变长 query。

### 3.3 共同契约

| 差异 | 适配规则 |
| --- | --- |
| 入口 | 保留 `flash_attn_func`、`flash_attn_varlen_func`、`flash_attn_with_kvcache`；内部按 qv、Q/K/V 维度分派，无新 `flash_mla` 公共入口 |
| scale | 显式值原样生效；缺省按 FA3：`(Dq + Dqv)^(-1/2)`，无 qv 则 `Dq^(-1/2)`。典型 decode 为 `1/sqrt(576)`，prefill 为 `1/sqrt(192)`；CANN 对照显式传同一值 |
| RoPE | CANN 的 queryRope/keyRope 是已形成的位置分量，不等于 FA3 的 `rotary_cos/sin`。本迁移接收已处理的分量，不能重复旋转 |
| dense/varlen | 对外 BSND / TND；CANN BSH、BNSD、NTD 等在模型边界用 reshape/permute 归一，支持最后一维连续的合法外层 stride，不每步强制复制完整 cache |
| 长度 | FA3 `cu_seqlens_*` 为带起始 0 的设备 int32 前缀和；`cache_seqlens` / `seqused_*` 是实际使用长度。CANN 常用 int64 且累计数组不带起始 0，内部重新解析，不能重解释指针 |
| 长度上界 | `max_seqlen_*`、cache 容量、每样本实际长度分别处理；padding 槽位不参与 softmax。不能只拿 batch 中最大实际长度生成所有样本的 mask |
| PA | `page_table:int32[B, max_pages]`，cache 为 `[num_pages, page_size, Hkv, D]`；携带 K、V 各自 stride，处理乱序页、尾页及首维非连续 |
| 头数 | decode 的迁移范围 Hkv=1；prefill 保留 CANN 与 FA3 共同支持的 Hq/Hkv 整除组合。不是把整套普通 GQA 移过来 |
| mask | 对外只用 `causal`、`window_size`；右下对齐的位置中心为 `i + Sk - Sq`。CANN 0/3/4 模式归一为无 mask/右下 causal/band |
| 窗口归一化 | 固定 FA3 版本在入口按 `Sk-1`、`Sq-1` 处理左右窗口，并有单 query 分支；数学窗口与实际归一化分支都要测试，不能只验证等长序列 |
| 输出 | O 与 V 的末维相同、dtype 与 FP16/BF16 输入相同；LSE 为 FP32，dense `[B,Hq,Sq]`、varlen `[Hq,Tq]`，数学量为自然对数 |
| 空行/空 KV | O 为 0。整个 K 张量容量为 0 时，`mha_fwd` host 分支令 LSE 为 `+inf`；容量非零但某行全被 mask 或样本实际 KV 长度为 0 时，`softmax::finalize` / `epilogue::store_zero` 写 `-inf`，split 全空合并也为 `-inf`。零 query 的 O/LSE 保留空 shape；不能把所有空行都改成 `+inf` |
| 返回值 | 按入口对齐：dense/varlen 的可选 LSE 返回二元组；固定版本 kvcache 开启 `return_softmax_lse` 实际返回 O、LSE 及两个 accumulator，共四项，不能只按旧 docstring 实现 |
| split | `num_splits=0` 自动、1 不分割、>1 请求分割，固定官方实现上限为 256；核数限制属于实现资源策略，不能把现仓仅 PA+TND、至多核数的限制误称为 FA3 限制 |

## 4. 功能边界

| 能力 | 处理与验收 |
| --- | --- |
| FP16/BF16；512+64 压缩表示；192/128 prefill | 必迁，含 out 和 LSE |
| dense、TND、实际长度、GQA 分组（prefill） | 必迁；含非均匀长度、多 token、小 batch、尾块及零长度边界 |
| PA、16 对齐且不超过 1024 的源页大小、合法 stride | 必迁可表达的 ND 路径；源 host `CheckFeatureBlockSize` 作为 CANN 边界，FA3 实际约束另核对；不照搬旧版文档的页大小假设 |
| 无 mask、右下 causal、有限/单侧窗口、FlashDecode | 必迁；包含功能组合，不能仅验证每个开关单独可用 |
| metadata、自动/显式调度、重复执行、图执行 | 本需求必需；普通 FA3 的既有路径需回归 |
| CANN 独立 512 维内容 K/V | FA3 qv 无等价表达，排除；见 3.1 |
| tree mask / sparseMode=9、任意用户 mask、独立左上 causal 接口 | FA3 hopper 公共入口没有等价参数，排除；保留能用标准右下窗口表达的计算 |
| 外部 NZ cache、CANN 布局字符串和特殊输出布局 | 不增加 CANN 专用公开参数；普通布局可由视图归一，NZ 外部格式按 FA3 不支持排除。内部是否使用分块布局由性能决定 |
| INT8 全量化、伪量化、后量化、CANN 专用量化因子 | 不迁移；不以 FA3 的 FP8 descale 冒充等价支持；950 的量化 MLA 不在芯片范围内 |
| PSE、公共前缀、TensorList、padding 参数、sink | 不能从 FIA 总签名推断是本次 MLA 必迁能力；按 arch22 实际校验与 FA3 交集记录结论。已检查的 MLA 路径拒绝多项此类参数，FA3 无等价的源能力也排除 |
| 新 KV 原地追加、内置 rotary、cache_leftpad、attention_chunk、FA3 额外维度/头数组合 | 不因 FA3 签名存在就扩为本次 CANN MLA 迁移任务；未实现组合明确报错，不允许忽略。若后来源能力盘点证明属于交集，补入计划 |
| softcap、pack_gqa、sm_margin | 数学或调度参数不能默默丢弃；MLA 适配分别实现其有效语义，硬件调度方式可不同，资源不足清楚报错 |
| backward | 原 FIA 为推理前向，迁移不扩反向范围；qv 路径上游测试也跳过 backward。新增 MLA 前向不能误走本仓等维 backward，需要明确拦截未支持的梯度请求 |

这张表是需求边界；下列来源清单和代表用例定义验收追踪关系，运行结果只记入 status。

### 4.1 源能力清单

下表的 CANN 路径相对于第 1 节固定源目录，FA3 路径相对于固定官方提交的 `hopper/`。函数名用于源码追踪，不把 host 接受或 tiling 分支存在等同于已实测通过。用例 ID 对应 4.2；排除项在后续入口接通时必须明确拒绝。

| 能力 | 固定来源及边界 | 结论 / 用例 |
| --- | --- | --- |
| 512+64，FP16/BF16，KV 头数 1，单/多 query | `op_host/arch22/fia_tiling_nonquant_mla.cpp::IsCapable`；`op_kernel/arch22/fia_block_cube_nonquant_mla.h` 的内容与 RoPE 乘法 | 纳入共享 KV 子集；D01、D02 |
| 192/128 prefill，RoPE 分离/合并，独立 V | `op_host/arch22/fia_tiling_nonquant.cpp::IsHighPerformanceTemplate`；`fused_infer_attention_score_tiling_check_feature.cpp::CheckFeatureHeadDim`；通用 `fia_kernel_nonquant.h` | 纳入；P01、P02，不能由 decode 文件名推断不支持 |
| prefill 分组头 | `op_host/arch22/fia_tiling_nonquant.cpp` 的 `set_gSize(n1Size / n2Size)`；`fused_infer_attention_score_tiling_check_consistency.cpp` 的 `CheckQShape` / `CheckKVShape`；FA3 `mha_fwd` 的 `num_heads % num_heads_k` 校验 | 纳入两端有效整除组合；P01、P02；512 路径仍只允许 Hkv=1 |
| BSH/BSND/BNSD、TND/NTD、特殊输出布局 | MLA `IsCapable` 和 `CheckFeatureLayout`，专用 decode 不接受 NTD query，通用 prefill 可接受 | BSND/TND 纳入；普通布局由视图归一，特殊输出不增加公开参数；D/P/T01、ST01 |
| 实际长度、累计长度、padding 槽位 | `CheckFeatureActualSeqLens*`、`GetQActualSeqLenForPostQuant` / `GetKVActualSeqLenForPostQuant`；FA3 `cu_seqlens_*` / `seqused_*` / `cache_seqlens` | 纳入；T01、U01；CANN int64 累计尾值与 FA3 带 0 的 int32 前缀和须数值转换 |
| ND PA、页大小、尾页 | `CheckFeatureBlockSize`、`CheckFeatureLayout`、MLA 与通用 `GenTilingKey` 的 PA 分支；FA3 `mha_fwd` 的 page table 解析 | 纳入；PA01；源页大小为 16 的倍数且 16..1024；不把 NZ 一并纳入 |
| PA 首维不连续，K/V/RoPE 独立 stride | `fia_tiling_nonquant_mla.cpp::FillArch22MlaKvStrideParams`；`tests/pytest/test_fia_mla_d512_pa_dim0_noncontiguous.py::STRIDE_PROFILES` | 纳入 FA3 可表达的 stride；PA02；压缩表示中 FA3 K 是 RoPE，FA3 V 是共享内容 KV |
| 无 mask、右下 causal、band/window | `CheckFeatureMask`，MLA `IsCapable` 的 sparse 0/3/4；`flash_api.cpp::mha_fwd` 窗口归一化 | 纳入；MS01..MS03；源 sparse 3/4 仍需合法 mask tensor，基线适配不能只传 mode |
| FlashDecode / split-KV | 两个 tiling 的 `IsFlashDecode` / `GenTilingKey`；`op_kernel/arch22/fia_block_vec_flashdecode_arch22.h`；官方 combine kernel | 纳入；FD01、FD02；源自动分割与 FA3 0/1/显式 split 分别记录，不能伪称 CANN 有同一公开参数 |
| O、可选 LSE、空张量、空行 | `CheckFeatureLse`、`fia_tiling_empty_tensor.cpp`；FA3 `mha_fwd`、`softmax.h::finalize`、`epilogue_fwd.hpp::store_zero` | 纳入；Z01、MS02；空行符号按 FA3 路径区分，见 3.3 |
| metadata、图、重复执行 | 本仓 `fa_metadata.aicpu`；官方 `get_scheduler_metadata`、`mha_fwd_get_scheduler_metadata` | 接入必需的适配，非宣称原 FIA 签名已有相同 metadata 协议；MD01、GR01 |
| softcap、pack_gqa、sm_margin | 官方 `flash_attn_interface.py` 与 `flash_api.cpp` 的数学/调度参数；源 FIA 不能直接等参对照 | 纳入有效语义；SC01、MD01；softcap 用数学 reference 验收，CANN 无同义参数时单列性能口径 |
| 独立 512 维内容 K/V | 源 PA stride 测试的独立 golden K/V；FA3 `mha_fwd` 的 qv 维度校验 | 排除 X01：FA3 qv 的内容点积必须使用输出 V；不能替换源 V 来掩盖差异 |
| tree、任意 mask、独立左上 causal | MLA `IsCapable` 支持 sparse 9，`CheckFeatureSparseMode` 限制；FA3 公共入口只提供 causal/window | 排除 X02：无等价公开参数；可表达的标准窗口计算仍纳入 MS01..MS03 |
| 外部 NZ cache | `CheckFeatureLayout` / `CheckFeatureHeadDim` 的 NZ 分支与源 PA 测试 | 排除 X03：FA3 无外部 NZ 协议；ND PA 必迁，CANN 原生 NZ 性能可另列 |
| 全量化、伪量化、后量化和量化因子 | MLA `IsCapable` 的输入/输出 dtype 校验；`CheckParaExistenceMlaNoquant`；其他 arch/量化模板 | 排除 X04：超出此次非量化 910B/C 范围，不用 FP8 descale 替代 |
| TensorList | `CheckFeatureTensorList` 拒绝任何非 NO_ROPE 模式；专用 MLA `IsCapable` 也拒绝 | 排除 X05：不属于此次 MLA 源能力 |
| PSE、公共前缀、query/KV padding 参数 | split 由 `CheckParaExistenceMlaNoquant` 拒绝；combined 由 `CheckFeaturePSE` / `CheckFeatureGqaPrefix` / `CheckFeatureLeftPadding` 拒绝非 NO_ROPE | 排除 X06：不因 FIA 总签名存在就迁移；实际长度与有效区末尾 padding 仍纳入 U01 |
| learnable sink | 专用 MLA `IsCapable` 拒绝；通用路径 `CheckFeatureLearnableSink` 可接受每 query head 的 BF16 sink | 排除 X07：固定 FA3 hopper 无 sink 参数，不能宣称所有 CANN MLA 都不支持 |
| KV 追加、内置 rotary、cache_leftpad、attention_chunk、额外维度/头数、反向 | 固定 FA3 入口有前述额外选项；源本次 MLA 是已形成 RoPE 的前向计算，padding 校验见 X06 | 排除 X08：不是已确认迁移交集；MLA 梯度请求与未实现选项明确拒绝 |

### 4.2 代表用例与参考实现边界

所有数值用例使用 FP16 和 BF16。`D` 表示共享 512+64，`P` 表示 192/128；除 D 的 Hkv=1 外，头数写为 Hq/Hkv。未特别标注时无 mask、默认 scale，两个表示均校验 O/LSE。下表是待接通功能的验收输入清单，**不是已运行 NPU 测试列表**；无意义或源拒绝的交叉组合须记录原因，不以 skip 冒充覆盖。

| ID | 代表输入 / 交叉条件 | 验收重点 |
| --- | --- | --- |
| D01 | D，BSND，B=2，Sq=1，Sk=257，8/1；默认 scale 与 0.23 | 两项点积、共享 KV、非整块 KV、输出 512；M02 首批基线 |
| D02 | D，BSND，B=1，Sq=7，Sk=65，16/1 | 多 query 不误分派到 prefill；M02 首批基线 |
| P01 | P，BSND，B=2，Sq=17，Sk=31，8/2；源 split 与 combined RoPE | 两种 CANN 表示对同一 FA3 输入，独立 V、GQA、输出 128；M02 首批基线 |
| P02 | P，BSND，B=1，Sq=33，Sk=129，8/8 和 8/1 | 非分组/最大分组及尾块，内容/RoPE 视图 row stride=192 |
| T01 | D/P，TND，Q 长度 [3,0,1]，KV 长度 [17,0,33]；增加 `seqused` 小于分配长度的变体 | FA3 前缀和 [0,3,3,4] / [0,17,17,50] 与 CANN 尾值 [3,3,4] / [17,17,50]；累计/实际长度不混用 |
| U01 | D/P，BSND 容量 Sq=7、Sk=33，Q 实用 [3,1]、KV 实用 [17,0]，padding 填 poison | mask 按样本实际长度；非零容量中的空 KV LSE=-inf；分配容量不等于实际长度 |
| ST01 | D/P，BSND 与 TND 外层切片、非零 storage offset、BNSD/NTD 归一视图 | 最后一维连续，batch/row/head stride 保留；不每次复制完整 cache |
| PA01 | D/P，BSND/TND query，Q [3,1]，KV [page_size+1,2*page_size-3]，乱序页；页大小遍历 16..1024 的 16 倍数 | 尾页、页表 padding、无关物理页；性能代表页 16/128/256/1024 |
| PA02 | PA01，K/V 首维 padding 分别取 (17,0)、(0,29)、(17,37)，另测所有外层 stride | D 的 K 为 RoPE、V 为共享内容，P 保留独立 192/128 K/V；地址计算独立；源独立 512 维内容 K/V 用例不能原样宣称可迁移 |
| MS01 | D/P，Sq=2、Sk=5，causal；dense/TND/PA | 允许列分别为 0..3、0..4，右下对齐 |
| MS02 | D/P，Sq=5、Sk=2，causal；dense/TND/PA，有/无 split | 前三行全空，O=0、LSE=-inf；后两行分别看 1、2 个 KV |
| MS03 | D/P，(Sq,Sk)=(3,5)/(1,3)/(5,2)，窗口 (1,0)/(0,0)/(-1,0)/(0,-1)，叠加 causal；另取左右界恰为 Sk-1/Sq-1 | 单侧窗口、单 query、归一化边界、跨 PA 页的有限窗口 |
| FD01 | D/P，Sq=1/7，Sk=8193，split=0/1/2/8/256，dense/TND/PA | O/LSE 与不分割一致，显式分割资源校验、长 KV 性能；CANN 使用原生自动策略 |
| FD02 | D/P，小 KV、split 多于有效块、部分/全部空 split | 中间 -inf 哨兵和合并，accumulator shape/dtype/返回数量，不把未初始化槽位当数值 |
| Z01 | D/P，(Sq,Sk)=(0,3)/(3,0)/(0,0)，再测 batch 内零长度 | 全局空 K 为 +inf，kernel 内空行 -inf；零 query 的空 shape 与无 NaN |
| MD01 | D/P，D01/P01 输入，有/无 metadata，scale 默认/0/0.23；pack_gqa None/False/True；sm_margin 0/合法正值/耗尽资源 | 默认 scale 不写死到调度；跨层复用和参数有效语义，普通 FA byte metadata 回归 |
| SC01 | D/P，D02/P01 输入，softcap 0/15；metadata 复用时修改正 softcap 数值 | softcap 作用于两个分量总分数乘 scale 后的值；`has_softcap` 只表达调度特征 |
| GR01 | D/P，T01/PA01，图内长度修改并重算 metadata；多次 replay、跨层、多 stream | 图内/图外 metadata 生命周期、fake 与真实输出、workspace 隔离 |

[独立 CPU reference](../../tests/common/mla_ref.py) 当前只接受规范 BSND 逻辑张量。高精度用 FP64，低精度对照用输入 dtype 点积与 FP32 softmax，最终 O 为输入 dtype、LSE 为 FP32；两者都不调用 FA3/CANN 或现仓普通 attention golden。`fa3_to_cann_inputs` 显式传同一 scale，prefill 可返回 stride 保真的 split 视图或 combined 表示；反向映射要求 512 维 K/V 为同一 storage view，拒绝数值暂时相等的独立 buffer。

[CPU 验收](../../tests/test_flash_attn_npu_v3_mla_reference.py) 用独立标量求和/指数 oracle 和手写可见列矩阵核对上述数学规则；不共用 reference 的 mask 或张量收缩。M01 覆盖 D/P 数学、共享边界、scale、GQA、stride、mask、softcap 与全局空张量；TND/PA 解析、metadata、split accumulator 和图仍按计划逐步验证。

M02a 的 [CANN 执行器](../../tests/mla_cann_baseline.py) 与 [910B 验收](../../tests/test_flash_attn_npu_v3_mla_cann.py) 仅覆盖无 mask 的 BSND D01/D02/P01，显式传入相同 scale 并保留 prefill split 视图的 192 行 stride。使用 `npu_fused_infer_attention_score_v2.out` 显式分配 Dv 维输出，避免 torch_npu 2.7.1 默认包装器将 combined 192/128 的输出错分配为 192。O/LSE 均调用原仓 `assert_fa_close`，无 softcap 时阈值固定为 `max(2 * max|pt-ref| + 2 * ULP(ref), 1e-5)`；不因基线误差修改阈值。

该执行器记录安装版本、进程实际加载的算子库及 SHA256、输入 shape/stride/offset、每轮 event span 和逐调用同步端到端时间。输入 H2D 与一次性视图创建不计时，输出分配及包装器内转换计时；event span 可能含 host 发射间隙，不当作纯 kernel 时间。安装包与固定源码的二进制对应关系未经验证时，结果仅验收执行器；M02b 必须建立固定源码基线并用 profiler 核实模板、kernel 时间与转换成本，之后才能用于迁移性能验收。

## 5. metadata 设计

metadata 指 FA3 的 `get_scheduler_metadata` → `scheduler_metadata` 调用链，不另造调用协议。

1. **公开签名以官方为准。** 使用 `headdim`、`headdim_v`、`has_softcap` 等官方参数及顺序。现仓额外 `softcap` / `softmax_scale` 如需兼容，只作为旧调用的关键字兼容层，不占官方位置参数、不成为 MLA 必填项；新调用和测试统一用官方形式。
2. **分离调度与计算数值。** metadata 保存切块、分核、split、长度等调度结果；前向本次调用提供 scale、softcap 数值、张量地址及 stride。metadata 接口没有 qv 参数，不应依据维度臆测 qv 存在，也不应把缺省 `1/sqrt(Dq)` 写死。只有 `has_softcap` 这类影响调度的特征进入调度契约。
3. **设备侧准备。** 复用现仓 AICPU 启动与事件同步，从固定 CANN 抽取不依赖 gert/optiling 的 MLA 调度函数，保留源判断条件和调度结果；FA3 显式 split 等契约差异单独映射。迁移调度计算位置和复用结果不等于重新设计切块、分核或 split 启发式。host 只用 shape、dtype、stride 等静态信息；长度读取与解析在设备侧完成，steady-state 不调用 `.cpu()`、`.item()` 或主机同步。
4. **不复制 GPU 内部二进制布局。** 新 MLA metadata 使用 NPU 上连续 int32 tensor，内容由 NPU 后端管理，带版本/类型及边界信息；兼容的是调用语义，不是 CUDA 与 NPU buffer 互换。普通 FA 现有 byte metadata 的过渡需保持既有调用可用，不能误送入 MLA kernel。
5. **有/无 metadata 共用调度。** 不传时内部生成；显式生成时可以复用相同调度条件。两条路径共享计算函数，不能分别维护两套容易漂移的 tiling 算法。
6. **生命周期明确。** 同设备、同调度参数与长度下可跨层复用；实际长度改变要重新计算。图内捕获 metadata 计算时，每次 replay 重新读取设备长度；图外生成的旧快照不会自动跟随长度张量变化。读写 workspace、计数器和 stream 的所有权要明确，禁止共享可变 scratch 造成并发污染。
7. **校验与图模式一致。** 校验尺寸、dtype、设备和可获得的静态参数；必要的内容检查留在设备侧。Python tensor 私有属性不能成为唯一正确性依据；不能为了校验在 graph 中读取设备标量。真实输出与 fake 的 metadata 大小、O/LSE/accumulator shape 必须一致。

M05b2a 对齐公开 `get_scheduler_metadata` 的官方位置参数及默认值，内部 custom op、fake 和 C++ 使用相同的 `has_softcap` 位置。`softcap` / `softmax_scale` 只作为公开关键字兼容项；普通 FA3 仍返回原 `uint8` 协议，其中保存数值，因此单独 `has_softcap=True` 暂时明确拒绝，须通过旧关键字提供正 `softcap`。旧调用省略 `has_softcap` 时仍以数值 `softcap` 启用该功能；显式 True 与零值冲突、负值或非有限值均拒绝。M05b2b3b 接通下述 dense MLA 的公开生成与复用；MLA 不接收这两个数值兼容关键字，scale 只由当次前向提供。其余 MLA metadata 组合继续明确拒绝。

## 6. 代码结构

实现接入点保持 [flash_api.cpp](../../csrc/ascend910/flash_attn_npu_3/flash_api.cpp) 和现有 Python 包。新增代码拟放在 `csrc/ascend910/flash_attn_npu_3/mla/`：

```text
Python FA3 入口 / custom op / fake
        ↓
C++ 校验、形状与 stride 解析、MHA/MLA 分派
        ↓
MlaLaunchParams（本次地址、scale 等） + MlaSchedule（可复用调度）
        ↓
CATLASS Kernel：保留 CANN tile 循环、缓冲、Cube/Vector 同步
        ↓
Block：QK + RoPE / online softmax / PV / rescale / split combine
        ↓
CATLASS Tile copy、Mmad、布局与硬件资源
```

- **以固定 CANN 源码为实现起点。** 将其已有 Kernel / Cube / Vector / FlashDecode 逻辑对应到 CATLASS Kernel / Block / Tile；保留函数来源及版权。CATLASS `examples/19_mla` 用于理解分层，不能替代 CANN 的实际执行流程。
- **允许的结构改动。** 文件拆分、类和模板组织、命名、参数封装、构建接入及框架依赖解耦；已有 Block / Tile 仅在运算、数据布局、精度及资源/同步契约等价时复用。无法等价复用时在本仓封装源逻辑，不为套用现成组件改变 CANN 行为。
- **保留的设备行为。** 对应输入和资源下的切块/分核策略、矩阵乘与 softmax/输出更新顺序、中间 dtype、累加和舍入位置、尾块处理、内部数据布局、缓冲数量与生命周期、预取时机及事件依赖。这些执行策略按固定 CANN 保持一致，结构重构围绕其接口和代码组织展开。
- **必要的接口差异单独处理。** 按第 3/5 节完成 FA3 参数、长度/stride、scale、mask、split 及 metadata 映射；保留原流水所需的调度字段。每项确需偏离 CANN 的行为要注明契约依据和最小影响范围，不能借接口适配顺带优化 kernel；不搬入范围外的 FIA 大签名、图引擎注册或量化框架。
- 910B/C 共用适用的算法，但 Cube/Vector 比例、可用核数和存储资源从平台信息获取；保留源 MLA 的 C1V1/C1V2 适配分支，不能用 910B 实测参数硬编码替代 910C 兼容性。
- 满足上述等价条件时复用已有 Tile；MLA 特有 Block 与 Kernel 先放本仓，避免每步改子模块。编译分派沿用轻量 host 头和独立实例化单元，更新生成器而非仅手改 autogen。
- metadata 的新 AICPU 源如独立成文件，必须同时修改 `setup.py`：当前构建只寻找固定的 `fa_metadata.aicpu`。header 改动的增量依赖也必须核查，防止链接旧对象。
- 测试使用独立数学 reference 和 CANN 基线适配器。生产入口不能直接调用 `torch_npu.npu_fused_infer_attention_score*` 充当移植实现。

### 6.1 首个 dense 实现的源对应关系

以下映射固定到第 1 节 CANN commit；路径仍相对于 FIA 根目录，`../common/` 指 `attention/common/`。目标路径相对于本仓 `csrc/ascend910/flash_attn_npu_3/mla/`，各模块的实现状态见 [plan](plan.md)。

| 固定源文件 / 函数 | CATLASS 层次与迁移位置 | 必须保留的行为 |
| --- | --- | --- |
| `op_host/arch22/fia_tiling_nonquant_mla.cpp`：`CalcInnerSize`、`CalcMBaseSize`、`Split`、`CalcMmResSize`、`CalcNormalWorkspaceSize` | `mla_schedule.h/.cpp`，host/AICPU 共用的无 gert 调度计算 | 切块、分核输入、workspace 尺寸及核数；不另写一套形状启发式 |
| `../common/op_host/split_core.cpp`：`SplitCore`、`CalcSplitInfo`、`CalcCostInfo`、`CalcSplitPlan`、`AssignByBatch/Row/Block`、`ForceAssign` | 调度模块私有的 `detail/cann_split_core.h/.cpp` | 枚举核数、cost、容差、遍历次序和相同 cost 时保留已有方案的规则；专用 MLA 使用此文件，不是 `split_core_v2.cpp`；保留源算法分支不表示开放对应功能 |
| `op_host/fused_infer_attention_score_tiling.h`；`op_kernel/arch22/fia_public_define.h`：`ConstInfo`、`RunInfo` | 调度字段与 Kernel 运行时参数 | 明确字段宽度、单位、端点语义和运行时状态，见 6.2 |
| `op_kernel/arch22/fused_infer_attention_score_v3.cpp`：`INVOKE_FIA_OP_GENERAL_IMPL` 的 `BSH_KVBSH_MLA_TILING` 分支 | 独立实例化单元、launcher 和生成器 | FP16/BF16、无 PA、无 FD、BSND 输入输出、C1V1/C1V2；不带入 FIA 注册与大签名 |
| `op_kernel/arch22/fia_kernel_nonquant_mla.h`：`Init`、`InitCalcParamsEach`、`CalcParams`、`FlashAttention`、`ExecuteTask` | `kernel/mla_dense.hpp` | 核区间解析、三项任务环、两级延迟流水、初始化及排空 |
| `op_kernel/arch22/fia_block_cube_nonquant_mla.h`：`ComputeMm1/2`、`ProcessMm1/2` | `block/mla_cube.hpp` | QK 内容/RoPE 排列、L1/L0 切分、FP32 累加、Fixpipe 写回和原子加 |
| `op_kernel/arch22/fia_block_vec_nonquant_mla.h`：`SoftmaxFlashV2Compute`、`AmlaVecCompute`、`ProcessAmlaNupdate`、`ProcessVec1L/2L`、`DealBmm2ResBaseBlock` | `block/mla_vector.hpp` | 源 softmax 与 aMLA 更新、舍入、尾列、行分工、O/LSE 写回；不是普通 online softmax 的替代实现 |
| `../common/op_kernel/matmul.h`：`MatmulKPP`、`LoadDataToL0A/B`；`buffers_policy.h` | `tile/` 与 Cube Block 的局部资源管理 | L0 双缓冲、加载布局、首 K 清零、unitFlag 和小矩阵 PIPE_M barrier |
| `../common/op_kernel/memcopy/attn_copy_gm_to_l1.h`：`CopyQueryGmToL1`、`CopyKvGmToL1` | `tile/mla_copy.hpp` | BSND 的 S1G 展平次序、ND→NZ、跨 query/head 的复制分段 |
| `op_kernel/arch22/memory_copy.h`：`CopyAttenOutUbToGm`、`DataCopySoftmaxLseBSND` | `block/mla_vector.hpp` 的 dense 写回 | 保留 O 的 query/head 首段、中段和尾段复制，以及 LSE 的 `[B,Hq,Sq]` 寻址 |
| `../common/op_kernel/vector_common.h`：`RowDivs`、`RowMuls`、`ComputeSoftMaxLse` | `tile/mla_vector_ops.hpp` | 保留行列 repeat 选择、广播布局、向量指令及 barrier 顺序 |

首步对应 C1V2 dense 非 FD key 为 FP16 `105000000010000001`、BF16 `105000000010022221`；C1V1 分别为 `105000000110000001`、`105000000110022221`。key 只用于核对来源，FA3 不暴露 CANN key。源 `FiaKernelNonQuantMla::FlashDecode` 与 `fia_block_vec_flashdecode_arch22.h` 的合并保留到 M10；192/128 的通用 `fia_kernel_nonquant.h` 保留到 M11，均不混入首步分派。

### 6.2 调度字段和首步边界

首个前向范围为 **512+64 共享 KV，FP16/BF16，连续 BSND，B>0，`1 <= Sq*Hq <= 256`，`1 <= Sk <= 512`，Hkv=1，无 mask，显式 `num_splits=1`**。输入允许连续视图的非零 storage offset；接通时检查尺寸乘积、地址及 workspace 溢出。先开放 `flash_attn_func`，varlen/kvcache 接入随对应步骤验收。首步暂不接受 metadata、实际长度、非连续外层 stride、softcap、非默认调度选项或梯度请求；真实、fake、自定义算子和 C++ 的边界一致。

这是实现里程碑的临时边界，不是缩小第 4 节的最终契约。`Sq*Hq>256`、多块/长 KV、零容量分别在 M04c 子步骤接通；TND/stride、PA、mask、split 和调度选项沿用 M06..M16。

源 `mBaseSize` 初值为 256。`CalcMBaseSize` 仅在等长、`(B*Hkv)%aicNum != 0` 时，以两者最小公倍数计算可均分的 S1G 块；候选须整除 `Sq*Hq` 且在 128..256，否则保持 256。不能改成按头数选择 8/16 行。`DoOpTiling` 的正常分支只调用一次 `Split`；首轮 `splitKVFlag_` 为 false，非 PA 的 `s2BaseSize=min(Sk,512)`，不在分核后擅自重跑 `CalcInnerSize`。首步每个 S1G 块只有一个 S2 基本块，源分核不拆开基本块，因而无需 FD；调度验收仍逐例检查 `numOfFdHead==0`。

| 源字段 / 计算 | 本仓归属与单位 | 首步处理及后续约束 |
| --- | --- | --- |
| `baseParams.bSize/n2Size/gSize/s1Size/s2Size/headDim/headDimRope` | `MlaSchedule` 的静态几何；元素/头/序列计数 | 分别映射 B、1、Hq、Sq、Sk、512、64；整型窄化前检查范围 |
| `innerSplitParams.mBaseSize/s2BaseSize` | 可复用调度，元素数 | 保留上述源计算；`gS1` 对 BSND 是 `s1*Hq+head`，不是 head-major |
| `outerSplitParams.bN2End/gS1End/s2End`、`usedCoreNum` | 每核任务区间的排他结束三元组；gS1/S2 为块索引 | 第 0 核从 `(0,0,0)` 起，其余从前核结束点起；`CalcParams` 才乘基本块变成元素索引 |
| 平台 `aicNum/aivNum/cvRatio`、`CalcNumBlocks`、`CalcScheduleMode` | 资源信息、启动维度、BATCH_MODE | 平台核数、实际使用核数与启动 blockDim 分开；保留 `CalcTschBlockDim` 对应关系及 C1V1/C1V2。源端点数组容量是 26，移植时显式校验容量，不把它写成芯片核数 |
| `workspaceParams.mm1ResSize/mm2ResSize` | 每核每槽的元素数，字节计算用宽整数 | `M=align16(min(Sq*Hq,mBaseSize))`，`N=align32(s2BaseSize)`，`R1=M*N`、`R2=M*512`；这里 N 按 32 个元素对齐，不是 32 字节 |
| `fdParams` 的任务、split、归约端点及 `fdAccumOutSize/fdLogSumExpSize` | split 调度与额外 workspace | 首步确认没有 FD 后使用非 FD 实例；保留调度计算结果供来源对照，不能把非零 FD 字段强行清零后宣称等价 |
| `scaleValue`、输入输出地址、实际 stride、workspace 地址 | `MlaLaunchParams`，每次前向传入 | scale 从源 tiling 中移出，仍在 QK 完成后的同一 FP32 Muls 位置使用；缺省 `1/sqrt(576)`，不写入可复用 metadata |
| `softmaxLseFlag/outputLayout/batchContinuous/needInit`、长度、PA、mask 标志 | 可复用的布局/功能条件；地址另传 | 首步固定连续 BSND、非空无 mask、无长度数组；按源保留 LSE 初始化及同步。其余字段在对应功能接通时映射，不当作未定义 padding |
| `RunInfo.loop/bn2IdxInCurCore/isFirstSInnerLoop/isLastS2Loop/isValid` | 每次 launch、每核私有的运行状态 | 不放进跨调用复用的 metadata；任务环、缓冲轮换和状态初始化属于 Kernel |

M04b 先使用无设备长度读取的静态调度模块；M05 将同一计算接入 AICPU 和版本化 int32 metadata，不维护第二套公式。普通 FA byte metadata 继续使用原协议。

MLA 的内部 dense metadata 由 [`mla_metadata.h`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_metadata.h) 定义：版本 1 为 401 个 int32 字，前 8 字保存 magic、版本、类型、总字数、dtype、生成状态、源调度状态和保留位，其后完整保存 `MlaSchedule` 的几何、分核端点、FD 字段、workspace 和资源需求。64 位字段按低/高 32 位保存，允许连续 tensor 的奇数 storage offset，不能窄化 workspace 大小。调度失败或 dtype 未支持时，记录非成功状态并清零整个调度载荷；消费者必须先检查头部再读取调度。scale、输入输出地址和可变 scratch 不进入可复用记录。

[`mla_metadata.aicpu`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_metadata.aicpu) 直接编译相同的调度、分核和序列化源码，输出后执行设备写屏障。AICPU 发射参数携带静态形状、平台资源、输出缓冲和可选的设备长度地址/数量；长度值在 AICPU 读取。M05a 的独立驱动按既有事件协议在当前 stream 与 AICPU stream 之间同步；M05b1 增加独立设备消费链路，M05b2b3b 再接通公开 dense 子集。生产构建单独链接该 AICPU 对象，并跟踪包含的头文件、`.cpp` 及编译选项以判断是否需要重编译。

M05b1 的 [`mla_metadata_consumer.h`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_metadata_consumer.h) 共用 CPU/设备解码与校验，[`kernel/mla_metadata.hpp`](../../csrc/ascend910/flash_attn_npu_3/mla/kernel/mla_metadata.hpp) 负责设备读取。先验证大小、头部、生成状态和 dtype，再检查调用方独立提供的形状、平台、实际 launch blockDim、workspace 容量和本地内存条件；分核端点须单调且完整覆盖任务。64 位 workspace 字段不窄化；通过后才构造 Kernel 调度并进入原计算与同步。校验失败不改写调度输出，不进入 attention Kernel。此检查保证兼容性和地址边界，不验证记录来源真实性；`maxCost`、未启用的 FD 记账和未使用核的端点不驱动计算。记录须在生产/消费 stream 依赖完成后保持只读，跨调用 scratch 独立。独立验收驱动仍在 host 准备启动维度和分配预算。诊断驱动让各核以独占缓存行报告校验状态，供拒绝路径的不写输出检查使用。M05b2b2 增加同一设备适配头中的 `LoadDenseMetadata`：所有 Cube/Vector 在进入 attention Kernel 前校验同一份只读记录，任意非成功状态执行 `AscendC::Trap()`，由正常调用或同步点报告运行时异常；成功路径不需要状态 tensor、D2H 或主机同步。该检查不受 `NDEBUG` 影响。Trap 会使当前执行上下文进入错误状态，异常验收须用独立进程；不承诺出错后继续复用该 stream。M05b2b3a 先验收生产内部链路，M05b2b3b 再接通公开生成与复用，最终公开 metadata E2E 在 M05c 验收。

M04b2c 的生产适配由 `mla_forward.cpp` 读取实际平台资源、调用同一 `BuildDenseSchedule`、分配 O/LSE 和每次调用独立的 workspace，并传递本次 scale 与地址。`mla_launch_impl.hpp` 通过生成器分别提供 FP16/BF16 C1V2 编译单元，按输入 dtype 分派。M04b3 将同一 Kernel / Block / Tile 按输入类型实例化，保持 M04b2b 的计算、资源和同步次序；aMLA 因子、P 和 O 使用各自输入 dtype，保留第 6.3 节的舍入规则。两种 dtype 均为 2 字节，共用同一调度和 workspace 布局。当前只开放 910B C1V2，其他设备及比例在相应实例验收前明确拒绝。首步公开入口为不带预分配输出的 `flash_attn_func`；内部 `out_` 也明确拒绝，避免绕过现有 functional custom op 的 alias 契约。

M05b2b3a 将该前向改为内部 AICPU 生成 v1 metadata，再由生产 FP16/BF16 launcher 的 `LoadDenseMetadata` 校验并消费。每次调用独立分配记录、O/LSE 和 workspace；沿用普通 FA3 的当前 stream/AICPU stream 双事件依赖，对 metadata 的 AICPU 使用登记 allocator stream，并由提交闭包保留全部 tensor。host 暂时仍调用同一 `BuildDenseSchedule`，仅为 launch blockDim、分配容量和独立校验 context 提供边界；设备实际使用的切块与分核端点全部来自记录。scale 继续作为当次前向参数。此步不开放显式 metadata，也不改变 kvcache/实际长度等拒绝边界；`flash_attn_func` 的官方签名不增加 metadata 参数。重复 host 调度及公开复用成本留给后续接入和 M05c。

M05b2b3b 开放官方 `get_scheduler_metadata` 的 64/512 dense 调度，以及 `flash_attn_with_kvcache` 的只读连续 BSND 子集：仍为 FP16/BF16、Hkv=1、B>0、`1<=Sq*Hq<=256`、KV 容量 `1..512`，无 mask、显式 `num_splits=1`。生成器没有 qv 参数；它只选择调度族，前向仍独立检查真实 qv、Q/K/V 和记录的 dtype/形状。追加 KV、TND/PA、非连续输入、梯度及其他未接通选项均明确拒绝。

生成器要求 NPU 上的连续 `int32[B]` cache_seqlens；只读前向接受 `None`（整容量）、等于容量的整数或同设备的连续 `int32[B]`，允许非零 storage offset。生成时 AICPU 校验全部 B 个长度，非整容量值写失败状态并清零 payload；消费时所有 Cube/Vector 在进入 attention 前再次检查当次传入的全部长度，避免显式复用旧记录时忽略长度变化。设备值非法由正常同步点报告 Trap；static dtype、device、rank、size、stride 由真实/fake/C++ 一致校验，不做长度 D2H 或主机等待。metadata 是快照，不保存输入长度地址；更新长度后须重算，未重算的记录不自动变化。

公开 metadata 仍为 v1 的 401 字 int32 tensor，可克隆或用连续偏移视图跨层复用，不依赖 Python 私有属性；消费前检查其类型、形状和设备，内容由现有消费者校验。显式复用不再发射 AICPU 生成；无 metadata 的 kvcache 和 `flash_attn_func` 继续内部生成。生成/消费各自登记 allocator stream，提交闭包保留输入、长度和记录；生成完成后由当前 stream 等待 AICPU，跨 stream 使用时调用方建立正常 stream/event 依赖。记录在消费期间保持只读，各前向独立分配 O/LSE/workspace；scale 按当次调用生效。公开 kvcache 的 LSE 返回值为四项，未分割时两个 accumulator 是与官方包装一致的空 tensor（默认 FP32），真实/fake 一致。重复 host 调度和调用成本留给 M05c；图捕获生命周期留给 M16。

M05c1 为上述只读 dense kvcache 增加内部 C++ dispatcher `_mla_dense_forward`，只接收 q/k/v/qv、当次 scale、长度和 metadata，返回 O/LSE。公共入口先识别 MLA 并拒绝未支持选项；真实 C++ 和 fake 实现分别执行张量、scale、长度及记录外形校验。Autograd 层透传到真实/fake 分派，两者都明确拒绝 requires_grad；不提供反向。请求 LSE 时公共包装再创建两个遵循 PyTorch 默认浮点 dtype 的空 accumulator。此入口复用原 `DenseForward` 的调度、分配、stream 和设备校验，不改变公开签名、metadata 协议或 Kernel；通用前向及普通 FA3 保持原分派。

M05c2 为同一 dense 子集增加内部 C++ dispatcher `_mla_dense_metadata`，只接收 batch、Q/KV 长度上界、query 头数、dtype 和设备 cache_seqlens。公开生成入口先检查 head dimensions 并拒绝未支持选项；真实/fake 分别校验形状、dtype 和长度张量外形，再复用原 AICPU 生成、事件依赖及 allocator 生命周期。真实实现也明确拒绝 CPU 长度；Autograd 层透传，浮点长度即使 requires_grad 也按 dtype 拒绝。公共签名、401 字记录和设备长度校验保持不变，普通 FA3 继续使用通用 custom op 和原 byte metadata。

只读 dense kvcache 的 NPU qv 路径先做表示识别及选项拒绝，再由 `_mla_dense_forward` 的真实/fake 实现完成张量、长度、metadata 和 scale 校验，避免在 Python 重复检查全部张量。缺省 scale 由已验证 64+512 表示的 dispatcher 提供。整数 cache_seqlens 在创建设备长度张量前仍检查几何和整容量值；其他表示、CPU 输入、TND/PA 继续走公共识别/拒绝路径。

CANN 9.0 的 Bisheng ASC launcher 未提供调度模式参数；生产扩展用仅在 MLA 发射期间生效的线程局部作用域，为实际 ACL launch 设置源 `BATCH_MODE`。封装保留其他 launch 属性和错误码，未进入该作用域的普通 FA3 发射原样透传；编译器绕过已验收 ACL 路径或单次作用域出现非一次发射时明确失败。构建为 MLA 编译单元额外生成设备预处理依赖，避免最终 host `-MMD` 漏掉 `__NPU_ARCH__` 内的 Kernel 头文件。

`BuildDenseSchedule` 的平台输入携带核数、实际可用 L1/UB/L0 容量、libapi 预留和平台 `CalcTschBlockDim(c*cvRatio,c,c*cvRatio)` 的结果表；调度通过实际使用核数索引该表，避免在无 gert 模块中猜测芯片启动维度。输出采用固定容量数组，workspace 字段保留元素与字节单位；几何、源 uint32 最小公倍数/块数、资源和 uint64 workspace 加法检查失败时不改写调用方的原调度。

源 `SetSplitOutput` 固定读取 `usedCoreNum*2` 项 FD Vector 端点，而 C1V1 的源 vector 只有 `aicNum*cvRatio` 项。本仓按 vector 实际长度复制到 52 项容量数组，未用项初始化为零；CPU oracle 比较原 `SplitCore` 的完整输出 vector，不执行该可能越界的序列化循环。此调整不改变分核算法、FD 判定或设备计算。

M04c1 将上述 dense 范围扩展到 S1G 多块，`Sq*Hq>256` 使用同一个源 `CalcMBaseSize` / `SplitCore` 和 Kernel 三项任务环，不重新定义切块策略；KV 容量仍为 `1..512`，每个 M 块只有一个 S2 块且不启用 FD。`mBaseSize` 可以是 128..256 间任意源候选值（包括 129/255），非 256 候选须整除总行数；metadata 消费者据此检查基本块、完整端点和 workspace，401 字的版本 1 协议不变。公开 `flash_attn_func`、只读 kvcache 和 metadata 的真实/fake/C++ 边界同步扩展。

源字段和寻址仍有资源上界，均在窄化或分配前检查：`Sq<=2^30`（LSE DMA 字节 stride）、`Hq<=floor((2^32-1)/512)`（query 元素 stride）、`Sq*Hq<=2^32-256`（uint32 ceil-div 的 255 行余量），以及 `B*Sq*Hq<=floor((2^63-1)/1024)`（输出张量字节数）。调度继续检查源 uint32 LCM、总 M 任务数、平台资源及 workspace 溢出。这些是实现字段/地址限制，不是官方 FA3 的接口限制。

M04c2 将 KV 容量扩展到 `1..1024`，保留源 `s2BaseSize=min(Sk,512)`；513..1024 的每个 M 块顺序计算两个 S2 块。`num_splits=1` 禁止跨核分割 KV：先运行原 `SplitCore`，无 FD 时原样保留；原方案有 FD 时，将每个源端点向上对齐到所在 M 块结束处，合并因此产生的空区间，重新构造所有 `s2End=0` 的完整 M 分区。按源 `CalcBatchCache/CalcS1GCache` 的正常块和尾块成本重新计算各核及最大成本。只筛除原 FD 候选会丢弃原本均衡的分区，端点映射以源选中的分区为起点，在完整 M 块约束下保留非空分区。这是 FA3 禁止 split 的必要调度映射；源基本块、成本、原候选算法、设备数值和事件顺序均不变，不清零已有 FD 方案来伪造非 FD 记录。

metadata 仍为 v1 的 401 字记录，消费者按 S2 基块计算 QK/P workspace，不按完整 KV 容量分配；端点只允许完整 M 块，总物理块数 `B*ceil(Sq*Hq/mBaseSize)*ceil(Sk/s2BaseSize)` 须适合源 uint32。真实/fake/C++ 同步接受该 dense 范围。更长 KV、自动/显式多 split 和其他未接通组合继续拒绝。

### 6.3 数值路径与资源布局

QK 的 L1 循环顺序为 N→K→M，M/N 子块均为 128。576 维并非先算完整内容再算完整 RoPE：两组 288 依次为 `[content 0:256, rope 0:32]`、`[rope 32:64, content 256:512]`，每组按 K=96 的 L0 片累加到 FP32 L0C。Q 的 64 维 RoPE 一次复制跨越相邻 Q/P 槽，尾 M 用 padding 偏移保持相邻关系；复制前必须同时取得两个槽的释放事件。最终 QK 以 FP32 ND 写入 GM，再乘 scale。

Vector 保留 `SoftmaxFlashV2<float,...WITHOUT_BRC>` 和其动态 tiling；随后执行 `AmlaVecCompute`：计算 `round(-max/ln2)`、指数缩放因子、FP16/BF16 因子往返转换及 `cof/eps` 修正。P 在缩放后以 `CAST_ROUND` 转成输入 dtype。跨 S2 块的 `nUpdate` 含 `-30` 截断、`1.5` 修正、`1e-6` 和 `2^23`，以 int32 原子加更新 FP32 PV 缓冲的位表示；不能替换为常规 `O *= exp(old_max-new_max)`。

PV 同样按 N→K→M 循环，M/N 为 128、K 的 L1/L0 片分别为 256/128。首 S2 块覆盖写 FP32 GM，后续块在 nUpdate 完成后以 FP32 原子加写回。V2 每块消费完成事件，仅在最后 S2 块写输出：保留源对 `abs(PV)>1e10` 的清零、除以 `aMlaSum`、FP16 `CAST_ROUND` / BF16 `CAST_RINT`；LSE 则从原 softmax 的 max/sum 计算自然对数。单块范围也经过 aMLA 因子转换，不能省略这段数值路径。

| 资源 | 固定源布局与生命周期 |
| --- | --- |
| L1 | 每槽 `128*288*2=72 KiB`，Q/P 四槽与 K/V 三槽共 504 KiB；QK 与 PV 共用，迭代器跨任务延续。Q/P 在各自首个 N 子块加载，后续 N 复用，最后 N 才释放 |
| L0A / L0B / L0C | 分别为 `2*32 / 2*32 / 2*64 KiB`；A/B 共同跟随 K 片，C 轮换与 M/N 循环绑定；首 K 清零，最终 K/写回以 unitFlag=3 配对，其余累加用 2 |
| Vector UB | input1 `2*32 KiB`、input2 `2*8 KiB`、output1 `32 KiB`、output2 C1V2/C1V1 为 `4/8 KiB`、tmp1 `32 KiB`、mask tmp `8 KiB`；n/cof/aMlaSum/max/exp/sum 各 `2*1 KiB`，默认 max/sum 各 `1 KiB`，合计 `170/174 KiB`。即使无 mask 也保留源分区复用 |
| GM workspace | Kernel 顺序为双槽 QK FP32、P 输入 dtype、PV FP32、vec2 FP32。QK/P 用 `loop%2`，PV 用 `bn2IdxInCurCore%2`（每个新 S1G 任务递增），不是统一使用 tile 序号；调用间不得共用可变 scratch |
| workspace 预算 | 源 host 普通部分为 `2*aicNum*(6*R1+8*R2+8*mBaseSize)` 字节，包含 nUpdate/sum 的额外预留；设备按 `GetBlockNum()` 排四个段，nUpdate 实际在 UB 生成并操作 PV 的 int32 别名。首步保留预算与段界，不把预留当成新的 GM 数据流；框架 `libapiSize` 与用户 workspace 单列 |
| 任务 / 行分工 | 三项 `RunInfo` 环、两槽 GM；外层 M 基块最多 256。C1V2 将 M 按 16 行边界分给两条 Vector，M<=16 时第二条零行仍参与事件；C1V1 全部行由一条 Vector 处理，V1 内层每次至多 128 行 |

### 6.4 同步依赖与组件复用结论

`ExecuteTask(t)` 在 Cube 上依次发出 `QK(t)`、`PV(t-1)`；Vector 上依次执行 `V1(t-1)`、`V2(t-2)`。创建结束后仍须排空任务环。下表均为源 mode 2 跨 Cube/Vector 事件，编号不与本地 HardEvent 混为一个命名空间。

| 源事件 | 生产 → 消费 | 保护的数据 / 时机 |
| --- | --- | --- |
| `syncC1V1=7` | Cube `PIPE_FIX` → V1 wait | QK 的 FP32 GM 写回完成 |
| `syncV1C2=8` | V1 `PIPE_MTE3` → Cube wait | P 写回完成，PV 可以加载 |
| `syncC2V1=4` | Cube `PIPE_FIX` → 下一次 V1 nUpdate wait | 前次 PV 完成后才能修改其位表示；Cube 初始化先发一个 token，Vector 结束回收最后一个 |
| `syncV1NupdateC2=5` | V1 `PIPE_MTE3` → PV 首次 Fixpipe 前 wait | nUpdate 原子加完成；首块跳过 nUpdate 计算也必须发/收事件 |
| `syncC2V2=9` | Cube `PIPE_FIX` → V2 wait | PV 写回完成；V2 在非末块也消费事件 |

`syncV0C1=6` 在源常量中存在，但此非量化 dense 路径不使用。L1 Q/P 的本地事件为 2/3/4/5，K/V 为 6/7/1，保留 `MTE1_MTE2` 释放和 `MTE2_MTE1` 就绪配对；L0 的 `M_MTE1` / `MTE1_M` 与 unitFlag 共同约束覆盖写。Vector 保留 input 的 `V_MTE2` / `MTE2_V` 和 output 的 `MTE3_V` / `V_MTE3`；`AllocEventID/FreeEventID` 和零行同步不能被提前返回绕过。启用 LSE 时，源 `InitOutputSingleCore` 的初始化与 `SyncAll` 也在首步执行链内。

固定 CATLASS 的复用结论如下；源码核对仅确定可用条件，设备等价性仍须在实现步验证。

| 组件 | 核对结论 |
| --- | --- |
| `gemm/tile/tile_mmad.hpp::TileMmad` | 优先复用。FP16/BF16、M/N 已向 16 对齐时，可透传源的 K、initC、unitFlag，且包含同样的 `(M/16)*(N/16)<10` PIPE_M barrier；源 `MatmulKPP` 的 K 循环和加载/释放时机仍由本仓 Block 保留 |
| `gemm/tile/atlasa2/copy_l0c_to_gm.hpp::CopyL0CToGm` | FP32、RowMajor、无量化/无 ReLU 的特化可映射源 Fixpipe 的 M/N、src/dst stride 和 unitFlag；原子模式由源 Block 在同一位置开关，不能交给默认 epilogue |
| GM→L1、L1→L0 Tile | 在本仓封装源分段复制与加载，逐项匹配 NZ/ZN、转置、尾块和 stride 后再复用。`CopyGmToL1DynamicOptimized` 对小 M 改走 DataCopy，不能直接代替源 ND→NZ；Q 的跨槽 RoPE 复制也需专门保留 |
| `epilogue/block/block_epilogue_amla_tp1_softmax.hpp` / `block_epilogue_amla_tp1_rescale_o.hpp` | 同有 aMLA 思路，但使用自有归约、UB/事件和部分不同 Cast 模式；不能整体替换源 Vector。首步在本仓按源封装 `SoftmaxFlashV2Compute/AmlaVecCompute` 等函数 |
| `arch/resource.hpp`、`arch/cross_core_sync.hpp` | 可复用资源/简单 flag 表达；Resource 构造会 `pipe.Destroy()`，必须统一资源所有权，不能与另一套 TPipe 分配混用。带 reverse-depth 的同步封装会增加等待，首步保留源事件协议 |

### 6.5 必要的 FA3 差异与验收切面

首步只调整第 3 节的 q/qv/K/V 指针映射、公开返回结构、scale 默认值与参数存放位置，以及去除 gert/算子注册依赖；保持 O 的 BSND 写回和 LSE 的 `[B,Hq,Sq]` 寻址。`num_splits=1` 在此单 S2 块范围与源非 FD 结果一致；以后长 KV 可能被 CANN 自动分核到 FD，必须单列 FA3 禁止 split 的调度约束，不能将源 FD 结果直接套入非 FD kernel。mask/空行符号和设备长度解析在对应步骤按第 3 节适配。

调度的独立验收比较源 `SplitCore` 输出的基本块、端点数组、使用核数、FD 标志和 workspace，oracle 从固定 commit 读取，不能调用待测调度实现。覆盖单/多 batch、平台核数及其两侧、M 的 16/128/256 边界、KV 的 16/32/128/256/512 边界和两种 CV 比例。无实际长度数组的步骤无需读取 NPU tensor；后续 AICPU 必须复用同一计算。

首个 kernel 的精度用 D01/D02 加上述边界、内容两半与 RoPE 两半的独立扰动、非零 storage offset、默认/零/负/显式 scale、交错形状重复入队核对 O/LSE。性能代表固定为 D01（默认及 0.23）、D02、`(B,Sq,Hq,Sk)=(1,1,128,512)` 与 `(2,2,128,511)`，两种 dtype 按接通进度验收，保存同轮 CANN 分派、kernel 与公开调用 E2E。M04b 无 metadata 的数据仅作诊断；M05 在同组代表用例验收第 7 节的 metadata E2E 门槛。

M04c1 增加 257/511/512/513 行、非整齐 M 基块、跨 query/head 和 batch 的分核端点，以及每核超过三项任务的流水覆盖；两种 dtype 的 O/LSE 对照固定源并通过独立数学 reference。性能在原五组之外固定增加 `(B,Sq,Hq,Sk)=(1,257,1,257)`、`(2,7,73,65)`、`(1,4,128,512)`、`(2,171,3,511)`、`(10,3,86,65)` 和 `(1,129,128,33)` 六组默认 scale。固定 CANN 主机入口将 Decode MLA 的 Hq 限于 1..128，故 513 行和 M=129 的性能用例分别采用 `171*3` 和 `3*86`；这不限制 FA3 的 Hq，更多头数由固定源 Kernel 与数学对照验收。metadata 复用、生成加一次/四次前向均按第 7 节逐例比较，采样前固定 CPU 亲和性并在同一进程应用于双方。

M04c2 增加 KV 513/527/528/529/543/544/545/769/1023/1024、M 尾块、跨块最大值变化（含超过 aMLA `-30` 截断）、内容/RoPE 分量、重复执行和多流 metadata 复用。固定源 Kernel 使用相同的无 FD 调度，O/LSE 要求逐元素一致，再通过独立双 reference。原生 CANN 仍保留自动 FD 分派；当其数值顺序不同，两边分别按既定精度准则对照数学 reference，不能要求与非 FD 逐元素相同，也不能修改原生 CANN 分派作为性能基线。

在原 11 组性能用例之外，采样前固定增加 `(2,1,8,513)` 默认/0.23、`(1,7,16,769)`、`(1,1,128,1024)`、`(2,2,128,1023)`、`(10,3,86,769)` 和 `(21,171,3,1024)`，共七组。沿用双 dtype、预热 5 次、五轮各 20 个 wall 样本、R=1/4、相同 CPU 亲和性及逐例 1.0 门槛；单列 metadata 生成、原生 CANN 分派和 attention Kernel 时间。

## 7. 验收口径

**源实现对照：** 每个迁移模块须能追溯到固定 CANN 的文件/函数，注明结构重构和必要接口适配的对应关系。风格重构须经过构建、精度、边界、重复执行和相关性能检查；迁移结果以本仓实际构建和测试为依据。保留源级执行行为不承诺编译后二进制或耗时完全相同。

**功能和精度：** 对同一逻辑输入同时检查独立高精度 reference、低精度 PyTorch 对照和原 CANN。CANN 对照先统一 scale、mask、布局与共享 KV 语义；本次按 FA3 调整的差异以官方为准，不要求重现源不兼容行为。reference 不能调用待测 kernel，也不能在两个实现中共用同一个错误的长度/页表解析。复用仓库双 reference 精度准则，单独检查 LSE、inf 符号、NaN、空行和 padding；阈值在基线步骤固定，不为通过用例而放宽。

**性能：** 用户明确的“不劣化”指 **走 MLA metadata 路径的 FA3 公开调用端到端（E2E）延迟**。在同一健康、空闲的 910B、同一输入/精度/软件栈上比较。固定 seed、预热、计时次数和同步方式，交替重复测量 CANN/新实现；保存每例耗时及运行信息。验收目标是每个约定代表用例 `FA3 metadata 路径 E2E / 原 CANN E2E <= 1.0`，不默认引入 5% 或其他放宽阈值，也不能用平均加速掩盖退化用例。噪声未能分辨时补测，不能宣布不劣化。

E2E 从公开 Python 入口开始，测至对应调用段的 O/LSE 完成，计入入口校验、调度、输出/workspace 分配、必要转换、发射、流间依赖和全部设备计算。输入 H2D 与一次性视图准备沿用基线的对称计时边界；两边使用一致的完成同步方式。只测内部 launcher、device event span 或 profiler kernel 时间均不能替代该口径。

每个代表用例须在采集前固定 metadata 生命周期和复用次数。预生成后复用时，记录复用前向的 E2E，并单列 metadata 生成时间；实际调用段需要生成或更新 metadata 时，从 `get_scheduler_metadata` 开始计时。跨层复用须记录实际次数 R、包含一次生成及 R 次前向的完整调用段 E2E 和摊销值，并与 CANN 完成相同 R 次前向比较，不能任意增加 R 隐藏准备成本。生成加一次前向与复用前向分别报告，不能将复用数据标为包含生成成本。

kernel 时间及无 metadata 调用时间作为瓶颈诊断数据，性能验收使用上述 metadata E2E 门槛。实现按源对照完成 CATLASS 分层、FA3 接入和 metadata 支持后进行对应验收；E2E 优化限定在保持源设备行为的接口、调度结果生成/复用、重复准备和发射开销。设备计算瓶颈按源实现对照定位。

分别报告 kernel 时间、无 metadata 调用 E2E、metadata 生成时间、复用前向 E2E 及上述完整调用段 E2E；图模式单列。接口适配导致的必要转换计入相应端到端成本，CANN 基线使用固定版本的原生正确分派。对于原 CANN 的原生 NZ 等不同布局优势，另列原生最佳路径结果并解释表示差异，转换成本单独明确。

**覆盖：** 两种表示 × FP16/BF16 × dense/TND/PA × mask × 长短 KV × 单/多 query × 分组/尾块/stride；加 metadata 有无、重复使用、不同 scale、长度变更、split 自动/显式、graph replay。可采用代表性组合加针对性边界测试，不要求机械的全笛卡尔积，但每项源能力都要能追溯到用例或明确排除理由。

**版本可追溯：** CANN 源码 commit 不代表当前已安装算子版本。实测前记录实际生效的 OPP/算子库路径、构建版本或文件哈希，确认其对应本次来源；记录本仓扩展的实际加载路径与构建版本。910C 需要实现与资源配置兼容，按需求不作 910C 实机验收承诺。
