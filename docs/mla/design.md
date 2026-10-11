# CANN FIA MLA → FA3 NPU：契约与架构

本文保存已确认的迁移契约和实现依据。**第 1–4 节定义要迁什么，第 5–6 节说明记录与代码如何组织，第 7 节定义怎样验收。** 当前开放能力、运行结果和环境只在 [status](status.md) 维护，实施顺序和完成条件只在 [plan](plan.md) 维护；本文的目标能力不等于已接通能力。

| 核对内容 | 阅读位置 |
| --- | --- |
| 来源、迁移范围与排除项 | [固定依据](#1-范围与固定依据)、[功能边界](#4-功能边界) |
| 参数、布局、长度与输出 | [接口映射](#3-两种-mla-的接口映射)、[非分页入口分派](#671-非分页入口与记录选择)、[代表用例](#42-代表用例与参考实现边界) |
| 记录格式、校验与复用 | [metadata 协议](#5-metadata-设计)、[共同调度约束](#62-调度字段与边界) |
| 分配前缀、used 与真实地址 | [长度与布局](#63-长度与布局调度)、[stride 与 storage](#64-输入-stride-与地址校验) |
| 分页缓存的目标、地址与设备校验边界 | [PA 契约与来源](#643-nd-paged-cache-契约与来源)、[v6 协议与消费者](#57-v6nd-pa-记录) |
| Kernel / Block / Tile 与生产接入 | [代码结构](#6-代码结构)、[数值路径](#65-数值路径与资源布局)、[同步依赖](#66-同步依赖与组件复用结论)、[接入与构建](#67-生产接入与构建) |
| 精度、性能与证据 | [验收口径](#7-验收口径)、[参考工具边界](#72-参考工具与对照边界) |

术语：BSND 为 `[B,S,H,D]`，TND 为 `[T,H,D]`；“分配长度”决定物理起点，“used”限定其中参与计算的前缀。PA 表示 paged attention；FD 表示 FlashDecode/split-KV。下文的“独立组件/探针”有自己的验收边界，不能据其通过推断公开前向已开放。

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
| PA | `page_table:int32[B, max_pages]`，cache 为 `[num_pages, page_size, Hkv, D]`；携带页表行 stride 和 K、V 各自 stride，处理乱序页、尾页及首维非连续；目标契约与固定源映射见 [6.4.3](#643-nd-paged-cache-契约与来源) |
| 头数 | decode 的迁移范围 Hkv=1；prefill 保留 CANN 与 FA3 共同支持的 Hq/Hkv 整除组合。不是把整套普通 GQA 移过来 |
| mask | 对外只用 `causal`、`window_size`；右下对齐的位置中心为 `i + Sk - Sq`。CANN 0/3/4 模式归一为无 mask/右下 causal/band |
| 窗口归一化 | 固定 FA3 版本在入口按 `Sk-1`、`Sq-1` 处理左右窗口，并有单 query 分支；数学窗口与实际归一化分支都要测试，不能只验证等长序列 |
| 输出 | O 与 V 的末维相同、dtype 与 FP16/BF16 输入相同；LSE 为 FP32，dense `[B,Hq,Sq]`、varlen `[Hq,Tq]`，数学量为自然对数 |
| 空行/空 KV | O 为 0。非分页的整个 K 张量容量为 0 时，`mha_fwd` host 分支令 LSE 为 `+inf`；容量非零但某行全被 mask 或样本实际 KV 长度为 0 时，`softmax::finalize` / `epilogue::store_zero` 写 `-inf`，split 全空合并也为 `-inf`。零 query 的 O/LSE 保留空 shape；PA 须按 [6.4.3](#643-nd-paged-cache-契约与来源) 核对官方分支，不能按物理 cache 的空 shape 套用非分页规则 |
| 返回值 | 按入口对齐：dense/varlen 的可选 LSE 返回二元组；固定版本 kvcache 开启 `return_softmax_lse` 实际返回 O、LSE 及两个 accumulator，共四项，不能只按旧 docstring 实现 |
| split | `num_splits=0` 自动、1 不分割、>1 请求分割，固定官方实现上限为 256；核数限制属于实现资源策略，不能把现仓仅 PA+TND、至多核数的限制误称为 FA3 限制 |

## 4. 功能边界

| 能力 | 处理与验收 |
| --- | --- |
| FP16/BF16；512+64 压缩表示；192/128 prefill | 必迁，含 out 和 LSE |
| dense、TND、实际长度、GQA 分组（prefill） | 必迁；含非均匀长度、多 token、小 batch、尾块及零长度边界 |
| PA、16 对齐且不超过 1024 的源页大小、合法 stride | 必迁 ND 路径；CANN `GetBlockSize` / `CheckFeatureBlockSize` 要求 16..1024 的 16 倍数；固定 FA3 允许任意正页大小，TMA 整除条件只选择实现。因此交集保留全部 64 种源页大小，不额外限制为 128/256 的倍数 |
| 无 mask、右下 causal、有限/单侧窗口、FlashDecode | 必迁；包含功能组合，不能仅验证每个开关单独可用 |
| metadata、自动/显式调度、重复执行、图执行 | 本需求必需；普通 FA3 的既有路径需回归 |
| CANN 独立 512 维内容 K/V | FA3 qv 无等价表达，排除；见 3.1 |
| tree mask / sparseMode=9、任意用户 mask、独立左上 causal 接口 | FA3 hopper 公共入口没有等价参数，排除；保留能用标准右下窗口表达的计算 |
| 外部 NZ cache、CANN 布局字符串和特殊输出布局 | 不增加 CANN 专用公开参数；普通布局可由视图归一，NZ 外部格式按 FA3 不支持排除。内部是否使用分块布局由性能决定 |
| INT8 全量化、伪量化、后量化、CANN 专用量化因子 | 不迁移；不以 FA3 的 FP8 descale 冒充等价支持；950 的量化 MLA 不在芯片范围内 |
| PSE、公共前缀、TensorList、padding 参数、sink | 按 4.1 的 X05–X07 排除；TensorList/PSE/公共前缀/左 padding 不在所取源 MLA 能力内，源 prefill 可接受 sink，但固定 FA3 无等价参数。有效区末尾 padding 仍纳入实际长度验收 |
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

所有数值用例使用 FP16 和 BF16。`D` 表示共享 512+64，`P` 表示 192/128；除 D 的 Hkv=1 外，头数写为 Hq/Hkv。未特别标注时无 mask、默认 scale，两个表示均校验 O/LSE。下表覆盖已接通和待接通功能的目标验收输入，**不是已运行 NPU 测试列表**；无意义或源拒绝的交叉组合须记录原因，不以 skip 冒充覆盖。

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

代表用例定义输入与预期行为；参考实现、源执行器各自覆盖到哪里，见 [7.2](#72-参考工具与对照边界)。

## 5. metadata 设计

metadata 指 FA3 的 `get_scheduler_metadata` → `scheduler_metadata` 调用链，以及 func/varlen 的内部生成链路。兼容公开调用语义，NPU 记录布局由本后端管理，不与 CUDA buffer 互换。B 表示 batch，以下字数单位均为 int32。记录中的 token 起点属于分配坐标，不是元素或字节地址；真实地址还须结合各输入的 stride，见 [6.4](#64-输入-stride-与地址校验)。

| 协议 | version / kind | 精确字数 | 表达的布局与长度 |
| --- | --- | --- | --- |
| [v1](#52-v1整容量-dense-记录与消费者) | 1 / 1 | 401 | BSND Q/KV 均使用整容量 |
| [v2](#53-v2bsnd-实际-kv-长度记录与消费者) | 2 / 2 | 401+B | BSND Q 使用整容量，KV 保存逐样本实际长度 |
| [v3](#54-v3紧凑-tnd-记录与设备消费) | 3 / 3 | 405+4B | 紧凑 TND Q/KV；两轴起点来自分配前缀，使用整个分配段 |
| [v4](#55-v4tnd-q-与-bsnd-kv-混合记录) | 4 / 4 | 406+5B | TND Q 使用整个分配段，BSND KV 保存 64 位物理起点与实际长度 |
| [v5](#56-v5分配与-used-记录) | 5 / 5 | 410+8B | BSND Q/KV、TND Q/KV、TND Q + BSND KV；分别保存分配起点、容量与可选 Q/KV-used |
| [v6](#57-v6nd-pa-记录) | 6 / 6 | 408+5B | ND PA，BSND/TND Q 分配与可选 Q-used、非累计 KV 长度；公开生成、独立设备消费与 attention |

协议能表达某布局，不代表该布局已接通生产入口。生成记录、设备校验、独立 attention、入口调用分别验收；当前接入层次见 [status](status.md#当前可调用能力)，生产分派见 [6.7](#67-生产接入与构建)。v1–v4 均不表达独立 Q-used 或 TND KV-used；v5 复用的归一与调度见 [6.3.4](#634-分配长度与-used-的-cpu-归一)。

### 5.1 签名与复用规则

- 保留官方参数名、顺序与默认值。`get_scheduler_metadata` 没有 `qv`、`cu_seqlens_k` 或 `seqused_q` 参数；`flash_attn_func` / `flash_attn_varlen_func` 没有显式 metadata 参数，采用内部生成。显式复用通过只读 kvcache 接入，不增加公共参数。
- 非分页生成器必须接收设备连续 `int32[B]` 的 `cache_seqlens`。无 Q 前缀时生成 v2，`max_seqlen_q/k` 对应 BSND 分配容量；提供同设备连续 `int32[B+1]` 的 `cu_seqlens_q` 时生成 v4，`max_seqlen_q` 是每段 Q 上界，`max_seqlen_k` 是物理 cache 容量。v4 的 Q 总量由 AICPU 从前缀尾值推导，前向再与真实 Q shape 核对。
- `None` / `0..Sk` 整数便利输入只属于 kvcache。None 表示整容量；整数先在 host 校验再物化设备 tensor。混合路径的 None 也物化为设备长度，再使用 v4。显式记录仍须逐项核对当次前缀和长度，None 不表示信任记录快照。
- 记录保存静态几何、dtype、调度特征、切块、分核及长度快照，不保存地址、scale、stride 或可变 scratch。前向独立校验真实 qv 与 Q/K/V；softcap 接入后的数值同样由前向提供，不由 `has_softcap` 代替。
- 无记录时内部生成，显式复用时省去 AICPU 生成；两者共用源调度函数。host 只读 shape/dtype/stride 等静态信息，长度值由 AICPU/设备读取，不增加长度 D2H、`.item()` 或主机同步。host 分配预算仍可能计算调度，见 [6.7](#67-生产接入与构建)。
- 同设备、同调度条件和长度下可跨层复用；长度变化后重新生成。记录可克隆或用连续偏移视图复用，不依赖 Python 私有属性。每次调用独占 O/LSE/workspace；stream 和输入只读要求见 [5.8](#58-stream-与内存生命周期)。

普通 FA3 保留 `uint8` 协议。其旧数值参数 `softcap` / `softmax_scale` 为关键字兼容项，仍写入 byte 记录；单独 `has_softcap=True` 拒绝，须同时给出正 `softcap`。省略 `has_softcap` 时可由正 `softcap` 启用；True 与零值冲突、负值或非有限值均拒绝。MLA 不接受这两个生成器数值关键字。

**PA 生成接口与协议边界。** 上述 v1–v5 均不表达分页，不能把物理页当作 BSND batch，或只设置一个 PA 标志后复用旧记录。v6 保存 PA 类型、页大小、每样本逻辑容量、Q 分配/实际长度和 KV 实际长度及源调度；编码与独立调用 context 见 [5.7](#57-v6nd-pa-记录)。公开生成器 `get_scheduler_metadata(..., page_size=P)` 只知道逻辑几何和长度，既没有页表、物理页数，也没有 K/V storage 信息；生成器不校验或快照这些数据。P 须为 16..1024 的 16 倍数，`max_seqlen_k=C` 须在 0..16384 且可被 P 整除；`cache_seqlens` 仍必须是同设备连续 `int32[B]`，可选 `cu_seqlens_q` 是同设备连续 `int32[B+1]`。无 Q 前缀时 Q 使用 BSND 整容量，有前缀时使用 TND 分配段，总 Q 由 AICPU 推导。公开 PA 前向仍待单独接入；届时须核对记录的页大小与逻辑容量，再独立校验当次页表和物理 cache，见 [6.4.3](#643-nd-paged-cache-契约与来源)。

同长度、Q 分配和调度条件下，记录可跨合法页表映射、物理 cache 和 stride 复用；页表变化不要求重算调度，但每次消费仍检查有效页号。长度、Q 前缀或页大小改变则重生成；页表和 cache 在当次异步调用结束前只读。内部 Q-used 每次生成，不开放显式复用，也不新增公共参数。原 v1–v5 与普通 FA3 byte 协议保持兼容，PA 与非 PA 记录必须互相拒绝。

#### 共同编码与失败规则

[`mla_metadata.h`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_metadata.h) 定义共同前缀：`magic=0x4d4c4131`；前八字依次为 magic、version、kind、wordCount、dtype、生成状态、源调度状态、保留位。其后 393 字为 `MlaMetadataSchedule`，保存几何、端点、FD 记账、workspace 与局部资源需求。保留位为零；64 位字段拆为低/高 32 位，只要求 4 字节对齐，允许奇数 storage offset。

v2–v6 生成器先按静态 B 计算精确字数，要求 wordCount 不超过 `UINT32_MAX`，再检查输出地址、对齐、地址加法及与输入的区间重叠。不安全输出返回失败且不写记录；区间判断不做可能溢出的终点加法。只读输入可以相互别名，不能与输出重叠。输出安全但 dtype、描述符、形状、长度值或调度失败时，写非成功头部并清零八字头后的**全部载荷**，包括快照。v3–v6 成功记录从调度器拥有的快照序列化，不二次读取原始前缀或 used。

| 生成错误 | v1 / v2 | v3 | v4 / v5 / v6 |
| --- | --- | --- | --- |
| dtype 不支持 | `UnsupportedDtype=1` | 同左 | 同左 |
| 形状、平台或调度失败 | `ScheduleFailure=2`，另存调度状态 | 同左 | 同左 |
| 输入描述符错误 | `InvalidCacheLengths=3` | `InvalidPrefixes=3` | `InvalidLengths=3` |
| 长度数值错误 | v1 非整容量为 `UnsupportedCacheLength=4`；v2 越界为 `ScheduleFailure/UnsupportedShape` | `ScheduleFailure`，另存调度状态 | 同左 |

各协议消费者使用独立调用 context，先核对精确字数、版本/类型、生成状态、dtype，再校验形状、平台、实际 blockDim、workspace、局部资源、长度和端点。协议之间相互拒绝，不因 magic/dtype 相同就互换；64 位 workspace 和地址不能窄化。校验失败不改写返回调度。

设备适配层另检查输入地址、4 字节对齐、地址加法和编译 dtype。每个 Cube/Vector 在 TPipe、attention 同步和 O/LSE/workspace 写入之前完成校验；`Load*Metadata` 遇错执行不受 `NDEBUG` 影响的 `AscendC::Trap()`。**生成器写失败记录，消费者才触发设备异常**，由调用或同步点报告；异常用独立进程验收，不承诺错误上下文可继续使用。

消费者保证兼容性和计算地址边界，不认证记录来源或调度最优性；`maxCost` 仅作诊断。FD 记账及未使用核端点按各版本规则检查，例如 v5 要求未启用的数组和未用端点为零。空容量也不能绕过记录/长度校验；全局空容量见 [6.2.3](#623-全局空容量)，非空容量中的零长度按 [6.3](#63-长度与布局调度) 处理。

### 5.2 v1：整容量 dense 记录与消费者

[`BuildDenseMetadata`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_metadata.cpp) 生成固定 `int32[401]`，可选 KV 长度必须逐项等于容量，缺省使用整容量。v1 只用于无长度和记录的整容量前向，不由公开 metadata 生成器返回；生成失败清零调度载荷。

[`mla_metadata_consumer.h`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_metadata_consumer.h) 实现 CPU/设备共用校验，[`kernel/mla_metadata.hpp`](../../csrc/ascend910/flash_attn_npu_3/mla/kernel/mla_metadata.hpp) 提供设备适配。非空 dense 端点须单调并覆盖全部任务；`CheckDenseCacheLengths` 重新核对当次长度。通过后才构造 `MlaDenseKernelSchedule`。

### 5.3 v2：BSND 实际 KV 长度记录与消费者

[`mla_bsnd_metadata.h`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_bsnd_metadata.h) 在 401 字共同前缀后追加 B 个实际 KV 长度。隐式整容量也保存快照，B=0 时仍以版本/类型区分 v1；分配要求 `0<=B<=UINT32_MAX-401`。

[`BuildBsndKvLengthMetadata`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_bsnd_metadata.cpp) 接受 `[0,Sk]` 的长度，复用 `BuildBsndKvLengthSchedule`。非均匀、等长短序列和全零实际长度均按实际长度调度，workspace 按物理容量预算。[`mla_bsnd_metadata.aicpu`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_bsnd_metadata.aicpu) 编译同一调度与序列化源码。

[`mla_bsnd_metadata_consumer.h`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_bsnd_metadata_consumer.h) 将快照与当次长度逐项比较；缺省当次长度表示整容量。端点须单调且位于完整 M 块之间；省略尾部须从样本边界开始，且省略样本的 KV 长度全部为零。非空容量中的全零长度仍校验完整调度和 workspace。

[`kernel/mla_bsnd_metadata.hpp`](../../csrc/ascend910/flash_attn_npu_3/mla/kernel/mla_bsnd_metadata.hpp) 的 `LoadBsndKvLengthMetadata` 校验后，将快照交给 [实际长度 Kernel](#631-bsnd-实际长度与零尾部)。记录 blockDim 必须匹配实际发射；可在生成时将平台 blockDims 固定为全核，以免长度更新后重生成/消费需要 D2H。

### 5.4 v3：紧凑 TND 记录与设备消费

[`mla_tnd_metadata.h`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_metadata.h) 的 401 字共同前缀后依次保存 Q/KV 总容量及 Q/KV max 上界，再为每个样本保存四字 `queryStart、kvStart、queryLength、kvLength`，单位为 token。调度 `s1Size/s2Size` 是观测最大段长；分配要求 `0<=B<=floor((UINT32_MAX-405)/4)`。

[`BuildTndMetadata`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_metadata.cpp) 要求两组连续 `int32[B+1]` 前缀，包括 B=0 时的两组 `[0]`；数值规则见 [6.3.2](#632-紧凑-tnd-解析与调度)。[`mla_tnd_metadata.aicpu`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_metadata.aicpu) 编译同一解析、调度和序列化源码。无 used 的 varlen 每次内部生成 v3；任一 used 存在时改用 v5。固定官方生成器无 `cu_seqlens_k` 参数，不提供公开显式 v3/v5 复用。

[`mla_tnd_metadata_consumer.h`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_metadata_consumer.h) 核对总容量、max 上界、观测最大段长和两组完整前缀。逐样本 M/S2 块数、端点和零成本后缀不能用矩形容量代替；跨空 Q 样本保留源 batch 坐标。末端所在及其后每个样本均须 Q=0 或 KV=0，端点可落在零 KV 样本的合法 M 块之间。

[`kernel/mla_tnd_metadata.hpp`](../../csrc/ascend910/flash_attn_npu_3/mla/kernel/mla_tnd_metadata.hpp) 的 `LoadTndMetadata` 校验三组输入（记录与两组前缀），返回独立 `MlaTndKernelSchedule`，防止误传 BSND Kernel；空容量仍核对全部前缀。

### 5.5 v4：TND Q 与 BSND KV 混合记录

[`mla_tnd_bsnd_metadata.h`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_bsnd_metadata.h) 的 0 起始索引 401..405 字依次保存 `totalQuery、maxQueryLength、kvCapacity、totalKvCapacity.low、totalKvCapacity.high`，物理 KV 总量为 64 位 `B*Sk`。每个样本五字为 `queryStart、kvStart.low、kvStart.high、queryLength、kvLength`，单位为 token。KV 起点始终为 `uint64(b)*Sk`；`s1Size` 为观测 Q 最大段长，`s2Size` 为物理 Sk，不能用实际 KV 长度替代。

[`TndBsndMetadataWords` / `BuildTndBsndMetadata`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_bsnd_metadata.cpp) 要求 `0<=B<=floor((UINT32_MAX-406)/5)`，接收 Q 前缀 `int32[B+1]` 和实际 KV 长度 `int32[B]`。B=0 仍要求 Q `[0]`，仅此时允许 KV 为 null/count=0；底层生成器的 null 不表示整容量。真实/fake/C++ 校验静态 max/head/dtype/设备/连续性；不按 `B*max_seqlen_q*Hq` 的矩形上界拒绝紧凑 Q，真实总 Q、前缀值、KV 长度和资源留给设备校验。

[`mla_tnd_bsnd_metadata.aicpu`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_bsnd_metadata.aicpu) 共用混合解析与序列化。生产 `ComputeMlaTndBsndMetadataFromPrefix` 在缓冲/描述符检查后从 Q 尾值推导总量；非空物理容量绑定全核 blockDim，空 batch/query/KV 绑定单校验核。独立 `ComputeMlaTndBsndMetadata` 保留显式总 Q 与调用方平台表的语义，二者使用同一 v4 格式和失败记录。

[`mla_tnd_bsnd_metadata_consumer.h`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_bsnd_metadata_consumer.h) 使用独立 context 和 `MlaTndBsndKernelSchedule`，核对 Q 总量/max 上界、Sk、64 位 `B*Sk`、完整 Q 前缀、当次 KV 长度及每个物理起点。Q 前缀/描述符错误为 `InvalidPrefixes=12`，KV 长度/物理起点/描述符错误为 `InvalidCacheLengths=11`。变长端点、资源和零成本后缀见 [6.3.3](#633-tnd-q-与-bsnd-kv-的混合调度)。

[`kernel/mla_tnd_bsnd_metadata.hpp`](../../csrc/ascend910/flash_attn_npu_3/mla/kernel/mla_tnd_bsnd_metadata.hpp) 的 `LoadTndBsndMetadata` 校验记录、Q 前缀和 KV 长度后，将五字 span 交给 `MlaTndBsnd`。独立驱动与生产 [`mla_tnd_bsnd_launch_impl.hpp`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_bsnd_launch_impl.hpp) 保留相同的 64 位 KV 寻址。

### 5.6 v5：分配与 used 记录

[`mla_used_metadata.h`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_used_metadata.h) 保留 401 字共同调度前缀，随后 Q/KV 各四字为 `layout、totalTokens.low、totalTokens.high、maxLength`；layout 的 0/1 分别表示 BSND/TND。第 409 字为 `usedFlags`，bit 0/1 分别表示 Q/KV-used 显式提供，其余位为零；空 batch 的非空地址、零计数数组仍是显式提供。每样本八字为 `queryStart.low/high、kvStart.low/high、queryCapacity、kvCapacity、queryLength、kvLength`，起点与容量均按物理分配，单位为 token。分配要求 `0<=B<=floor((UINT32_MAX-410)/8)`。

[`UsedMetadataWords` / `BuildUsedMetadata`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_used_metadata.cpp) 接收三种已约定布局以及两组分配前缀、两组可选 used。四组描述符与输入/输出区间均在读取前检查，随后调用 `BuildUsedSchedule`。缺省 used 使用整段分配，并保留源数组的累计/非累计语义；显式整段 used 不必与缺省 used 产生相同切块，因此记录同时保存 used 标记。Q/KV 任一轴为空仍完整校验其他轴，非空容量中的全零 used 保留调度与 workspace。

[`mla_used_metadata.aicpu`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_used_metadata.aicpu) 直接读取设备前缀/used，编译同一归一、调度和序列化源码。独立 [生成 probe](../../tests/mla_metadata_probe.cpp) 与生产 `UsedForward` 共用该生成器。v1–v4 消费者拒绝 v5。

[`ReadUsedMetadata`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_used_metadata_consumer.h) 使用独立 `MlaUsedMetadataContext`、当次四数组描述符及 `MlaUsedKernelSchedule`。先验证精确字数、v5 头部、生成状态和 dtype，再检查布局、总容量/max 上界、used 标记及全部描述符；值读取留在设备。逐样本对照当前分配前缀、两轴 64 位物理起点、分配长度和 used；缺省 used 必须等于完整分配段，B=0 的显式零长度数组仍须保留存在标记。解码失败保持返回结构不变。

调度校验复用既有平台、空调度、workspace 和局部资源检查，按实际 Q/KV 长度检查变长 M 端点及整个零成本后缀，并检查未用端点、非 FD 记账与总 tile 溢出；`maxCost` 仅作诊断，不验证划分最优性或记录真实性。全零 used 的非空分配仍须具有完整 workspace。设备 [`LoadUsedMetadata`](../../csrc/ascend910/flash_attn_npu_3/mla/kernel/mla_used_metadata.hpp) 在每个 Cube/Vector 核上校验当次 dtype、发射核数、地址/对齐与上述内容，失败执行 release `Trap`，调用方必须在通过后才分配事件或写输出。独立校验 probe、[attention probe](../../tests/mla_used_attention_probe.cpp)、[BSND used launcher](../../csrc/ascend910/flash_attn_npu_3/mla/mla_bsnd_used_launch_impl.hpp) 和 [TND used launcher](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_used_launch_impl.hpp) 共用该校验。纯参数结构位于 `mla_used_metadata.h`，host launcher 不包含设备校验函数。

[`MlaUsed`](../../csrc/ascend910/flash_attn_npu_3/mla/kernel/mla_used.hpp) 按已校验的 Q 布局选择 BSND/TND 输出，并将八字 span 交给既有 `MlaNonPaged` 任务环。实际 Q 决定 M 块边界，实际 KV 决定 S2 循环；任一实际长度为零时不读取该样本的 Q/KV 数据。TND 两轴读取分配前缀起点，BSND 的逻辑 token 起点为 64 位 `b*capacity`，输入元素地址由连续布局或独立 stride 描述计算；缩短 used 不压紧后续样本。Block/Tile 的数值和事件顺序不变；全局空 KV 初始化分配区域为 `O=0、LSE=+inf`，非空容量中的零 KV 有效行使用 `O=0、LSE=-inf`。未使用 query 槽位不新增输出承诺。

### 5.7 v6：ND PA 记录

[`mla_paged_metadata.h`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_paged_metadata.h) 固化 `version=6、kind=6` 与精确 `408+5B` 字；要求 `0<=B<=floor((UINT32_MAX-408)/5)`。前 401 字沿用共同编码，第 401..404 字为 Q 的 `layout、totalTokens.low/high、maxLength`，layout 0/1 表示 BSND/TND；第 405 字仅 bit 0 表示显式 Q-used，第 406/407 字分别是 P 与逻辑容量 C。P/C 约束沿用 [PA CPU 调度](#643-nd-paged-cache-契约与来源)，源 `schedule.s2Size=ceil(max(KV length)/P)*P` 独立保存在共同调度中。

每个样本五字为 `queryStart.low/high、queryCapacity、queryLength、kvLength`，单位为 token。Q 起点按原分配，BSND 支持超过 uint32 的起点，TND 保存分配前缀且不随 used 缩短而移动；`maxLength` 是 BSND 分配容量或 TND 前缀上界，源 `s1Size` 对 TND 仍是观测最大分配段长。记录没有密集 KV 起点、页表、物理页数、stride、地址或 scale；这些信息不影响同条件下的 metadata 复用。

[`PagedMetadataWords` / `BuildPagedMetadata`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_paged_metadata.cpp) 由 CPU 与 AICPU 生成器共用，复用 `BuildPagedSchedule` 与 `SerializeMlaSchedule`。三组输入为 TND 的 Q 分配前缀、可选 Q-used 和必需的逐样本 KV 长度；BSND 不接收 Q 前缀。B=0 的 TND 仍要求 `[0]`，仅 B=0 允许 null/count=0 的 KV；非空地址且零计数的 Q-used 仍保存存在标记。描述符、区间重叠、失败清零和 4 字节对齐遵循共同规则；成功载荷来自调度器拥有的快照。Q 分配/used 为空也必须校验 KV，所有 KV 长度为零时保留长度快照和零 workspace 校验调度；不据此定义 PA 设备空行输出。

[`ComputeMlaPagedMetadata`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_paged_metadata.aicpu) 使用独立 launch ABI 的三组设备地址/计数，调用上述序列化器的 `fromQueryPrefix=true` 分支：TND 总 Q 在输出区间、dtype 和全部长度描述符校验后，从 Q 前缀尾值推导，忽略 launch 中的 TND total；BSND 仍校验显式分配总量。初始零、单调、分配上界、used 和 KV 值继续由共同调度器检查，不读回 host。成功或失败记录写完后执行 `DataStoreBarrier`；不安全输出及空 launch 参数返回 1 且不写记录。

`MlaPagedMetadataContext` 固化调用方独立提供的 dtype、B、Hq、Q 分配、used 标记、P/C、平台核数、实际 blockDim、workspace/libapi 和局部资源；当前长度描述符仍由当次调用提供，不能从记录反推调用条件。物理页池、页表及 stride 由独立的 `MlaPagedCacheInputs` 当次描述提供，不序列化进 v6。

[CPU 协议测试](../../tests/test_flash_attn_npu_v3_mla_paged_metadata.py) 用独立 Python 解码规范与固定源 PA parser/tiling/SplitCore 逐字比较完整记录、端点和资源；测试解码器还检查逐字损坏与调用条件变化，旧 v1–v5 CPU 消费者在读载荷前拒绝 v6。这里的严格源字段一致性包含诊断 `maxCost`，不要求设备消费者认证源划分最优性。测试解码器和实际设备消费者各自验收；Copy、attention 与公开接入的完成情况见 [status](status.md#当前可调用能力)。

[独立 AICPU 测试](../../tests/test_flash_attn_npu_v3_mla_paged_metadata_aicpu.py) 在 910B 比较 CPU 与固定源完整记录，检查 prefix/used/KV 更新、双事件依赖、allocator stream 登记和释放后的快照。危险输出通过仅测试用的包装核调用相同入口并保存返回码，以便检查 guard 和输入不变；这不构成消费者 release Trap 验收。生成器的分配前缀和只读长度可相互别名，非零跨度不能与输出重叠；空数组以 live storage 地址保留显式 used 存在标记。

公开 `get_scheduler_metadata(page_size=P)` 经 `PagedMetadataOp` 和专用 dispatcher 接入同一 AICPU 生成器。真实 C++ 与 fake 校验精确字数、P/C、Q 几何、dtype 和长度 tensor 描述；BSND 检查完整矩形 Q 的字节界限，TND 不按 `B*max_seqlen_q` 限制实际总量。长度值与 TND 总 Q 仍仅在设备检查，失败写完整清零的失败载荷，不由生成器触发消费者 Trap。输出为同设备连续 `int32[408+5B]`，不附加 Python 私有属性。B、Q 上界和 C 均非零时，公开记录的 blockDim 固定为平台全核；任一静态容量为零时固定为单核，均不依赖设备长度。设备判定的零 Q/KV 仍保存单工作核与零 workspace，blockDim 沿用静态选择。复用既有双事件握手、异步 tensor 持有及 allocator stream 登记。[公开测试](../../tests/test_flash_attn_npu_v3_mla_paged_metadata_public.py) 验证全部 P 的入口一致性、固定源完整记录、长度更新、fake/compile 与生命周期。此入口没有 Q-used 参数，也不发射 PA attention。

[`ReadPagedMetadata`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_paged_metadata_consumer.h) 使用独立调用 context、三组当前长度和当次 cache 描述，先检查精确字数、v6 头部/状态、dtype、Q 分配、used 标记及 P/C，再检查全部描述符、逐样本快照、源 S2 与有效页。Q 起点保持 64 位，KV 长度始终非累计；页表和 cache 不绑定记录，同条件下可换映射、物理池与合法 stride 后复用。调用方须先以 `BuildPagedCacheLayout` 校验原始视图及 storage；消费者再核对该布局的几何、完整跨度、可访问元素计数、地址/对齐与溢出边界，不读取 cache 内容。

页表索引以 64 位计算，每样本仅访问前 `ceil(L/P)` 项；负页号及 `page>=Np` 拒绝，其余 padding 不读取。Q 分配/used 为空也校验所有有效页，M=0 或 Np=0 仅在全部 L=0 且描述合法时通过。长度全部通过后，复用共同平台、局部资源、workspace 和空调度校验，并按 PA 基块检查变长 M 端点、未用槽位、FD 记账及整个零成本后缀。失败保持返回调度不变；新增 `InvalidPageTable`/`InvalidPage` 状态追加到原有错误枚举，旧编号不变。

[`LoadPagedMetadata`](../../csrc/ascend910/flash_attn_npu_3/mla/kernel/mla_paged_metadata.hpp) 在每个 Cube/Vector 校验编译 dtype 和真实发射核数，失败执行不依赖 `NDEBUG` 的 `AscendC::Trap()`，不要求 D2H 或主机等待。调用方必须在通过之后才构造 TPipe、设置 attention 事件或写 O/LSE/workspace，并保持 metadata、长度、页表和 cache 在异步执行期间只读且存活。[CPU 消费者测试](../../tests/test_flash_attn_npu_v3_mla_paged_metadata_consumer.py) 和 [910B release 探针](../../tests/test_flash_attn_npu_v3_mla_paged_metadata_device.py) 检查固定源记录、拒绝不写诊断输出、真实 Trap 及流生命周期；这些校验输出不是 attention O/LSE，也不定义 PA 空行输出。

### 5.8 stream 与内存生命周期

v1–v6 的生产或独立生成路径均使用当前 stream / AICPU stream 的双事件依赖：输入就绪后生成，写完执行设备屏障，生成完成后当前 stream 才能读取记录。`MlaMetadataArgs` / `MlaTndMetadataArgs` / `MlaTndBsndMetadataArgs` / `MlaUsedMetadataArgs` / `MlaPagedMetadataArgs` 中的地址仅属于当次发射；对 AICPU 使用的长度和记录登记 allocator stream，并保留 tensor 生命周期。生成完成后记录不依赖原长度存储，但每次消费仍须与当次长度比较。

v5 的 `UsedForward` 持有两组分配前缀与两组可选 used，生成和 attention 使用同一组当次描述符，并在 worker/current stream 登记其非空 storage。B=0 的显式空 used 以零计数、非空地址表示存在：host 使用已持有 metadata 的地址作不读取的零跨度描述符，不将空 Tensor 的 null `data_ptr` 误当作缺省参数。

跨 stream 复用由调用方建立正常 stream/event 依赖；前向对本次输入、长度和记录登记 stream。生成或消费期间不得改写长度/记录或共享可变 scratch。图内生成应在 replay 时重读设备长度，图外快照不会自动更新；fake/compile 与完整 graph replay 的验收层次见 [status](status.md)。新增协议须更新 version、kind 和 wordCount 校验，避免旧记录被新语义误读。

## 6. 代码结构

接入点为现有 [Python FA3 包](../../flash_attn_npu_3/flash_attn_npu_interface.py) 与 [flash_api.cpp](../../csrc/ascend910/flash_attn_npu_3/flash_api.cpp)。MLA 适配位于 `csrc/ascend910/flash_attn_npu_3/mla/`。BSND、TND 和混合布局共用以下分层，公开入口与具体 launcher 见 [6.7](#67-生产接入与构建) 和下表：

```text
Python FA3 入口 / custom op / fake
        ↓
C++ 校验与分派：mla_validation.h、mla_forward.cpp
        ↓
各布局 LaunchArgs（当次地址、scale、独立 context）
        + 内部生成或显式复用的 AICPU metadata
        ↓
各布局 launcher：Load*Metadata → 对应 KernelSchedule
        ↓
MlaDense / MlaTnd / MlaTndBsnd / MlaUsed
        ↓
MlaNonPaged（MlaTaskRing 别名）任务环、缓冲、Cube/Vector 同步
        ↓
block/mla_cube.hpp、mla_vector.hpp：QK + RoPE / softmax / PV / aMLA 更新
        ↓
tile/：源复制与向量操作、CATLASS TileMmad
```

v2 消费后将快照传给 `MlaDense<InputT, true, HasStrides>` 实际长度特化。独立驱动与生产发射分开，计算组件按下表复用；生产入口是否开放以 [status](status.md) 为准。

| 布局 | 驱动与记录消费 | 计算组件 |
| --- | --- | --- |
| BSND Q/KV | `mla_launch_impl.hpp` 消费 v1/v2 | `MlaDense<InputT, HasKvLengths, HasStrides>` → `MlaNonPaged<InputT, HasKvLengths, false, false, false, HasStrides>` |
| 紧凑 TND Q/KV | [`mla_tnd_launch_impl.hpp`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_launch_impl.hpp) 与 [`mla_tnd_attention_probe.cpp`](../../tests/mla_tnd_attention_probe.cpp) 消费 v3 | `MlaTnd<InputT, HasStrides>` → `MlaNonPaged<InputT, true, true, false, false, HasStrides>`，共用 Cube / Vector / Tile |
| TND Q + BSND KV | [`mla_tnd_bsnd_launch_impl.hpp`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_bsnd_launch_impl.hpp) 与 [`mla_tnd_bsnd_attention_probe.cpp`](../../tests/mla_tnd_bsnd_attention_probe.cpp) 消费 v4 | [`kernel/mla_tnd_bsnd.hpp`](../../csrc/ascend910/flash_attn_npu_3/mla/kernel/mla_tnd_bsnd.hpp)::`MlaTndBsnd<InputT, HasStrides>` → `MlaNonPaged<InputT, true, true, true, false, HasStrides>`，共用 Cube / Vector / Tile |
| 分配与 used：BSND Q/KV | `mla_bsnd_used_launch_impl.hpp` 消费 v5 | `MlaUsed<InputT, false, HasStrides>` → `MlaNonPaged<InputT, true, false, false, true, HasStrides>` |
| 分配与 used：TND Q/KV 或 TND Q + BSND KV | `mla_tnd_used_launch_impl.hpp` 消费 v5；独立对照使用 `mla_used_attention_probe.cpp` | 两布局共用 `MlaUsed<InputT, true, HasStrides>` → `MlaNonPaged<InputT, true, true, false, true, HasStrides>`；按 span 分离分配起点与有效长度，按四路独立 stride 计算元素地址 |

v3/v4 各自使用独立 context 和 Kernel 调度类型，防止将布局不同的起点快照误传给 BSND 或紧凑 TND。混合 span 的 KV 起点从两个 uint32 字恢复为 uint64，并在任务环 `RunInfo` 与 Tile 坐标中保持 64 位。FD 合并与 192/128 通用路径按计划迁移。

PA 的独立链路通过编译期特化复用上述任务环和 Cube/Vector：

| 层次 | 代码入口 | 职责 |
| --- | --- | --- |
| 地址描述 | [`mla_paged_layout.h/.cpp`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_paged_layout.h) | host 静态 storage/stride 描述；CPU 动态页表检查与分段查询是另两个接口，见 [6.4.3](#643-nd-paged-cache-契约与来源) |
| 调度与记录 | `mla_paged_schedule.h`、`mla_tnd_schedule.cpp`、`mla_paged_metadata.*` | 逻辑几何/长度 → 源 PA 调度 → v6；不接收物理页表 |
| 当次设备校验 | [`mla_paged_metadata_consumer.h`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_paged_metadata_consumer.h)、[`kernel/mla_paged_metadata.hpp`](../../csrc/ascend910/flash_attn_npu_3/mla/kernel/mla_paged_metadata.hpp) | v6 + 当前长度 + 当次 cache 描述/页表 → 调度或 Trap；不搬运 cache 内容 |
| 页内 Copy | [`tile/mla_paged_copy.hpp`](../../csrc/ascend910/flash_attn_npu_3/mla/tile/mla_paged_copy.hpp) | 已校验的 cache 描述/页表与逻辑 token 范围 → L1 NZ；QK 联合搬运内容/RoPE，PV 搬运共享 V |
| BSND PA attention | [`kernel/mla_paged.hpp`](../../csrc/ascend910/flash_attn_npu_3/mla/kernel/mla_paged.hpp) 与 [`mla_paged_attention_probe.cpp`](../../tests/mla_paged_attention_probe.cpp) | 独立 v6 消费 → `MlaPagedBsnd` → `MlaTaskRing` 的 PA 特化；含 Q-used、设备空行初始化及页内 Copy，尚无生产发射 |

`setup.py` 编入 v6 AICPU 生成器；PA attention 实例、消费者和地址 `.cpp` 仍只由独立测试调用。共用 Cube 包含 PA Copy header，但生产实例保持非分页特化。构建范围与运行调用方必须分别核对，不能从 `.so` 含有某个对象推断公开能力。

- 以固定 CANN 源码为起点，保留来源及版权；许可证位于 [`LICENSE.cann`](../../csrc/ascend910/flash_attn_npu_3/mla/LICENSE.cann)。CATLASS 示例用于理解分层，不能替代源执行流程。
- 允许文件拆分、命名、类/模板组织、参数封装、构建接入和 gert/optiling 解耦。复用已有组件须保证运算、布局、精度、资源与同步契约等价；无法等价时在本仓封装源逻辑，不修改 CATLASS 子模块来容纳常规迭代。
- 保留切块/分核策略、数值顺序、中间 dtype、累加与舍入、尾块、内部布局、缓冲数量及生命周期、预取和事件依赖。FA3 必需的指针/长度/stride/mask/split 差异单列，不顺带改变 Kernel 算法。
- 目标兼容 910B/C 的资源配置；平台核数、Cube/Vector 比例及存储容量来自查询。调度和源逻辑保留 C1V1/C1V2；生产开放的平台见 status，合成 C1V1 调度测试不等于 910C 前向验收。
- 新 AICPU 文件须加入 `setup.py`，设备 header 的增量依赖也须覆盖；构建细节见 [6.7](#67-生产接入与构建)。生产入口不调用原 CANN attention 算子充当迁移实现。

### 6.1 压缩 MLA 的来源映射

以下映射固定到第 1 节 CANN commit；路径仍相对于 FIA 根目录，`../common/` 指 `attention/common/`。目标路径相对于本仓 `csrc/ascend910/flash_attn_npu_3/mla/`，各模块的实现状态见 [plan](plan.md)。

| 固定源文件 / 函数 | CATLASS 层次与迁移位置 | 必须保留的行为 |
| --- | --- | --- |
| `op_host/arch22/fia_tiling_nonquant_mla.cpp`：`CalcInnerSize`、`CalcMBaseSize`、`Split`、`CalcMmResSize`、`CalcNormalWorkspaceSize` | `mla_schedule.h/.cpp`，host/AICPU 共用的无 gert 调度计算 | 切块、分核输入、workspace 尺寸及核数；不另写一套形状启发式 |
| 同一 tiling：`DealSameSeqEachBatch`、`CreateSplitInput`；`SplitCore` 的 `GetS1SeqSize/GetS2SeqSize` | `mla_bsnd_schedule.h/.cpp`，BSND 实际 KV 长度调度 | int32 长度完整校验后数值转换为源 int64；非均匀长度的 M 基块、成本与零长度尾部端点规则 |
| `op_host/fused_infer_attention_score_tiling_info_parser.cpp`：`GetS1Size`、`GetMaxActualSeq`、`ProcessKVActualSeqLoop`；同一 tiling 与 `SplitCore` 的累计长度分支 | `mla_tnd_schedule.h/.cpp`，CPU/AICPU 共用紧凑 TND 解析与调度 | 带 0 的 int32 前缀和校验、token 起点、数值转换的 int64 尾值、逐样本 M 块与零成本尾部 |
| 同一 parser 与 tiling：Q 累计长度、KV 非累计实际长度组合 | `mla_tnd_bsnd_schedule.h` 声明，`mla_tnd_schedule.cpp` 实现并复用变长调度 | Q 紧凑起点与 KV 的 64 位 `b*Sk` 分开，物理 KV 容量不随实际长度缩短 |
| 同一 tiling 的 `CreateSplitInput` 非累计 Q/KV 分支及 `SplitCore` | `mla_used_schedule.h` 声明，`mla_tnd_schedule.cpp` 的 CPU/AICPU 共用 used 入口 | 分配起点/容量与实际使用长度分别保存，Q/KV 独立选择累计尾值或非累计 used，复用变长端点映射 |
| `../common/op_host/split_core.cpp`：`SplitCore`、`CalcSplitInfo`、`CalcCostInfo`、`CalcSplitPlan`、`AssignByBatch/Row/Block`、`ForceAssign` | 调度模块私有的 `detail/cann_split_core.h/.cpp` | 枚举核数、cost、容差、遍历次序和相同 cost 时保留已有方案的规则；专用 MLA 使用此文件，不是 `split_core_v2.cpp`；保留源算法分支不表示开放对应功能 |
| `op_host/fused_infer_attention_score_tiling.h`；`op_kernel/arch22/fia_public_define.h`：`ConstInfo`、`RunInfo` | 调度字段与 Kernel 运行时参数 | 明确字段宽度、单位、端点语义和运行时状态，见 6.2 |
| `op_kernel/arch22/fused_infer_attention_score_v3.cpp`：`INVOKE_FIA_OP_GENERAL_IMPL` 的 `BSH_KVBSH_MLA_TILING` 分支 | 独立实例化单元、launcher 和生成器 | FP16/BF16、无 PA、无 FD、BSND 输入输出、C1V1/C1V2；不带入 FIA 注册与大签名 |
| `op_kernel/arch22/fia_kernel_nonquant_mla.h`：`Init`、`InitCalcParamsEach`、`CalcParams`、`FlashAttention`、`ExecuteTask` | `kernel/mla_dense.hpp` | 核区间解析、三项任务环、两级延迟流水、初始化及排空 |
| `op_kernel/arch22/fia_block_cube_nonquant_mla.h`：`ComputeMm1/2`、`ProcessMm1/2` | `block/mla_cube.hpp` | QK 内容/RoPE 排列、L1/L0 切分、FP32 累加、Fixpipe 写回和原子加 |
| `op_kernel/arch22/fia_block_vec_nonquant_mla.h`：`SoftmaxFlashV2Compute`、`AmlaVecCompute`、`ProcessAmlaNupdate`、`ProcessVec1L/2L`、`DealBmm2ResBaseBlock` | `block/mla_vector.hpp` | 源 softmax 与 aMLA 更新、舍入、尾列、行分工、O/LSE 写回；不是普通 online softmax 的替代实现 |
| `../common/op_kernel/matmul.h`：`MatmulKPP`、`LoadDataToL0A/B`；`buffers_policy.h` | `tile/` 与 Cube Block 的局部资源管理 | L0 双缓冲、加载布局、首 K 清零、unitFlag 和小矩阵 PIPE_M barrier |
| `../common/op_kernel/memcopy/attn_copy_gm_to_l1.h`：`CopyQueryGmToL1`、`CopyKvGmToL1` | `tile/mla_copy.hpp` | BSND 的 S1G 展平次序、ND→NZ、跨 query/head 的复制分段 |
| 同一文件：`CopyKKropePAGmToL1`、`CopyKvGmToL1::ProcessPageAttention` | `tile/mla_paged_copy.hpp`，复用 `mla_copy.hpp` | 页内拆段、QK 每段共用页号查询、K/V 独立 64 位 stride、NZ 偏移；事件仍由 Block 管理 |
| `op_kernel/arch22/memory_copy.h`：`CopyAttenOutUbToGm`、`DataCopySoftmaxLseBSND` | `block/mla_vector.hpp` 的 dense 写回 | 保留 O 的 query/head 首段、中段和尾段复制，以及 LSE 的 `[B,Hq,Sq]` 寻址 |
| `../common/op_kernel/vector_common.h`：`RowDivs`、`RowMuls`、`ComputeSoftMaxLse` | `tile/mla_vector_ops.hpp` | 保留行列 repeat 选择、广播布局、向量指令及 barrier 顺序 |

对应 C1V2 dense 非 FD key 为 FP16 `105000000010000001`、BF16 `105000000010022221`；C1V1 分别为 `105000000110000001`、`105000000110022221`。key 只用于核对来源，FA3 不暴露 CANN key。源 `FiaKernelNonQuantMla::FlashDecode` 与 `fia_block_vec_flashdecode_arch22.h` 的合并保留到 M10；192/128 的通用 `fia_kernel_nonquant.h` 保留到 M11，均不混入现有 dense 分派。

### 6.2 调度字段与边界

本节集中定义布局共用的字段、平台和空容量规则。逐样本长度及物理起点见 [6.3](#63-长度与布局调度)，真实输入地址见 [6.4](#64-输入-stride-与地址校验)。这些是后端实现边界，不能当作官方 FA3 的通用限制。

#### 6.2.1 基本块、字段与无 split 映射

`mBaseSize` 初值 256。固定源 `CalcMBaseSize` 仅在等长且 `(B*Hkv)%aicNum != 0` 时，按最小公倍数求可均分 S1G 的候选，须整除 `Sq*Hq` 并在 128..256；非均匀实际 KV 长度保持 256。不能改成按头数选择 8/16 行。正常 tiling 只调用一次 `Split`；非 PA 的 `s2BaseSize=min(Sk,512)`，不在分核后擅自重跑 `CalcInnerSize`。

| 源字段 / 计算 | 本仓归属与单位 | 规则 |
| --- | --- | --- |
| `baseParams.bSize/n2Size/gSize/s1Size/s2Size/headDim/headDimRope` | `MlaSchedule` 静态几何，元素/头/序列计数 | 对共享 KV 分别为 B、1、Hq、Sq、Sk、512、64；窄化前校验 |
| `innerSplitParams.mBaseSize/s2BaseSize` | 可复用调度，元素数 | 保留源基本块；BSND 的 `gS1=s1*Hq+head`，不是 head-major |
| `outerSplitParams.bN2End/gS1End/s2End`、`usedCoreNum` | 每核区间的排他结束三元组；gS1/S2 是块索引 | 第 0 核从零开始，其余从前核端点开始；Kernel 再乘基本块 |
| `aicNum/aivNum/cvRatio`、`CalcNumBlocks/CalcScheduleMode` | 平台资源、启动维度、`BATCH_MODE` | 核数、使用核数、blockDim 分开；端点容量 26 不是芯片核数 |
| `mm1ResSize/mm2ResSize` | 每核每槽元素数，字节计算用宽整数 | `M=align16(min(Sq*Hq,mBaseSize))`、`N=align32(s2BaseSize)`、`R1=M*N`、`R2=M*512`；N 对齐 32 个元素 |
| `fdParams`、`fdAccumOutSize/fdLogSumExpSize` | split 调度及 workspace 记账 | 保留源字段作对照；非 FD 路径不消费 split，不能清零原 FD 方案冒充等价 |
| `scaleValue`、地址、stride、workspace 地址 | 当次 launch 参数；独立于记录 | scale 仍在 QK 后 FP32 Muls 生效，缺省为 `1/sqrt(576)` |
| `softmaxLseFlag/outputLayout/batchContinuous/needInit`、长度/PA/mask 标志 | 布局与功能条件；地址另传 | 按功能步骤映射；dense 保留 LSE 初始化与同步，未接通组合明确拒绝 |
| `RunInfo.loop/bn2IdxInCurCore/isFirstSInnerLoop/isLastS2Loop/isValid` | 每次 launch、每核私有状态 | 任务环和缓冲轮换不进入可复用记录 |

**FA3 禁止 split 的映射。** `num_splits=1` 先运行原 `SplitCore`；无 FD 时保留原结果，有 FD 时将端点向上对齐到完整 M 块结束处、合并空区间，重建 `s2End=0` 分区。按源 `CalcBatchCache/CalcS1GCache` 重算正常块与尾块成本。基本块、原候选算法和设备数值/事件顺序不变；映射结果与原生非 FD 输出分开验收，不称为原生 CANN 输出。

#### 6.2.2 平台与溢出

`MlaPlatformInfo` 携带可用 L1/UB/L0、libapi 预留及平台 `CalcTschBlockDim(c*cvRatio,c,c*cvRatio)` 表，按实际使用核数索引，不能猜测 blockDim。源 `SetSplitOutput` 对 C1V1 的 FD vector 可能按两倍核数读取；本仓按 vector 实长复制到 52 项容量数组并清零剩余项，oracle 比较原完整 vector，不执行该越界序列化循环。

字段与地址限制由 [`mla_schedule.h`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_schedule.h) 定义：`B<=2^32-1`、`Sq<=2^30`、`Hq<=floor((2^32-1)/512)`、`Sq*Hq<=2^32-256`、`B*Sq*Hq<=floor((2^63-1)/1024)`。还须检查 uint32 LCM/总物理块数、资源及 uint64 workspace 加法。它们是实现限制，不是官方 FA3 接口限制。QK/P workspace 按 S2 基块预算，不按完整 KV 容量分配；调度失败不改写调用方原输出。

#### 6.2.3 全局空容量

`B=0`、`Sq=0` 或 `Sk=0` 时，v1/v2 保留真实形状、dtype、平台信息，使用一个校验核；基本块、端点、FD、workspace 和局部内存需求为零。生成和消费仍校验全部当次长度，不能因空输出绕过错误上报。校验后跳过 attention，不访问 Q/K/V/qv 或分配 attention workspace；空 batch/query 返回空 shape，全局空 KV 在同一 stream 填 `O=0、LSE=+inf`。非零容量中的样本零 KV 为 `O=0、LSE=-inf`，与全局空容量分开验收。

### 6.3 长度与布局调度

[`BuildDenseSchedule`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_schedule.cpp) 处理整容量 BSND；以下入口复用同一 `detail/cann_split_core.*`，由 host/CPU 驱动与相应 AICPU 生成器编译同一源码。长度先完整校验，再数值转换为固定源格式；分配起点与 used 分别处理。记录格式见 [第 5 节](#5-metadata-设计)。

#### 6.3.1 BSND 实际长度与零尾部

[`BuildBsndKvLengthSchedule`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_bsnd_schedule.cpp) 先校验全部 int32 长度再数值转换为源 int64，不重解释指针；缺省或整容量长度退化为既有 dense 结果。源 `CalcSplitPlan` 在成本耗尽后可能把尾部零 KV 样本留在计算端点之外；消费者校验省略部分逐样本为零。设备计算按源 `InitActualSeqLenQ/KV`、`InitOutputSingleCore`、`GetTaskDealMode/DEAL_ZERO` 独立初始化空行，不能仅凭计算分区覆盖它们。

`MlaDense<InputT, true, HasStrides>` 复用 dense 的 Cube / Vector / Tile：KV 分配容量与 batch 起点不随实际长度变化，物理 stride 见 [6.4.1](#641-bsnd-输入-stride-与地址校验)；S2 块数、尾块宽度和最后一块标志来自已校验 v2 的逐样本快照。零 KV 任务只推进轴，不递增任务环计数；存在零 KV 时，按全部输出范围清零 O，LSE 初始化为 FA3 的 `-inf`，因此端点之外的零尾部同样有定义结果。非空行由原数值路径覆盖输出；默认 `MlaDense<InputT>` 保持整容量行为。生产入口和独立驱动在任何 TPipe、attention 事件或输出写入前调用 v2 校验消费者。

固定源 `FiaInfoParser::ProcessKVActualSeqLoop` 对 BSND 中任意短 KV 设置 host `needInit`，源 probe 原样传入该标志；迁移特化仅在存在零 KV 时清零 O，避免重复初始化随后必被覆盖的非空输出。源零 KV 的 LSE 为 `+inf`，对照逐项检查此差异，并按第 3 节契约验收迁移结果的 `-inf`。

#### 6.3.2 紧凑 TND 解析与调度

[`ParseTndLengths` / `BuildTndSchedule`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_schedule.h) 只接收 Q/KV 均为紧凑 TND 的输入，由 CPU 驱动和 v3 AICPU 生成器共用。两组前缀和均须为 `int32[B+1]`，包括 B=0 时的 `[0]`；起始 0、单调非降、尾值等于各自总容量，逐段差值不超过调用方的 `maxQueryLength/maxKvLength`。max 参数是上界，可大于观测最大值；源调度的 `s1Size/s2Size` 使用观测最大段长。先完成全部校验，再分配拥有存储的逐样本 token 起点与长度，并数值转换为不带起始 0 的 int64 累计尾值送入既有 `SplitCore`。不保留输入地址，失败不改写原结果；该入口不接受 used，独立 CPU used 见 [6.3.4](#634-分配长度与-used-的-cpu-归一)；混合 BSND KV 使用 [6.3.3](#633-tnd-q-与-bsnd-kv-的混合调度) 的独立入口；v3 协议见 [5.4](#54-v3紧凑-tnd-记录与设备消费)。

固定源 `ProcessKVActualSeqLoop` 的 `isSameActualseq` 比较**累计尾值**，不是 TND 各段差值，因此等段长 TND 与 BSND 可能选择不同 M 基块。例如 B=10、Q 每段 2、Hq=128、KV 每段 511，在 20 核时 TND 的 M=256、BSND 的 M=128；TND KV 改为 `[511,0,...,0]` 后累计尾值相同，M=128。该源行为保留，由原始 parser 函数与 tiling 独立差分验证；不以 BSND 的等长判断替换。BSND 等价检查在基本块相同的用例比较全部字段，布局特有的基本块差异另验计算覆盖与成本。

TND 的 M 块数随样本变化，禁止用 `B*统一块数` 展平端点。无 split 映射保留源端点跨空 query 样本的 batch 坐标，只向上取整被 S2 切开的 M 块并合并重复区间；逐批成本前缀计算包含尾 M 块。省略的后缀必须逐样本满足 Q=0 或 KV=0。总 Q 或总 KV 为零时沿用全局空容量的单校验核、零 workspace 规则，仍校验所有前缀和；两者总量非零但有效样本互不相交时保留源全零成本调度及 workspace。

TND 的 batch/max/head 字段沿用 [6.2.2](#622-平台与溢出) 的静态界限，其中 `maxKvLength<=16384` 限制每段长度上界，**不是整个 TND KV tensor 的总容量上限**。总 Q/KV 另须 `<=INT32_MAX`；地址上界检查 `totalQuery*Hq`，而非矩形 `B*maxQueryLength*Hq`。总物理块数按实际各段的 M/S2 块乘积求和，须落在 uint32 内；LCM、平台资源和 uint64 workspace 加法仍独立检查。公开紧凑 TND 的真实/fake/C++ 入口采用相同静态界限，前缀数值与调度资源由设备生成器及消费者校验。

[`kernel/mla_tnd.hpp`](../../csrc/ascend910/flash_attn_npu_3/mla/kernel/mla_tnd.hpp) 的独立计算入口只接受 `MlaTndKernelSchedule`，与 BSND 复用 `MlaNonPaged` 任务环及 Cube/Vector 数值路径。每个任务保存该样本的 Q/KV token 起点；M 块按实际 Q 段长推进，空 Q/KV 样本跳过计算，输出初始化覆盖端点外的零成本尾部。Tile 接收绝对 token 起点，O 写 `[Tq,Hq,512]`，LSE 直接写 FA3 `[Hq,Tq]`，不复制或转置完整输出。LSE 使用 64 位地址；头间字节跨度超过 DMA 的 uint32 字段时逐头写入。全局空 KV 的 O/LSE 初始化之间保留 MTE3→Vector 依赖，符号遵循 3.3。

#### 6.3.3 TND Q 与 BSND KV 的混合调度

[`ParseTndBsndLengths` / `BuildTndBsndSchedule`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_bsnd_schedule.h) 是 CPU/AICPU 共用的混合布局解析与调度入口。Q 接收紧凑 `int32[B+1]` 前缀和，沿用 6.3.2 的起始零、单调、尾值、max 上界及地址检查；KV 接收恰好 B 个 `int32` 实际长度，每项在 `0..Sk`，`Sk=0..16384` 为 BSND 的单样本分配容量，不要求 cache 物理连续。B=0 仍要求 Q 的 `[0]`，零长 KV 数组可用空指针；非空 batch 不接受空 KV 指针表示整容量。调度输出持有独立的起点与长度，失败不改写已有结果。

KV token 起点固定为 `uint64(b)*Sk`，短长度和零长度不压缩 cache。`B*Sk` 不受紧凑 KV 的 `INT32_MAX` 总量限制，合法起点可超过 uint32；B/Sk 的既有界限保证最大 KV 字节跨度小于 `INT64_MAX`。Q token 起点来自前缀和，v4 的 Q 长度始终是完整分配段；独立 Q-used 由 [6.3.4](#634-分配长度与-used-的-cpu-归一) 的归一组件与 v5 协议表达。

固定源 `CreateSplitInput` 接收数值转换的 int64 Q 累计尾值与非累计 KV 长度，分别设置 `isAccumSeqS1=true`、`isAccumSeqS2=false`。调度 `s1Size` 使用观测 Q 最大值，`s2Size` 保留物理 Sk；观测 KV 最大值单独保存。`ProcessKVActualSeqLoop` 的等长判定比较原 KV 数组值，因此 B=10、Q 每段 2、Hq=128、20 核、Sk=511 且 KV 每段 511 时 M=128，而紧凑 TND 为 M=256。S2 基块按容量取 `min(Sk,512)`，任务成本和总物理执行块数按各样本实际 Q/KV 长度计算；后者须落在 uint32 内。

混合布局与紧凑 TND 共用变长 M 块端点映射、零成本后缀检查和 workspace 计算。总 Q=0、B=0 或 Sk=0 时使用单校验核与零 workspace；非空物理容量中的全零 KV 保留源零成本调度和 workspace。[独立 CPU 对照](../../tests/test_flash_attn_npu_v3_mla_tnd_bsnd_schedule.py) 的源 oracle 读取固定 CANN Git blob，不调用迁移解析器。CPU 与 [v4 AICPU 生成器](#55-v4tnd-q-与-bsnd-kv-混合记录) 共用此组件；其 64 位起点不能写入紧凑 v3 的四字 span。独立 attention 消费完整 v4 后，Q 使用紧凑起点、KV 使用物理起点，按当次样本 Q/KV 长度推进任务；O 为 `[Tq,Hq,512]`，LSE 为 `[Hq,Tq]`。非空物理容量中全零实际 KV 仍为 `O=0、LSE=-inf`，物理 Sk=0 则为 `O=0、LSE=+inf`；两者都先校验记录。公开混合前向复用这条链路，其调用段按第 7 节验收。

#### 6.3.4 分配长度与 used 的 CPU 归一

[`ParseUsedLengths` / `BuildUsedSchedule`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_used_schedule.h) 是 CPU/AICPU 共用组件，覆盖 BSND Q/KV、TND Q/KV 和 TND Q + BSND KV。每个轴描述物理总 token 数与每样本容量上界；BSND 总量必须等于 `B*capacity`，TND 总量须 `<=INT32_MAX`，其 `int32[B+1]` 分配前缀须起始 0、单调且尾值等于总量。BSND 不接受前缀。静态 Q 字段/地址、KV 每段上界与既有路径相同，TND 地址检查仍按实际总容量而非矩形上界。

两组可选 used 均为 `int32[B]`，分别完整校验 `0 <= used[b] <= 本轴分配段长`，包括实际 Q 为零时的 KV 校验。缺省 used 使用整个分配段；空 batch 的 TND 仍要求 `[0]`。校验全部形状、计数和数值后才分配拥有存储的 span；失败不改写结果，不保留输入指针。span 分别保存 64 位 Q/KV token 起点、32 位分配长度和实际长度：BSND 起点为 `b*capacity`，TND 起点为分配前缀，缩短 used 不移动后续样本。BSND Q/KV 起点均可超过 uint32。

基本块与 workspace 的几何沿用分配容量；TND 使用观测的最大**分配段长**，调用方 max 参数仍只是上界。送入固定源 `CreateSplitInput` / `SplitCore` 的实际长度按轴选择：显式 used 数值转换为非累计 int64；缺省 TND used 沿用分配累计尾值；缺省 BSND used 使用整容量长度。源 `isSameActualseq` 仍比较所选 KV 数组的原值，因此 TND 显式整段 used 与省略 used 可有不同 M 基块。此处是 FA3 的分配前缀与 used 到源调度字段的适配；物理寻址另外使用 span，不用非累计 used 推算 TND 起点。

任务、成本和 uint32 物理执行块上界按实际 Q/KV 长度计算，复用既有变长 M 端点映射与逐样本零成本后缀检查。全局空分配沿用单校验核、零 workspace；非空分配中的全零 used 保留零成本调度和 workspace。独立 [CPU 测试](../../tests/test_flash_attn_npu_v3_mla_used_schedule.py) 从固定 Git blob 编译原始 parser/tiling/SplitCore，源端独立转换输入，另以原始测试输入枚举有效任务和成本。[v5 生成、设备校验与 attention](#56-v5分配与-used-记录) 使用同一长度语义；入口分派见 [6.7](#67-生产接入与构建)，既有 v1–v4 语义不变。

### 6.4 输入 stride 与地址校验

长度调度产生逻辑 token 起点，输入描述把它换算成真实元素地址。Q、qv、K、V 各有自己的 `data_ptr()`、stride 与 storage offset；metadata 不保存这些地址或 stride。下述 CPU 地址校验、设备消费及生产接入是不同验收层次，当前层次见 [status](status.md#当前可调用能力)。

#### 6.4.1 BSND 输入 stride 与地址校验

[`BuildBsndTensorLayout`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_bsnd_layout.h) 是独立 CPU 组件，接收单个 FP16/BF16 BSND 视图的 shape、元素 stride、storage offset 与 storage 字节数。Q、qv、K、V 分别构造描述，不推导另一张量的 stride。它只检查 64/512 维输入的地址，调度形状限制、dtype/device、跨张量匹配仍由前向边界另行检查。生产 BSND 前向在分配输出和发射前复用该组件，并检查 `data_ptr()` 加真实跨度的指针溢出。

描述保存 64 位 `batchStride/rowStride/headStride` 和相对 `data_ptr()` 的 `spanElements`；末维 stride 必须为 1，外层 stride 非负。只读的广播、重叠、外层切片及 BNSD 归一视图均可表达。非空视图的相对地址为 `b*batchStride+s*rowStride+h*headStride+d`；独立校验整个分配视图的末端及 storage offset 相加后的字节范围均不超过 `INT64_MAX`，且不越过 storage。每次乘加前检查溢出，不能用 used 长度缩短校验范围。空 batch/sequence 没有可达元素，`spanElements=0`，允许 offset 超过 storage；空维度/单例维度中未使用的 stride 不作字节窄化，负 stride 和非法末维仍拒绝。失败不修改输出描述。

固定 FA3 的 `set_params_fprop` / `mha_fwd` 分别读取各张量的 `data_ptr()` 和真实 stride。`data_ptr()` 已包含 storage offset，launch 不得再次加 offset；offset 只用于主机侧 storage 边界校验，stride 不写入 metadata。CPU 验收用真实 PyTorch 视图的标量地址与独立 Python 大整数 oracle；不将其记为设备执行验收。

BSND 设备组件的 `HasStrides` 模板实例将四路描述从 Kernel 传入 Cube Block / Copy Tile。前向根据四个输入的连续性选择连续或 stride 实例，覆盖整容量 v1、实际 KV 长度 v2 和 Q-used v5。Q/qv 仍按源 S1G 顺序分成首段 heads、中间矩阵和末段 heads，每一路分别计算 batch/row/head 地址；共享内容 KV 的 QK 和 PV 均使用 V 的真实布局。O/LSE、任务环、切块、数值及同步流程不变，不复制完整 cache；metadata 不绑定 stride，复用时仍检查当次长度。真实/fake/C++/compile 共用接受范围，完整 storage 与 scale 校验保留在算子的 fake/C++ 边界，公开包装器跟踪时不调用 Dynamo 不支持的 `storage_offset()` 或符号 `math.isfinite()`。

Copy Tile 沿用固定 CANN 源 `attention/common/op_kernel/memcopy/attn_copy_gm_to_l1.h` 的 ND→NZ 搬运和回退，并保留来源声明。固定源单矩阵在 `srcDValue>=65536` 时逐行复制，多矩阵在 `srcNdMatrixStride>65536` 时逐矩阵复制。FA3 适配始终保留 64 位地址运算，并将多矩阵回退边界改为 **`>=65536`**：910B 的实际 `Nd2NzParams::srcNdMatrixStride` 为 uint16，65536 会截断为零，独立实测已发现原边界产生错误输出。零 row/matrix stride 也使用逐行/逐矩阵复制；矩阵 stride 合法但 head stride 超限时必须同时走回退，不能窄化后再判断。

独立 [stride attention 验收](../../tests/test_flash_attn_npu_v3_mla_bsnd_stride.py) 用真实 NPU storage 检查 65535/65536/65537、实际使用的 >uint32 stride、空/单例轴、广播/重叠、BNSD、四路偏移及 used poison。固定源接收计时前独立物化的连续逻辑输入，执行相同非 FD 调度；任意四路 stride 没有等价的源 ABI。因此该源精度/组件性能对照须注明物理布局差异和转换边界，不能记为相同物理布局或公开 E2E 验收。

[生产 stride 验收](../../tests/test_flash_attn_npu_v3_mla_bsnd_stride_production.py) 另检查上述边界的公开与原始 C++ 调用、完整 storage 拒绝、长度/metadata 更新、stride/scale 复用及流生命周期。公开 E2E 双方接收相同的物理视图；一次性视图准备与 H2D 对称排除，原生 CANN 的逐次 Slice/AsStrided/BroadcastTo 转换计入调用耗时并单列设备耗时。内部 Q-used 没有公开等价入口，生产 Kernel 的同调度对照继续单列。

#### 6.4.2 TND 输入 stride 与地址组合

[`BuildTndTensorLayout`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_layout.h) 将单个 `[T,H,D]` 视图映射为 `[1,T,H,D]`，复用 BSND 的完整 storage/有符号字节范围校验。描述保存 64 位 `tokenStride/headStride/spanElements`；要求 `T>=0、H>0、D=64/512`、外层 stride 非负、末维 stride=1。空 token、单例、广播、重叠和非零 offset 沿用 6.4.1 的规则，失败不修改输出。

样本 b 的元素地址相对 `data_ptr()` 为 `(allocationPrefix[b]+token)*tokenStride+head*headStride+d`。前缀决定分配起点，used 只限定有效区；不能用 used 累加值或 `b*maxLength` 代替前缀。Q/qv/K/V 分别使用自己的描述；混合布局的 K/V 则使用各自 BSND 的 batch/row/head stride。两种描述均校验完整分配区，不能因任一轴 used 为零而跳过另一轴的校验。

该组件只校验单 tensor 地址；dtype/device、四路形状、分配前缀、used 和调度资源仍由各自边界检查。[CPU 验收](../../tests/test_flash_attn_npu_v3_mla_tnd_layout.py) 以独立大整数 oracle 和真实 PyTorch 标量地址验证四路视图、T01/used/BSND cache 组合。组件是否已链接生产、是否有设备 stride 验收，以 status 为准；CPU 地址成立不代表 launcher 已消费该描述。

独立设备链路将已校验的 TND 描述表示为 `batchStride=0、rowStride=tokenStride`，复用 `MlaBsndInputLayouts` 的四路 row/head 描述。`MlaTnd`、`MlaTndBsnd` 与 `MlaUsed` 的 `HasStrides` 实例在完成 v3/v4/v5 校验后传给 Cube / Copy Tile；Q 按 TND 分配前缀寻址，KV 的寻址模式由布局及 v5 已校验的 KV layout 决定。连续实例保留原地址路径，输出布局、metadata 字节协议、任务环、数值与同步顺序不变。生产前向复用同一设备组件，并在 host 发射前增加下述四路完整 storage 校验。

独立 [TND stride attention 验收](../../tests/test_flash_attn_npu_v3_mla_tnd_stride.py) 覆盖 ST01/NTD、四路 offset、T01 与 used poison、混合 cache、空/广播/重叠、65535/65536/65537 和实际读取的 >uint32 stride。固定源接收从原始测试输入独立物化的连续视图；TND 的有效段按 used 压紧，混合 cache 保留分配槽位。物化与压紧在计时前完成，组件结果不能解释为相同物理布局或公开 E2E。

生产 `TndForward`、`TndBsndForward` 与 `UsedForward` 在分配输出及发射前调用地址组件，覆盖 TND Q/KV v3/v5 和混合 TND Q + BSND KV v4/v5 的全部四路输入；连续输入也检查完整 storage。混合 Q/qv 使用 TND 描述，K/V 使用独立 BSND 描述，内部混合 Q-used 继续复用 TND used launcher。按四路连续性选择原连续实例或 `HasStrides` 实例，stride 不写入 metadata，分配前缀和 used 仍逐次在设备检查。Python/fake 与 C++ 保持相同的 storage 边界；compile 跟踪时由算子 fake 检查、执行时由真实 C++ 检查。公开混合 v4 的 metadata 可跨合法视图复用，但仍须匹配当次 Q 前缀、KV 长度及静态几何。

[生产 TND stride 验收](../../tests/test_flash_attn_npu_v3_mla_tnd_stride_production.py) 覆盖公开 varlen、内部和原始 C++、真实/fake/compile、前缀/used 更新、完整 storage 拒绝与流生命周期。公开性能双方接收相同的物理视图；原生 FIA 的逐次布局转换，以及 used 适配的 gather/scatter，均计入 R=1/4 E2E。profiler 区分最终 attention 与带相同 FIA API 前缀的转换任务；显式适配器 wall 与全部转换设备耗时分别报告。

#### 6.4.3 ND paged cache 契约与来源

本节先定义 **PA 目标契约**，再说明地址、Copy、独立 attention 与调度组件；设备 v6 消费校验见 [5.7](#57-v6nd-pa-记录)。这些层次分别验收，实机结果见 [status](status.md)，组件通过不能代替公开接入验收。目标限 FP16/BF16、512+64 共享 KV、Hkv=1、无 mask、`num_splits=1`。Q/qv 复用 BSND 或 TND stride 描述；cache 为 `k:[Np,P,1,64]`、`v:[Np,P,1,512]`，页表为同设备 `int32[B,M]`，末维 stride=1，行 stride 可为合法非负值。K/V 各自末维连续、外层非负 stride 和独立 storage offset，沿用完整 storage/字节校验；只读广播、重叠和重复页号不因地址相同而拒绝。

##### 入口、容量与固定来源

目标公开入口为只读 `flash_attn_with_kvcache` 的 `page_table` 和 `get_scheduler_metadata` 的 `page_size`；BSND Q 使用整容量，TND Q 提供 `cu_seqlens_q` 与 `max_seqlen_q`。`flash_attn_func` / `flash_attn_varlen_func` 的固定签名没有页表，不能据“支持 TND query”给 varlen 新增 PA 参数。`page_table` 与 `cu_seqlens_k` 互斥；Q-used 只在内部前向接入，保持分配起点不随 used 缩短而移动。M07 不接入 `cache_batch_idx/kv_batch_idx`、追加 KV、rotary、mask、split 或 192/128。

逻辑 KV 容量为 `C=M*P`，不等于 `Np*P` 或最大实际长度。M07 沿用每样本容量 `0..16384` 的后端边界，乘法先按宽整数检查；物理页池和跨 batch 总量不套用 16384 上限，另受页号可表达性、字段与地址界限约束。前向 `cache_seqlens=None` 表示 C，整数表示每样本同一 `0..C` 长度，tensor 是连续 `int32[B]`。整数物化用逻辑 B；TND 下 B 来自 Q 前缀长度，不能照抄官方包装器以 `q.shape[0]`（Tq）建长度 tensor 的分支。显式生成器仍只收长度 tensor，`max_seqlen_k` 须与前向 C 一致。

| 固定源锚点 | 核实结果与迁移映射 |
| --- | --- |
| CANN parser `GetBlockSize`；arch22 `CheckFeatureBlockSize` / `CheckKVShapeForPageAttention` | P 为正数、16 对齐且 ≤1024；FP16/BF16 的 32 字节对齐同样要求 16。固定官方 `flash_attn_interface.py` 的 kvcache 文档明确允许任意页大小，`flash_api.cpp::get_pagedkv_tma` 以页大小整除 tile 选择 TMA，非 TMA 路径仍存在；迁移保留 16..1024 全部 64 种 |
| CANN parser `GetKvLayout` / `GetS2SizeForPageAttention` / `GetBlockTableDim2` | ND cache 的源表示为 `BnBsH` 或 `BnNBsD`，外部 NZ 排除。普通 host tiling 从最大有效页数求 S2，最大 workspace 分支用页表列数；下发的 `maxBlockNumPerBatch` 是实际页表宽度。不能把源调度 S2、逻辑容量 C、页表行距和物理页数合为一个量 |
| arch22 `memory_copy.h::GetKvActSeqMode`；MLA `CreateSplitInput` | PA 的 KV 长度恒为 `BY_BATCH`；Q 为 TND 时才把 Q 前缀数值转换为不带起始 0 的 int64 累计尾值。不能照搬非分页 TND 对照中的累计 KV 长度 |
| MLA tiling `CalcInnerSize` / `CalcMBaseSize` / `Split` | 初始 S2 基块为 512；PA 且 P<512、`512%P!=0` 时向下对齐为 `floor(512/P)*P`，再按源次序处理短序列和分核。例：P=112、长 KV 时基块为 448；P=528 时仍为 512，仍须跨页复制。保留源 FD 候选与既有无 split 映射，不额外运行一次切块启发式 |
| MLA tiling `FillArch22MlaKvStrideParams`；Cube `Init` 中 K/V/RoPE 初始化 | 源分别下发 `keyBnStride/valueBnStride/keyRopeBnStride`。FA3 `v` 同时映射源内容 K 与 V，FA3 `k` 映射 RoPE K；两个 FA3 cache 的首维 stride 必须独立，不能因共享内容 KV 而共享 RoPE stride |
| `attention/common/op_kernel/memcopy/offset_calculator_v2.h::OffsetCalculatorImpl<..., GM_KV_PA_BNBD, ...>`；`parser.h::BlockTableParser` | 逻辑 token 先拆页号/页内位置，再查页表取物理页；源地址结果是 uint64。页表行距以 int32 表项计数，K/V stride 以各自输入元素计数，须独立传递；源未在这些 device helper 中校验页号范围，迁移消费者须在取 cache 地址前校验 |
| 同目录 `attn_copy_gm_to_l1.h::CopyKKropePAGmToL1` / `CopyKvGmToL1` | QK 同时搬运内容 K 与 RoPE K，每段共用一次页号查询、各自计算地址；PV 单独搬 V。复制在页边界拆段，保留 L1 NZ 偏移、内容/RoPE 两半的顺序及事件，不先 gather 完整 cache |
| `op_kernel/arch22/fused_infer_attention_score_v3.cpp` 的 BSH/TND、KVBSH/KVBNSD、`PAGEDCACHE_MLA_TILING` 分支 | 双 dtype、非 FD ND PA 有独立实例；FD 和 NZ 是其他分派，不能用编译成功或非分页 probe 代替 PA 原生运行证据 |
| `tests/pytest/test_fia_mla_d512_pa_dim0_noncontiguous.py` 的 `_build_block_table` / `STRIDE_PROFILES` / `_cpu_golden` | 源覆盖乱序页、尾页、无关物理页、未用列 -1 和三路独立 dim0 padding；golden 的内容 K/V 独立。迁移重构为共享内容 KV 后保留独立 RoPE/内容 stride，不把源独立 K/V 数据原样当成 FA3 可表达用例 |
| 官方 `flash_api.cpp::mha_fwd` / `mha_fwd_get_scheduler_metadata` | 页表末维连续且保存 `page_table_batch_stride`，K/V 保存各自 stride；前向由页表列数和 P 推导容量，生成器仅收到 P 与逻辑长度。metadata 不绑定页号与物理 cache 地址，不能在公开签名之外增加页表参数 |

CANN 表内路径仍相对于第 1 节 FIA 根目录，标明 `attention/common/` 的路径相对于源仓根目录。页大小/分派的源码可表达性与实机正确性分开验收；源测试仅提供用例依据，实际运行证据见 status。

##### 地址与有效页

对样本 b 的有效 KV 长度 L、token t，令 `j=t/P`、`r=t%P`、`p=page_table[b,j]`。FA3 cache X 的元素偏移为 `p*X.stride(0)+r*X.stride(1)+h*X.stride(2)+d`，基址为含 storage offset 的 `data_ptr()`；页表地址独立为 `b*table.stride(0)+j`。页号先以有符号 int32 校验 `0<=p<Np`，再转宽整数做乘加，不能先把 -1 转为无符号地址。每次复制长度为 `min(剩余行数,P-r)`；不能假定相邻逻辑页物理连续，也不能把整个 512 行块只查一次页表。

只要求每样本前 `ceil(L/P)` 项是合法页号，其余列可保留 -1 等 poison，不能因为 padding 中的哨兵拒绝合法调用。长度先检查 `0<=L<=C`，然后检查有效页；即使对应 Q-used 为 0，KV 长度和其有效页仍校验，反之 Q 前缀/used 也不因 KV 为 0 而跳过。host 检查静态 shape、dtype、同设备、stride、storage 和算术边界；页表值和动态长度由设备读取，禁止 D2H。所有参与执行的核在 attention 同步、TPipe 和 O/LSE/workspace 写入前完成消费者校验，错误执行 release Trap；运行期间页表不可被其他 stream 改写。

##### 空值与分层

B/Tq/Sq 为零的合法调用保留空输出 shape，但仍验长度、页表描述与必要的有效页。P 始终为正；M=0 或 Np=0 时所有 L=0 是必要条件。固定官方 PA 前向计算 `total_k=B*P`，不按物理 cache 的 `numel` 或 `M*P` 判断 host 全局空 KV；正常非空页表/物理池中 L=0 的有效行按设备空行规则输出 `O=0、LSE=-inf`。这个 host 分支本身不足以证明空表/空物理池的整条设备路径可调用。消费者通过只说明描述、页号和调度合法；Copy、attention 与公开路径仍须分别验收，未接通组合明确拒绝，不预先套用非分页 `+inf`。CANN `CheckBlockTable` 拒绝零元素页表，FA3 所需的空值适配须与原生 CANN PA E2E 分开记录。

**分层。** CPU 的 PA 地址描述优先复用 BSND/TND 的完整 storage 校验；PA 调度复用既有长度归一、源 SplitCore、无 split 端点与资源预算，仅增加源 PA 切块和逻辑容量信息。新增协议负责描述调度，消费者负责当次页表/物理 cache 校验，Kernel 负责逻辑 token 循环，Block/Tile 负责页内搬运；沿用现有 Cube/Vector 数值与同步流程。独立组件放在本仓 910 MLA 目录，保留 CANN 来源/版权，不修改 CATLASS 子模块或 CANN 源仓。

##### 静态地址描述与 CPU 动态检查

[`mla_paged_layout.h/.cpp`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_paged_layout.cpp) 的三个接口必须区分：`BuildPagedCacheLayout` 只接收 shape、stride、storage offset/字节数等静态描述，不读取页表或 cache buffer，可由 host 为 NPU 视图构造描述。**仅 `ValidatePagedCachePages` 和 `GetPagedCopySegment` 读取 CPU host buffer**，不得传设备指针或为调用它们而 D2H。生产所需的动态检查由设备消费者执行；Q/qv 地址仍由 BSND/TND 组件检查。

| 接口 | 职责与边界 |
| --- | --- |
| `BuildPagedCacheLayout` | 分离逻辑 B、M、`C=M*P`、物理 Np、P、页表行距及 K/V 独立 storage；复用 BSND 完整 storage 校验，要求 `Np<=INT32_MAX+1` 使所有物理页号可由 int32 表达；不承担调度的 C≤16384 限制 |
| `ValidatePagedCachePages` | 校验每样本长度和全部有效页，仅读取前 `ceil(L/P)` 项，不读取 padding；不因 Q-used=0 跳过样本 |
| `GetPagedCopySegment` | 返回有效逻辑范围的首个页内段及 K/V 独立元素偏移，由调用方按 `rowCount` 迭代；零 count 返回零描述，失败保持输出不变 |

[`PA 地址测试`](../../tests/test_flash_attn_npu_v3_mla_paged_layout.py) 从固定 Git blob 提取 offset、页表 parser、QK 联合 Copy 和 PV Copy，替换搬运操作为 CPU 参数 trace，与组件和独立标量 oracle 对比。它验证地址、页边界、L1 偏移参数和查询次数，**不验证设备 Copy/L1 数值、同步、attention 或性能**；注入源 layout getter 的外层 stride 也不证明原生 FIA host 接受这些视图。大于 uint32 的 cache 地址在该套件中只作算术检查。

##### 独立设备 Copy

[`PagedCache` / PA Copy Tile](../../csrc/ascend910/flash_attn_npu_3/mla/tile/mla_paged_copy.hpp) 复用上述描述、页内分段语义和已有 ND→NZ Tile。调用方须先完成当次完整描述、长度和有效页校验，保证 QK 两坐标使用相同的合法 token 范围，并管理 L1 容量、事件与输入生命周期。Copy 不重复运行 metadata 校验；零 count 不读取页表或 cache，不能据此跳过前置校验。内容 K 与 RoPE K 的页号每段只读取一次，两路地址分别按 V/K stride 计算；PV 只搬运 V。

[`设备搬运测试`](../../tests/test_flash_attn_npu_v3_mla_paged_copy.py) 用独立 C1V2 probe 导出 L1 NZ，逐元素比较固定源 Copy 与原始 host 输入的标量索引；覆盖全部页大小、内容/RoPE 两半、PV 四个 D 块、双缓冲循环、零 count、真实 offset/stride 与大地址。源 Copy 的测试适配只注入 layout stride；无法等价表达的零 stride 或 >uint32 页表行距在测试侧先物化源输入，不能宣称原生 FIA 接受相同物理视图。诊断导出另同步 MTE3，固定源与迁移侧使用相同的探针缓冲/事件流程。

[`Copy 性能驱动`](../../tests/mla_paged_copy_profile.py) 单列两种 dtype 的 12 例预定矩阵。计时包含页表查询、GM→L1、每 tile 的 512 字节 MTE1 消费及双缓冲事件；初始化和 L1→GM 诊断导出不在计时内。BSND/TND 标签描述单/多 Q 块对应的复制工作量，Copy 本身不接收 Q tensor，也不代表 attention 调度或公开 PA E2E。

##### 独立 BSND/TND PA attention

[`MlaPagedBsnd` / `MlaPagedTnd`](../../csrc/ascend910/flash_attn_npu_3/mla/kernel/mla_paged.hpp) 复用既有任务环、Cube/Vector 数值流程与事件顺序；v6 的五字 span 提供 Q 分配起点、Q-used 和非累计 KV 长度。BSND Q 由样本槽位及独立 stride 寻址；TND Q/qv 使用已校验的分配前缀起点及各自 token/head stride，Q-used 只限制有效行，不移动后续样本。TND 的 O/LSE 为 `[Tq,Hq,512]` / `[Hq,Tq]`，总 Q 使用完整分配量。Cube 的编译期 PA 分支将 QK 联合内容/RoPE 与 PV 交给分页 Copy Tile，源调度 S2 不用于物理 cache 地址。各核在 `TPipe`、同步与 O/LSE/workspace 写入前完整消费 v6，检查当次长度、完整 cache 描述和所有有效页，包括 Q 分配或 used 为零的样本。

两种 Q 布局的独立 attention 都为空 query 保留空输出；非零 Q 的零 KV 有效行输出 `O=0、LSE=-inf`，合法空页表或空物理池也使用该 PA 设备空行语义。所有样本 KV 长度为零时使用已校验的单工作 C1V2、零 workspace 调度，不建立计算缓冲或读取 Q/cache。静态非空容量可保持全核发射，其余核不访问输出或预留 workspace；预算兼容见 [6.7.4](#674-bsnd-pa-静态发射计划)。公开 PA metadata 和前向仍各自独立验收。

[`BSND`](../../tests/test_flash_attn_npu_v3_mla_paged_attention.py) / [`TND 独立 attention 测试`](../../tests/test_flash_attn_npu_v3_mla_tnd_paged_attention.py) 比较未修改的固定源 PA/non-FD Kernel 和双数学 reference；数学侧从原始分配前缀、Q-used、长度和页表逐 token 索引，不读取 metadata span。固定源保留可原生表示的 K/V 独立页间距，其他 Q/cache 行/页表 stride 在测试侧计时前物化。源 TND Q 按原始 used 独立压紧并提交累计尾值，源 PA KV 仍接收非累计长度，源 `[Tused,Hq]` LSE 转为 `[Hq,Tused]` 后比较；迁移侧保留完整分配间隙和 poison。此转换不能解释为原生 FIA host 支持所有视图。非空有效行 O/LSE 要求与固定源逐元素一致，零 KV 的源 `+inf` 与 FA3 `-inf` 分开检查；零工作调度/空表由数学对照验收，不伪造源可调用结果。独立 probe 的调用耗时不是公开 PA E2E。

##### CPU/AICPU 调度

[`ParsePagedLengths` / `BuildPagedSchedule`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_paged_schedule.h) 在 `mla_tnd_schedule.cpp` 中复用分配/used 归一、变长 SplitCore、无 split 端点映射与资源预算；由独立 v6 生成器使用，没有公开调用方。输入仅为逻辑几何及执行 CPU/AICPU 可读的长度数组，不接收页表、物理页池或 stride。P 必须为 16..1024 的 16 倍数，C 必须为 0..16384 且为 P 的整数倍；先限制 C 再计算虚拟 batch 总量，地址组件本身仍不施加这一调度上限。

Q 使用 BSND 分配容量或 TND 分配前缀，可选 Q-used 不移动 64 位分配起点；KV 必须显式提供逐样本 `int32[B]` 长度，包括 Q 分配/used 为零时，只有 B=0 才允许空指针与零计数。内部借用 BSND 虚拟分配校验 `0<=L<=C`，随后丢弃其 KV 起点；输出仅保留 Q 分配起点/容量、实际 Q/KV 长度及观测最大值，不把虚拟密集 KV 起点当作物理地址。输入数值全部校验后复制，失败保持输出不变。

调度分别保存逻辑 C 与固定源常规 PA 的 `s2Size=ceil(max(L)/P)*P`，不使用源最大 workspace 分支的 `s2Size=C`。Q 的 `s1Size` 仍按分配容量，TND 使用观测最大分配段长；基本块先按 PA 规则向下对齐 512，再裁到 `s2Size`，之后只运行一次源 SplitCore 和既有无 split 映射。B=0、Q 分配为空或所有 KV 长度为零时返回单校验核、零基本块/端点/workspace；分配非空、KV 非零而 Q-used 全零时保留零成本调度和 workspace。这是 CPU 调度边界，不定义 PA 空行 O/LSE，也不表示空表或空池已接通。

[`PA 调度测试`](../../tests/test_flash_attn_npu_v3_mla_paged_schedule.py) 从固定 Git blob 编译原始 parser/tiling/SplitCore，独立输入适配不调用迁移归一函数；逐字段比较分块、完整端点、FD 与 workspace，并用标量枚举有效任务和成本。原生 FD 候选另用原始 tiling 加迁移 SplitCore 对照完整字段，保留其原始结果，与 FA3 无 split 映射分开。零分配/全零 KV 的校验调度单独验收，不直接执行原始 `CalcInnerSize(0)`；这些证据不替代设备或公开 PA 验收。

### 6.5 数值路径与资源布局

QK 的 L1 循环顺序为 N→K→M，M/N 子块均为 128。576 维并非先算完整内容再算完整 RoPE：两组 288 依次为 `[content 0:256, rope 0:32]`、`[rope 32:64, content 256:512]`，每组按 K=96 的 L0 片累加到 FP32 L0C。Q 的 64 维 RoPE 一次复制跨越相邻 Q/P 槽，尾 M 用 padding 偏移保持相邻关系；复制前必须同时取得两个槽的释放事件。最终 QK 以 FP32 ND 写入 GM，再乘 scale。

Vector 保留 `SoftmaxFlashV2<float,...WITHOUT_BRC>` 和其动态 tiling；随后执行 `AmlaVecCompute`：计算 `round(-max/ln2)`、指数缩放因子、FP16/BF16 因子往返转换及 `cof/eps` 修正。P 在缩放后以 `CAST_ROUND` 转成输入 dtype。跨 S2 块的 `nUpdate` 含 `-30` 截断、`1.5` 修正、`1e-6` 和 `2^23`，以 int32 原子加更新 FP32 PV 缓冲的位表示；不能替换为常规 `O *= exp(old_max-new_max)`。

PV 同样按 N→K→M 循环，M/N 为 128、K 的 L1/L0 片分别为 256/128。首 S2 块覆盖写 FP32 GM，后续块在 nUpdate 完成后以 FP32 原子加写回。V2 每块消费完成事件，仅在最后 S2 块写输出：保留源对 `abs(PV)>1e10` 的清零、除以 `aMlaSum`、FP16 `CAST_ROUND` / BF16 `CAST_RINT`；LSE 则从原 softmax 的 max/sum 计算自然对数。单块范围也经过 aMLA 因子转换，不能省略这段数值路径。

| 资源 | 固定源布局与生命周期 |
| --- | --- |
| L1 | 每槽 `128*288*2=72 KiB`，Q/P 四槽与 K/V 三槽共 504 KiB；QK 与 PV 共用，迭代器跨任务延续。Q/P 在各自首个 N 子块加载，后续 N 复用，最后 N 才释放 |
| L0A / L0B / L0C | 分别为 `2*32 / 2*32 / 2*64 KiB`；A/B 共同跟随 K 片，C 轮换与 M/N 循环绑定；首 K 清零，最终 K/写回以 unitFlag=3 配对，其余累加用 2 |
| Vector UB | input1 `2*32 KiB`、input2 `2*8 KiB`、output1 `32 KiB`、output2 C1V2/C1V1 为 `4/8 KiB`、tmp1 `32 KiB`、mask tmp `8 KiB`；n/cof/aMlaSum/max/exp/sum 各 `2*1 KiB`，默认 max/sum 各 `1 KiB`，合计 `170/174 KiB`。即使无 mask 也保留源分区复用 |
| GM workspace | Kernel 顺序为双槽 QK FP32、P 输入 dtype、PV FP32、vec2 FP32。QK/P 用 `loop%2`，PV 用 `bn2IdxInCurCore%2`（每个新 S1G 任务递增），不是统一使用 tile 序号；调用间不得共用可变 scratch |
| workspace 预算 | 源 host 普通部分为 `2*aicNum*(6*R1+8*R2+8*mBaseSize)` 字节，包含 nUpdate/sum 的额外预留；设备按 `GetBlockNum()` 排四个段，nUpdate 实际在 UB 生成并操作 PV 的 int32 别名。保留预算与段界，不把预留当成新的 GM 数据流；框架 `libapiSize` 与用户 workspace 单列 |
| 任务 / 行分工 | 三项 `RunInfo` 环、两槽 GM；外层 M 基块最多 256。C1V2 将 M 按 16 行边界分给两条 Vector，M<=16 时第二条零行仍参与事件；C1V1 全部行由一条 Vector 处理，V1 内层每次至多 128 行 |

### 6.6 同步依赖与组件复用结论

`ExecuteTask(t)` 在 Cube 上依次发出 `QK(t)`、`PV(t-1)`；Vector 上依次执行 `V1(t-1)`、`V2(t-2)`。创建结束后仍须排空任务环。下表均为源 mode 2 跨 Cube/Vector 事件，编号不与本地 HardEvent 混为一个命名空间。

| 源事件 | 生产 → 消费 | 保护的数据 / 时机 |
| --- | --- | --- |
| `syncC1V1=7` | Cube `PIPE_FIX` → V1 wait | QK 的 FP32 GM 写回完成 |
| `syncV1C2=8` | V1 `PIPE_MTE3` → Cube wait | P 写回完成，PV 可以加载 |
| `syncC2V1=4` | Cube `PIPE_FIX` → 下一次 V1 nUpdate wait | 前次 PV 完成后才能修改其位表示；Cube 初始化先发一个 token，Vector 结束回收最后一个 |
| `syncV1NupdateC2=5` | V1 `PIPE_MTE3` → PV 首次 Fixpipe 前 wait | nUpdate 原子加完成；首块跳过 nUpdate 计算也必须发/收事件 |
| `syncC2V2=9` | Cube `PIPE_FIX` → V2 wait | PV 写回完成；V2 在非末块也消费事件 |

`syncV0C1=6` 在源常量中存在，但此非量化 dense 路径不使用。L1 Q/P 的本地事件为 2/3/4/5，K/V 为 6/7/1，保留 `MTE1_MTE2` 释放和 `MTE2_MTE1` 就绪配对；L0 的 `M_MTE1` / `MTE1_M` 与 unitFlag 共同约束覆盖写。Vector 保留 input 的 `V_MTE2` / `MTE2_V` 和 output 的 `MTE3_V` / `V_MTE3`；`AllocEventID/FreeEventID` 和零行同步不能被提前返回绕过。启用 LSE 时，源 `InitOutputSingleCore` 的初始化与 `SyncAll` 也在 dense 执行链内。

固定 CATLASS 的复用结论如下；源码核对确定可用条件，组件实际接入仍须通过设备等价验收。

| 组件 | 核对结论 |
| --- | --- |
| `gemm/tile/tile_mmad.hpp::TileMmad` | 优先复用。FP16/BF16、M/N 已向 16 对齐时，可透传源的 K、initC、unitFlag，且包含同样的 `(M/16)*(N/16)<10` PIPE_M barrier；源 `MatmulKPP` 的 K 循环和加载/释放时机仍由本仓 Block 保留 |
| `gemm/tile/atlasa2/copy_l0c_to_gm.hpp::CopyL0CToGm` | FP32、RowMajor、无量化/无 ReLU 的特化可映射源 Fixpipe 的 M/N、src/dst stride 和 unitFlag；原子模式由源 Block 在同一位置开关，不能交给默认 epilogue |
| GM→L1、L1→L0 Tile | 在本仓封装源分段复制与加载，逐项匹配 NZ/ZN、转置、尾块和 stride 后再复用。`CopyGmToL1DynamicOptimized` 对小 M 改走 DataCopy，不能直接代替源 ND→NZ；Q 的跨槽 RoPE 复制也需专门保留 |
| `epilogue/block/block_epilogue_amla_tp1_softmax.hpp` / `block_epilogue_amla_tp1_rescale_o.hpp` | 同有 aMLA 思路，但使用自有归约、UB/事件和部分不同 Cast 模式；不能整体替换源 Vector。本仓按源封装 `SoftmaxFlashV2Compute/AmlaVecCompute` 等函数 |
| `arch/resource.hpp`、`arch/cross_core_sync.hpp` | 可复用资源/简单 flag 表达；Resource 构造会 `pipe.Destroy()`，必须统一资源所有权，不能与另一套 TPipe 分配混用。带 reverse-depth 的同步封装会增加等待，保留源事件协议 |

### 6.7 生产接入与构建

[Python 包装与 fake](../../flash_attn_npu_3/flash_attn_npu_interface.py)、[C++ 通用分派](../../csrc/ascend910/flash_attn_npu_3/flash_api.cpp)、[MLA 静态校验](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h) 与 [专用前向](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp) 共同实现入口。下文描述非分页 512+64 路径的分派和发射机制；当前开放范围见 [status](status.md#当前可调用能力)，launcher/Kernel 对应关系见 [第 6 节开头](#6-代码结构)。

#### 6.7.1 非分页入口与记录选择

长度 tensor 均为同设备、连续 int32；分配前缀为 `[B+1]`，实际长度为 `[B]`。三种布局的 Q/K/qv/V 均允许末维连续、外层非负 stride，完整 storage 在发射前校验。`max_seqlen_*` 描述分配上界，used 不移动后续样本的分配起点；具体规则见 [6.3](#63-长度与布局调度)。

| 公开入口 | 输入与长度 | 记录选择 |
| --- | --- | --- |
| `flash_attn_func` | BSND Q/KV，使用整容量 | 每次内部 v1，无显式复用参数 |
| `flash_attn_varlen_func` | TND Q/KV，两组分配前缀、两组 max；可选 `seqused_q/k` | 任一 used 存在时每次内部 v5，否则 v3；无显式复用参数 |
| `flash_attn_with_kvcache`，BSND Q | Q 整容量，只读 BSND cache；`cache_seqlens` 为 None（整容量）、`0..Sk` 整数或长度 tensor | 无长度且无记录走 v1，否则内部 v2 或显式 v2 |
| `flash_attn_with_kvcache`，TND Q | Q 使用完整分配段，须给 Q 前缀和 `max_seqlen_q`；只读 BSND cache，KV 长度同上 | 内部 v4 或显式 v4；整数长度按 cache 的 B 物化 |
| `get_scheduler_metadata` | 必须给 KV 长度 tensor，不接受 None/整数；`headdim=64, headdim_v=512` | 无 Q 前缀生成 v2，有 Q 前缀生成混合 v4；精确字数见 [第 5 节](#5-metadata-设计) |

内部 `_flash_attn_forward` / C++ `fwd` 的 used 分支共用 `UsedForward`、v5 生成/校验与 `MlaUsed`，不增加公共签名：

| 内部输入 | 记录选择与边界 |
| --- | --- |
| BSND Q/KV | 有 `seqused_q` 时每次内部 v5，可选 `seqused_k`；无 Q-used 沿用 v1/v2；Q-used 与显式 metadata 组合拒绝 |
| TND Q/KV | 前缀、max、可选 used 及 v3/v5 选择与公开 varlen 相同；拒绝显式 metadata |
| TND Q + BSND KV | 无 Q-used 沿用 v4；有 `seqused_q` 时每次内部 v5，可选 `seqused_k`，复用 TND used launcher；Q-used 与显式 metadata 组合拒绝，混合入口始终拒绝 `max_seqlen_k` |

#### 6.7.2 校验、输出与发射

非分页 512+64 接入只接受 `causal=False`、`window_size=(-1,-1)`、`softcap=0`、`num_splits=1`、`pack_gqa=None`、`sm_margin=0`、`attention_chunk=0` 和 `rotary_interleaved=True`。PA、追加 KV、rotary/量化输入、cache 重映射/leftpad 等选项由 Python 与 C++ 明确拒绝。metadata 生成器要求 `has_softcap=False`、`max_seqlen_k_new=0`，拒绝数值 scale/softcap 兼容关键字。`page_size=None` 沿用非分页 v2/v4；指定合法 P 时生成 v6，公开 PA 前向仍明确拒绝。

func/varlen/内部前向的 `num_splits` 默认 1，kvcache/metadata 默认 0，调用 MLA 后两者须显式传 1。生成器 `qkv_dtype` 默认 BF16，必须匹配当次输入。前向默认 scale 见 [3.3](#33-共同契约)，显式值须能表示为有限 FP32；scale 不进入 MLA metadata。

公开包装校验选项，真实 C++ 与 fake 独立校验张量、scale 和记录外形；设备校验记录内容。Autograd 层透传，MLA 输入拒绝梯度；内部预分配 `out_` 也拒绝，以保持 functional custom op 的 alias 契约。BSND、TND、混合前向分别经 `DenseForward`、`TndForward`、`TndBsndForward`；used 分支经各自边界检查后共用 `UsedForward`。专用 custom op 与通用 `fwd` 汇入同一实现，不维护两套数值路径。

混合前向的 B/Sk 来自物理 cache，总 Q 来自 Q shape。输出遵循 [3.3](#33-共同契约)：BSND O/LSE 为 `[B,Sq,Hq,512]` / `[B,Hq,Sq]`，TND 为 `[Tq,Hq,512]` / `[Hq,Tq]`。请求 LSE 时 func/varlen 返回二项，kvcache 返回四项；未 split 的两个 accumulator 为同设备空 tensor，dtype 随 PyTorch 默认浮点 dtype（通常 FP32），真实/fake 一致。

`UsedForward` 只按静态分配容量预算 launch/workspace，前缀与 used 值留在设备检查。混合布局的 Q 起点来自 Q 分配前缀，KV 起点为 64 位 `b*Sk`；KV-used 缺省时使用整个 cache 槽位。O/LSE 保留分配区 shape，有效区及空值的输出承诺见 [5.6](#56-v5分配与-used-记录)。

各前向查询平台，为每次调用分配 O/LSE 和独立 workspace。`DenseForward` **仍在 host 调用 `BuildDenseSchedule`**，用于 launch、分配预算及独立校验 context；计算切块与端点来自设备记录。显式复用省去 AICPU 生成，不表示消除了全部 host 调度。

v2/v3/v4/v5 的非空物理容量按平台全核 blockDim 生成和发射，工作核数取设备记录；按源 `CalcNormalWorkspaceSize` 的最大 M=256 预算 workspace。TND 按静态总容量和 max 上界分配，不按矩形 `B*maxQ*Hq` 拒绝紧凑 Q。该预算不决定实际切块或数值路径。空 batch/query/KV 容量使用单校验核、零 workspace；空 KV 先校验记录，再初始化 O/LSE：BSND v1/v2 由 `DenseForward` 在同 stream 排入填充，TND/混合/used 由对应 Kernel 写入。非空容量中的全零实际长度仍保留完整资源校验。

#### 6.7.3 实例与构建依赖

[`generate_kernels.py`](../../csrc/ascend910/flash_attn_npu_3/autogen/generate_kernels.py) 生成 FP16/BF16 C1V2 实例，BSND、TND、混合布局及 BSND/TND used 各用对应 launcher，混合 Q-used 复用 TND used launcher，共用 Kernel / Block / Tile。两种 dtype 均为 2 字节，调度/workspace 相同，数值舍入按 [6.5](#65-数值路径与资源布局)。修改实例集合应改生成器，不仅手改 autogen。

CANN 9.0 Bisheng ASC launcher 未提供调度模式参数；MLA 发射使用线程局部作用域给实际 ACL launch 设置源 `BATCH_MODE`，保留其他属性及错误码，普通 FA3 发射原样透传。绕过已验收 ACL 路径或作用域内不恰好发射一次时明确失败。

[`setup.py`](../../setup.py) 的 910 FA3 构建包含七个 AICPU 单元：普通 FA 的 `fa_metadata.aicpu`，以及 MLA v1–v6 的 `mla_metadata.aicpu`、`mla_bsnd_metadata.aicpu`、`mla_tnd_metadata.aicpu`、`mla_tnd_bsnd_metadata.aicpu`、`mla_used_metadata.aicpu`、`mla_paged_metadata.aicpu`。v6 已纳入编译及传递依赖，由公开 PA metadata 入口和独立测试桥发射；生产前向尚不发射 PA attention。生产 v4 生成器由 metadata 入口和混合前向共用，v5 由内部 BSND/混合 Q-used 和公开 varlen used 前向共用。编译选项、包含的头文件和 `.cpp` 均参与依赖检查；MLA 另生成设备预处理依赖，防止 host `-MMD` 漏掉 `__NPU_ARCH__` 内的 Kernel 头。已有 `.so` 不证明源码已生效，复现须核对加载路径、哈希和链接对象。

#### 6.7.4 BSND PA 静态发射计划

[`BuildPagedBsndPlan`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_paged_plan.cpp) 是供 BSND PA 生产接入使用的独立 host 适配器，当前尚未编入生产扩展或接入前向。输入只有 Q/qv/K/V 和页表的 shape、stride、storage 描述，以及 dtype、Q-used 是否存在和平台资源；不接收或读取 tensor 地址、长度值、页号及 metadata。复用既有布局校验，覆盖全部分配区，分离 `C=M*P`、物理池大小和两路 cache 的独立 stride。输出包括精确 v6 字数、逻辑 shape、四路布局、cache 描述、AICPU 平台参数和独立消费者 context；失败保持输出不变。

本适配器只接受 C1V2。它通过零 KV 容量的 `BuildDenseSchedule` 复用静态字段、平台和局部内存检查，不运行 SplitCore、不分配逐 batch 数组。B/Sq/C 任一为零时选单核和零 workspace，否则选平台全核，并固定生成器所有 `blockDims` 槽位；物理池大小及 Q-used 不参与此选择。workspace 按 M≤256、源 PA 的页大小对齐及逻辑容量预算，并检查含 libapi 的总字节数不超过 `INT64_MAX`。预算覆盖该静态几何下所有成功的 v6 调度，不预判动态长度能否通过源 tile/字段约束。

静态容量非空、实际 KV 全零时，v6 仍是单工作核、零 workspace 的记录，发射核数沿用静态选择。PA 消费者先完整校验当前长度、有效页、平台和记录，再允许调用方保留非零 workspace 预算；空记录的切块、端点、FD 记账和 workspace 载荷仍必须全零，Kernel 不访问预留区。静态空容量沿用零预算要求。该适配不改变协议或空行语义，也不开放公开 PA 前向。未来调用方仍须校验真实 tensor 的 rank/dtype/device、地址和长度/metadata 描述符，物化便利长度并管理异步生命周期。[CPU 计划测试](../../tests/test_flash_attn_npu_v3_mla_paged_plan.py) 与 [PA attention 测试](../../tests/test_flash_attn_npu_v3_mla_paged_attention.py) 分别验收静态计划及设备预算兼容。

## 7. 验收口径

**源实现对照：** 每个迁移模块须能追溯到固定 CANN 的文件/函数，注明结构重构和必要接口适配的对应关系。风格重构须经过构建、精度、边界、重复执行和相关性能检查；迁移结果以本仓实际构建和测试为依据。保留源级执行行为不承诺编译后二进制或耗时完全相同。

**功能和精度：** 对同一逻辑输入同时检查独立高精度 reference、低精度 PyTorch 对照和原 CANN。CANN 对照先统一 scale、mask、布局与共享 KV 语义；本次按 FA3 调整的差异以官方为准，不要求重现源不兼容行为。reference 不能调用待测 kernel，也不能在两个实现中共用同一个错误的长度/页表解析。复用仓库双 reference 精度准则，单独检查 LSE、inf 符号、NaN、空行和 padding；阈值在基线步骤固定，不为通过用例而放宽。

**阶段范围：** 性能只验收本步已迁移的能力和执行路径。在 FD 迁移完成前，全部约定形状均用固定 CANN 源的非 FD Kernel、相同逻辑输入/基本块/核数/端点，与迁移 Kernel 对照；布局转换的边界按 [7.1](#71-固定性能矩阵与阶段边界) 和 [7.2](#72-参考工具与对照边界) 单列。记录逐例 Kernel 耗时及独立 probe 调用耗时，不能把预分配 probe 的调用时间当成公开 E2E。原生 CANN 自动调度同样不启用 FD 的用例，另按下述公开 metadata E2E 门槛验收。原生 CANN 会启用尚未迁移 FD 的用例，其原生 E2E 仅记录能力差距，不作为非 FD 迁移的阻塞项；不因此前置 FD、开放 `num_splits=0` 或改变 `num_splits=1` 语义。

源执行路径是否已迁移，按固定源调度字段和分派证据判定，不能按耗时结果筛选。保留完整预定矩阵和原始数据；FD、mask、PA 等能力接通后，对应路径再纳入验收。CANN 没有公开的 FA3 `num_splits=1` 等价开关时，使用明确标识的独立源 probe 作组件对照，保留原生 CANN 分派及库，不将改装调用冒充原生 E2E。

**公开 E2E：** 用户明确的“不劣化”指 **走 MLA metadata 路径的 FA3 公开调用端到端（E2E）延迟**。在同一健康、空闲的 910B、同一输入/精度/软件栈上比较。固定 seed、预热、计时次数和同步方式，交替重复测量 CANN/新实现；保存每例耗时及运行信息。对双方执行路径均已迁移的约定代表用例，验收目标仍是 `FA3 metadata 路径 E2E / 原 CANN E2E <= 1.0`，不默认引入 5% 或其他放宽阈值，也不能用平均加速掩盖退化用例。噪声未能分辨时补测，不能宣布不劣化。

E2E 从公开 Python 入口开始，测至对应调用段的 O/LSE 完成，计入入口校验、调度、输出/workspace 分配、必要转换、发射、流间依赖和全部设备计算。输入 H2D 与一次性视图准备沿用基线的对称计时边界；两边使用一致的完成同步方式。只测内部 launcher、device event span 或 profiler kernel 时间均不能替代该口径。

每个代表用例须在采集前固定 metadata 生命周期和复用次数。预生成后复用时，记录复用前向的 E2E，并单列 metadata 生成时间；实际调用段需要生成或更新 metadata 时，从 `get_scheduler_metadata` 开始计时。跨层复用须记录实际次数 R、包含一次生成及 R 次前向的完整调用段 E2E 和摊销值，并与 CANN 完成相同 R 次前向比较，不能任意增加 R 隐藏准备成本。生成加一次前向与复用前向分别报告，不能将复用数据标为包含生成成本。

kernel 时间及无 metadata 调用时间用于已迁移组件的源行为对照和瓶颈定位；不能替代适用用例的 metadata E2E 门槛。实现按源对照完成 CATLASS 分层、FA3 接入和 metadata 支持后进行对应验收；E2E 优化限定在保持源设备行为的接口、调度结果生成/复用、重复准备和发射开销。设备计算瓶颈按源实现对照定位。

分别报告 kernel 时间、无 metadata 调用 E2E、metadata 生成时间、复用前向 E2E 及上述完整调用段 E2E；图模式单列。接口适配导致的必要转换计入相应端到端成本，CANN 基线使用固定版本的原生正确分派。对于原 CANN 的原生 NZ 等不同布局优势，另列原生最佳路径结果并解释表示差异，转换成本单独明确。

**覆盖：** 两种表示 × FP16/BF16 × dense/TND/PA × mask × 长短 KV × 单/多 query × 分组/尾块/stride；加 metadata 有无、重复使用、不同 scale、长度变更、split 自动/显式、graph replay。可采用代表性组合加针对性边界测试，不要求机械的全笛卡尔积，但每项源能力都要能追溯到用例或明确排除理由。

**版本可追溯：** CANN 源码 commit 不代表当前已安装算子版本。实测前记录实际生效的 OPP/算子库路径、构建版本或文件哈希，确认其对应本次来源；记录本仓扩展的实际加载路径与构建版本。910C 需要实现与资源配置兼容，按需求不作 910C 实机验收承诺。

### 7.1 固定性能矩阵与阶段边界

性能形状在采样前固定，双 dtype 使用同一矩阵，原生自动 FD 用例保留并单列，不按耗时删选。**矩阵定义以代码中的清单为准**；下表区分定义文件与采样驱动，不再在文档维护一份用例数量。公开和组件采样的转换边界见 [7.2](#72-参考工具与对照边界)。

| 范围 | 固定矩阵定义 | 采样驱动与计量边界 |
| --- | --- | --- |
| 整容量 BSND、S1G 多块、长 KV | [`mla_profile_cases.py`](../../tests/mla_profile_cases.py)：`CASES/S1G_CASES/KV_CASES/LONG_KV_CASES` | [`mla_dense_profile.py`](../../tests/mla_dense_profile.py) 同调度组件；[`mla_forward_profile.py`](../../tests/mla_forward_profile.py) 公开 E2E |
| BSND 实际 KV 长度 | [同一清单](../../tests/mla_profile_cases.py) 的 `BSND_CASES`；[BSND attention](../../tests/test_flash_attn_npu_v3_mla_bsnd_attention.py) 的 `PERFORMANCE_CASES` | [`mla_forward_profile.py`](../../tests/mla_forward_profile.py) 公开 E2E；[`mla_bsnd_profile.py`](../../tests/mla_bsnd_profile.py) 同调度组件 |
| 紧凑 TND | [TND attention](../../tests/test_flash_attn_npu_v3_mla_tnd_attention.py) 与 [`mla_varlen_profile.py`](../../tests/mla_varlen_profile.py) 的 `PERFORMANCE_CASES` | 组件由 [`mla_tnd_attention_probe.cpp`](../../tests/mla_tnd_attention_probe.cpp) 验收；公开驱动计入每次内部 v3 生成与 R=1/4 调用段 |
| TND Q + BSND KV | [混合 attention](../../tests/test_flash_attn_npu_v3_mla_tnd_bsnd_attention.py) 与 [`mla_mixed_profile.py`](../../tests/mla_mixed_profile.py) 的 `PERFORMANCE_CASES` | [`mla_tnd_bsnd_profile.py`](../../tests/mla_tnd_bsnd_profile.py) 同调度组件；公开驱动计量复用、生成加前向、内部生成及原生转换 |
| 三布局分配/used | [used attention](../../tests/test_flash_attn_npu_v3_mla_used_attention.py) 的 `PERFORMANCE_CASES` | [`mla_used_profile.py`](../../tests/mla_used_profile.py)；同逻辑输入/调度，源 TND 有效段在计时前压紧 |
| 内部 BSND/混合 Q-used | [BSND used](../../tests/test_flash_attn_npu_v3_mla_bsnd_used.py)、[混合 used](../../tests/test_flash_attn_npu_v3_mla_mixed_used.py) 的 `PERFORMANCE_CASES` | [`mla_bsnd_used_profile.py`](../../tests/mla_bsnd_used_profile.py)，混合使用 `--layout mixed`；生产 Kernel、AICPU、内部 wall 分开，无公开等价入口 |
| 公开 varlen used | [varlen used](../../tests/test_flash_attn_npu_v3_mla_varlen_used.py) 的 `PERFORMANCE_CASES` | [`mla_varlen_used_profile.py`](../../tests/mla_varlen_used_profile.py)；内部 v5 与原生必要转换计入 R=1/4 E2E |
| 独立 stride attention | [BSND stride](../../tests/test_flash_attn_npu_v3_mla_bsnd_stride.py)、[TND/混合 stride](../../tests/test_flash_attn_npu_v3_mla_tnd_stride.py) 的 `PERFORMANCE_CASES` | [`mla_bsnd_stride_profile.py`](../../tests/mla_bsnd_stride_profile.py)、[`mla_tnd_stride_profile.py`](../../tests/mla_tnd_stride_profile.py)；源连续物化及 used 压紧在计时前完成 |
| 生产 stride Kernel | [BSND](../../tests/test_flash_attn_npu_v3_mla_bsnd_stride_production.py)、[TND](../../tests/test_flash_attn_npu_v3_mla_tnd_stride_production.py)、[混合](../../tests/test_flash_attn_npu_v3_mla_mixed_stride_production.py) 的 `PERFORMANCE_CASES` | [`mla_bsnd_used_profile.py`](../../tests/mla_bsnd_used_profile.py) 的 `--layout bsnd-stride/tnd-stride/mixed-stride`；保留独立矩阵几何，使用生产核数，组件转换仍在计时外 |
| 公开 BSND stride | [`mla_profile_cases.py`](../../tests/mla_profile_cases.py) 的 `STRIDE_CASES` | [`mla_forward_profile.py`](../../tests/mla_forward_profile.py)；同物理视图，CANN 转换计入复用、生成加 R=1/4 E2E |
| 公开 TND/混合 stride | [TND](../../tests/test_flash_attn_npu_v3_mla_tnd_stride_production.py)、[混合](../../tests/test_flash_attn_npu_v3_mla_mixed_stride_production.py) 的 `PUBLIC_CASES` | [`mla_varlen_used_profile.py --stride`](../../tests/mla_varlen_used_profile.py)、[`mla_mixed_profile.py --stride-probes/--stride-producer`](../../tests/mla_mixed_profile.py)；同物理视图，计入原生转换和相应 metadata 调用段 |
| 独立 PA Copy | [`mla_paged_copy_profile.py`](../../tests/mla_paged_copy_profile.py) 的 `PERFORMANCE_CASES` | 同文件采样；P=16/112/128/256/528/1024，每种 dtype 各 12 例。相同物理视图、固定源 Copy 与迁移 Tile；不含 metadata、attention 或公开 E2E |
| 独立 BSND/TND PA attention | [BSND](../../tests/test_flash_attn_npu_v3_mla_paged_attention.py)、[TND](../../tests/test_flash_attn_npu_v3_mla_tnd_paged_attention.py) 的 `PERFORMANCE_CASES` | [`mla_paged_attention_profile.py`](../../tests/mla_paged_attention_profile.py)，TND 在三个路径参数后追加 `tnd`；同调度固定源 PA/non-FD Kernel，迁移计时包含 v6/当次页表校验，metadata 生成与测试侧物化/有效 Q 压紧在计时外；原生 FD 分派单列 |

源 host 把 decode Hq 限于 1..128，因此相应公开性能形状用 Sq 与 Hq 的乘积覆盖 S1G 边界；这不是 FA3 的 Hq 限制。更多头数由固定源 Kernel 与独立数学对照验收。整容量四组作同调度非 FD 组件对照，原生分派同路径的子集再作公开 E2E；`BSND_CASES` 的实际长度另行验收。公开全零实际长度同样计入对应路径的 E2E。

调度验收比较固定源的基本块、完整端点、使用核数、FD 字段与 workspace，并独立枚举任务，检查无遗漏/重复及每核成本。覆盖 B=0/1/多 batch、平台核数两侧、C1V1/C1V2、M 的 16/128/256 边界及尾块、KV 的 16/32/128/256/512 边界；实际长度增加 0/1/511/512/513/8193/16384、等长短序列、非均匀/全零长度及端点外零尾部。对应 [dense 调度检查](../../tests/test_flash_attn_npu_v3_mla_schedule.py) 与 [BSND 长度调度检查](../../tests/test_flash_attn_npu_v3_mla_bsnd_schedule.py)；oracle 从固定 commit 提取，不调用待测调度。

功能检查另覆盖 M 的 16/128/256 边界、非整齐 M=129/255、KV 对齐两侧及长 KV、内容/RoPE 各半独立扰动、非零 offset、scale 默认/零/负/显式值、跨块最大值变化（含 aMLA `-30` 截断）、双缓冲/任务环、重复执行和多流复用。固定源同调度非 FD Kernel 的 O 和非空行 LSE 要逐元素一致，再通过独立双 reference；零 KV 行单独检查源 LSE=`+inf`、迁移 LSE=`-inf` 的 FA3 适配，不把它判为数值误差。全局空容量另按 3.3 验收。原生自动 FD 数值顺序不同时分别检查数学精度，不强求非 FD 与 FD 逐元素一致。

公开性能采样采用双 dtype、预热 5 次、五轮各 20 个 wall 样本、R=1/4，同一进程对双方固定相同 CPU 亲和性；单列 metadata 生成、Kernel 和分派证据。varlen 当前没有独立公开 v3/v5 生成/复用入口，逐次生成计入前向，AICPU 时间由 profiler 单列。噪声不确定时保留原数据并补测。全局空容量在固定 dense CANN 基线之外，耗时只作诊断；独立 ctypes 生成器或记录校验驱动的调用耗时也不代表 attention 或公开 E2E。独立 TND 与混合布局的同调度 Kernel 与 probe 调用时间分别报告；公开入口接通时仍须补公开调用段验收。

M06 起的长度/布局检查必须区分分配位置与有效范围：FA3 带起始 0 的 `cu_seqlens` 决定样本起点，`seqused` 限定使用范围，不能按缩短后的累计和重排后续样本。对未使用 query 槽位只断言固定官方定义的输出区域。独立 oracle 不得与待测端共享长度/页表解析；全零实际长度、尾部零样本及 poison padding 单独覆盖。新增公开能力按本节同一路径 E2E 口径验收，TND varlen 的内部生成计入完整调用段。

### 7.2 参考工具与对照边界

[独立数学 reference](../../tests/common/mla_ref.py) 只接收规范 BSND 逻辑张量：高精度用 FP64，低精度用输入 dtype 点积与 FP32 softmax，O 为输入 dtype、LSE 为 FP32。两者不调用 FA3/CANN 或普通 attention golden；[CPU reference 测试](../../tests/test_flash_attn_npu_v3_mla_reference.py) 另用独立标量运算与手写可见列矩阵验数学、scale、GQA、stride、mask、softcap 和空值。reference 不自行解析 TND/PA、metadata、split accumulator 或图，调用方须独立解读原始输入。

`fa3_to_cann_inputs` 显式传同一 scale，prefill 可生成保留 row stride=192 的 split 视图或 combined 表示；反向映射要求 512 维 K/V 为同一 storage view，不接受数值暂时相等的独立 buffer。下面的布局转换均位于测试对照侧，生产前向不调用原 CANN attention。

| 对照工具 | 独立输入适配与比较范围 | 计时边界 |
| --- | --- | --- |
| [BSND 实际长度 attention](../../tests/test_flash_attn_npu_v3_mla_bsnd_attention.py) | 按原始有效长度逐样本切片调用双 reference；非零容量中的零 KV 单独检查迁移 LSE=-inf | 同调度固定源组件 |
| [紧凑 TND attention](../../tests/test_flash_attn_npu_v3_mla_tnd_attention.py) | 按独立逻辑段长切片；源用数值构造的 int64 累计尾值，不读取 v3 快照；源 LSE 转为 `[Hq,Tq]` | 同调度固定源组件，不等于公开 E2E |
| [混合 attention](../../tests/test_flash_attn_npu_v3_mla_tnd_bsnd_attention.py) | Q 按独立段长、KV 按 `b*Sk`；padding/空 Q cache 用 poison。源独立 TND Q + BSH KV 实例按 `GetKvActSeqMode` 接收累计 KV 尾值，物理 offset 仍为 `b*Sk`；CPU 调度 oracle 用非累计 KV | 原始源 Kernel header 不变；独立实例不是原生公开混合分派 |
| [used attention](../../tests/test_flash_attn_npu_v3_mla_used_attention.py) | 分配段累加起点、原始 used 限定有效区，不读 v5 span。源 TND 同时以累计长度决定起点/长度，因此独立压紧有效段；BSND 轴保留物理容量，迁移侧保留分配间隙和 poison | 源压紧在组件采样前完成；只比较有效 O/LSE 行，零 KV 符号差异单列 |
| [原生 BSND 执行器](../../tests/mla_cann_baseline.py) / [基线用例](../../tests/test_flash_attn_npu_v3_mla_cann.py) | `CannCall` 覆盖无 mask D01/D02/P01/P02、FD01、长 prefill 和实际 KV 长度；原生自动 split。不覆盖 TND、PA、mask、全局空容量 | 安装包结果只有在对应固定源码构建后才可用于源性能验收 |
| [公开 TND](../../tests/mla_varlen_profile.py) 的 `CannTndCall` | 两组 FA3 前缀转换为不带 0 的累计尾值；源输入为紧凑 TND，LSE 转为 `[Hq,Tq]` | H2D/一次性视图在计时外，原生 Python 长度提交在计时内 |
| [公开 varlen used](../../tests/mla_varlen_used_profile.py) 的 `CannTndUsedCall` | 根据原分配段与 used 构造设备索引；每次压紧有间隙的轴，FIA 后将 O/LSE 恢复到 Q 分配位置；缺省/整段 used 不复制该轴 | gather、分配、scatter、长度提交和完成同步均计入源 E2E；固定索引准备在计时外 |
| [公开混合](../../tests/mla_mixed_profile.py) 的 `CannMixedCall` | 原生 parser/checker 要求连续 Q/KV 同布局；每次从 BSND cache gather 紧凑 TND K/V。全零实际 KV 转为源空容量分派，保留其 tiling key | 两次 gather、分配、长度提交及 Q/qv 布局转换计入源 E2E；H2D/固定索引/零拷贝视图准备在计时外 |

混合 stride 沿用最后一行：K/V 的 batch/token 轴可零拷贝合并时用扁平索引，否则按原视图作两轴 gather；准备阶段不物化完整 cache。双方接收相同 tensor，记录 shape/stride/offset/storage；profiler 按完整任务序列区分 attention 与同名 API 前缀下的转换任务。[生产混合验收](../../tests/test_flash_attn_npu_v3_mla_mixed_stride_production.py) 另验完整 storage、metadata 跨视图复用、前缀/used 更新、流生命周期与真实 Trap。

CANN 执行器使用 `npu_fused_infer_attention_score_v2.out` 显式分配 Dv 维输出，避免 torch_npu 2.7.1 默认包装器把 combined 192/128 输出分配为 192。O/LSE 有限值沿用 `assert_fa_close`；无 softcap 时阈值固定为 `max(2 * max|pt-ref| + 2 * ULP(ref), 1e-5)`。inf 符号按 [3.3](#33-共同契约) 单独检查，不能为通过测试放宽阈值。

原生非 FD、FD 和空容量按固定源调度及实际 runtime tiling key 判定，不按耗时筛选。每次保存实际加载的 vendor/host/device 库、SHA256、输入描述、event span 与同步 E2E；event span 可能含 host 发射间隙，不当作纯 Kernel 时间。来源和 profiler 核验分别见 [provenance 测试](../../tests/test_flash_attn_npu_v3_mla_provenance.py)、[profile 测试](../../tests/test_flash_attn_npu_v3_mla_profile.py)。
