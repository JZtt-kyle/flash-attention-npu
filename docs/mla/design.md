# CANN FIA MLA → FA3 NPU：适配方案

状态：方案草案，待用户确认。本文描述目标契约，不表示这些能力已经实现。
当前进展见 [status.md](status.md)，候选迭代见 [plan.md](plan.md)。

## 1. 范围与固定依据

迁移集合是 **910B/C 上 CANN FIA 已有 MLA 前向能力，与官方 FA3 可表达能力的交集**，再加接入所必需的 metadata 支持。接口冲突按 FA3 调整；不搬迁普通 MHA/GQA、量化大框架、950 专用模板或反向算子。MLA prefill 中使用 GQA 的分组语义仍在范围内。

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

CANN 根目录：`/home/zhangqijia/repo/ops-transformer/attention/fused_infer_attention_score`。下列路径均相对于此目录：

| 来源 | 已核实的事实 |
| --- | --- |
| `op_host/arch22/fia_tiling_nonquant_mla.cpp`，`IsCapable` | 专用模板为 FP16/BF16，内容维 512、RoPE 64、KV 头数 1；含 PA、FlashDecode、mask 等分支 |
| `op_kernel/arch22/fia_kernel_nonquant_mla.h`、`fia_block_cube_nonquant_mla.h`、`fia_block_vec_nonquant_mla.h`、`fia_block_vec_flashdecode_arch22.h` | 解码流水、两个矩阵乘、softmax、输出更新及 split 合并 |
| `op_host/arch22/fia_tiling_nonquant.cpp`，`IsHighPerformanceTemplate`；`op_kernel/arch22/fia_kernel_nonquant.h` 及关联 Block | prefill 的内容维 128 + RoPE 64、V 维 128 在通用非量化路径中，不能按文件名只取带 `mla` 的文件 |
| `op_host/arch22/fused_infer_attention_score_tiling_check_feature.cpp` | 维度、布局、mask、页大小和特性交叉约束；`CheckFeatureHeadDim`、`CheckFeatureAxisInfo`、`CheckFeatureBlockSize` 等 |
| `tests/pytest/test_fia_mla_d512_pa_dim0_noncontiguous.py` | PA 首维 stride 回归；golden 分别使用独立 K 和 V，不能假定源代码总是忽略 V |

源码与文档可能不一致。例如 V5 文档对 decode 的 K/V 复用描述不能替代上述 kernel 与测试；TensorList 的文档描述也必须经过 arch22 host 校验核实。实现时将实际可运行组合固定为用例，未证实组合标为待核实，不能静默删减。

## 2. 现仓缺口

- [Python FA3 入口](../../flash_attn_npu_3/flash_attn_npu_interface.py) 已透传 `qv`；[910 C++ 前向](../../csrc/ascend910/flash_attn_npu_3/flash_api.cpp) 的 `q_v_` 目前只出现在参数声明，没有参与计算。这是数值正确性缺口。
- C++ 用 `empty_like(q)` 分配输出，并把 `embeddingSizeV` 设置为 Q 的维度；Python fake 却已经按 V 维度分配。MLA 的 64/512 和 192/128 都需要真实路径与 fake 一起核对。
- dense/varlen 包装器自动生成 metadata，却未传入 `headdim_v=v.shape[-1]`。仅修改 kernel 无法修复整个调用链。
- [metadata 参数](../../csrc/ascend910/flash_attn_npu_3/fa_metadata_args.h) 和 [AICPU 实现](../../csrc/ascend910/flash_attn_npu_3/fa_metadata.aicpu) 已存在，可以复用启动和流同步方式，不能直接把等维 FA 的 tiling 当作 MLA tiling。
- 现仓 metadata 为固定 2384 字节的 `uint8` tensor，并附加 Python 属性校验；预计算含 scale/softcap。官方为设备 `int32` 调度 tensor，公开参数是 `has_softcap`，没有 metadata 的 `softmax_scale` 参数。
- 现仓的窗口归一化、split 范围也与固定官方版本有差异。MLA 路径应独立核对官方分支，不能复用带有“对齐 GPU”注释的代码就视为已经对齐。

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
| 空行/空 KV | O 为 0；LSE 按固定上游实际路径验证，不能简单套用数学 `log(0)`。官方空 KV 分支使用 `+inf`，split 中间 LSE 还存在 `-inf` 哨兵 |
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

这张表是需求边界，不是运行验证清单。第一步用例盘点要细化可运行的交叉组合、来源与排除原因，尤其不能因通用模板文件很大而遗漏 prefill。

## 5. metadata 设计

metadata 指 FA3 的 `get_scheduler_metadata` → `scheduler_metadata` 调用链，不另造调用协议。

1. **公开签名以官方为准。** 使用 `headdim`、`headdim_v`、`has_softcap` 等官方参数及顺序。现仓额外 `softcap` / `softmax_scale` 如需兼容，只作为旧调用的关键字兼容层，不占官方位置参数、不成为 MLA 必填项；新调用和测试统一用官方形式。
2. **分离调度与计算数值。** metadata 保存切块、分核、split、长度等调度结果；前向本次调用提供 scale、softcap 数值、张量地址及 stride。metadata 接口没有 qv 参数，不应依据维度臆测 qv 存在，也不应把缺省 `1/sqrt(Dq)` 写死。只有 `has_softcap` 这类影响调度的特征进入调度契约。
3. **设备侧准备。** 复用现仓 AICPU 启动与事件同步，抽取不依赖 gert/optiling 的 MLA 调度函数。host 只用 shape、dtype、stride 等静态信息；长度读取与解析在设备侧完成，steady-state 不调用 `.cpu()`、`.item()` 或主机同步。
4. **不复制 GPU 内部二进制布局。** 新 MLA metadata 使用 NPU 上连续 int32 tensor，内容由 NPU 后端管理，带版本/类型及边界信息；兼容的是调用语义，不是 CUDA 与 NPU buffer 互换。普通 FA 现有 byte metadata 的过渡需保持既有调用可用，不能误送入 MLA kernel。
5. **有/无 metadata 共用调度。** 不传时内部生成；显式生成时可以复用相同调度条件。两条路径共享计算函数，不能分别维护两套容易漂移的 tiling 算法。
6. **生命周期明确。** 同设备、同调度参数与长度下可跨层复用；实际长度改变要重新计算。图内捕获 metadata 计算时，每次 replay 重新读取设备长度；图外生成的旧快照不会自动跟随长度张量变化。读写 workspace、计数器和 stream 的所有权要明确，禁止共享可变 scratch 造成并发污染。
7. **校验与图模式一致。** 校验尺寸、dtype、设备和可获得的静态参数；必要的内容检查留在设备侧。Python tensor 私有属性不能成为唯一正确性依据；不能为了校验在 graph 中读取设备标量。真实输出与 fake 的 metadata 大小、O/LSE/accumulator shape 必须一致。

## 6. 代码结构

实现接入点保持 [flash_api.cpp](../../csrc/ascend910/flash_attn_npu_3/flash_api.cpp) 和现有 Python 包。新增代码拟放在 `csrc/ascend910/flash_attn_npu_3/mla/`：

```text
Python FA3 入口 / custom op / fake
        ↓
C++ 校验、形状与 stride 解析、MHA/MLA 分派
        ↓
MlaLaunchParams（本次地址、scale 等） + MlaSchedule（可复用调度）
        ↓
CATLASS Kernel：tile 循环、缓冲、Cube/Vector 同步
        ↓
Block：QK + RoPE / online softmax / PV / rescale / split combine
        ↓
CATLASS Tile copy、Mmad、布局与硬件资源
```

- CANN 提供算法、调度和流水优化来源；CATLASS `examples/19_mla` 及 `include/catlass/*` 提供可复用组件和风格。示例不能替代完整 CANN 功能盘点。
- 迁移中保留适用的双缓冲、预取和压缩 KV 复用，逐步用组件表达；不把整个 FIA 大签名、图引擎注册、量化分支和 tiling key 树搬进本仓。
- 910B/C 共用适用的算法，但 Cube/Vector 比例、可用核数和存储资源从平台信息获取；保留源 MLA 的 C1V1/C1V2 适配分支，不能用 910B 实测参数硬编码替代 910C 兼容性。
- 复用已有 Tile；MLA 特有 Block 与 Kernel 先放本仓，避免每步改子模块。编译分派沿用轻量 host 头和独立实例化单元，更新生成器而非仅手改 autogen。
- metadata 的新 AICPU 源如独立成文件，必须同时修改 `setup.py`：当前构建只寻找固定的 `fa_metadata.aicpu`。header 改动的增量依赖也必须核查，防止链接旧对象。
- 测试使用独立数学 reference 和 CANN 基线适配器。生产入口不能直接调用 `torch_npu.npu_fused_infer_attention_score*` 充当移植实现。

## 7. 验收口径

**功能和精度：** 对同一逻辑输入同时检查独立高精度 reference、低精度 PyTorch 对照和原 CANN。CANN 对照先统一 scale、mask、布局与共享 KV 语义；本次按 FA3 调整的差异以官方为准，不要求重现源不兼容行为。reference 不能调用待测 kernel，也不能在两个实现中共用同一个错误的长度/页表解析。复用仓库双 reference 精度准则，单独检查 LSE、inf 符号、NaN、空行和 padding；阈值在基线步骤固定，不为通过用例而放宽。

**性能：** 在同一健康、空闲的 910B、同一输入/精度/软件栈上比较。固定 seed、预热、计时次数和同步方式，交替重复测量 CANN/新实现；保存每例耗时及运行信息。验收目标是每个约定代表用例 `新实现耗时 / 原 CANN 耗时 <= 1.0`，不默认引入 5% 或其他放宽阈值，也不能用平均加速掩盖退化用例。噪声未能分辨时补测，不能宣布不劣化。

分别报告 kernel 时间、普通调用端到端时间、metadata 生成时间与复用后的前向时间；图模式单列。接口适配导致的必要转换计入相应端到端成本，不拿 CANN 较慢的回退模板充当基线。对于原 CANN 的原生 NZ 等不同布局优势，可另列原生最佳路径结果并解释表示差异，不能隐藏转换成本后混为同一口径。

**覆盖：** 两种表示 × FP16/BF16 × dense/TND/PA × mask × 长短 KV × 单/多 query × 分组/尾块/stride；加 metadata 有无、重复使用、不同 scale、长度变更、split 自动/显式、graph replay。可采用代表性组合加针对性边界测试，不要求机械的全笛卡尔积，但每项源能力都要能追溯到用例或明确排除理由。

**版本可追溯：** CANN 源码 commit 不代表当前已安装算子版本。实测前记录实际生效的 OPP/算子库路径、构建版本或文件哈希，确认其对应本次来源；记录本仓扩展的实际加载路径与构建版本。910C 需要实现与资源配置兼容，按需求不作 910C 实机验收承诺。
