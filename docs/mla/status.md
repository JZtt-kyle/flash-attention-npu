# MLA 当前状态

当前闭环为 **M06d2c：used 的独立 Kernel 链路**，2026-10-10 基于 `feat/mla-cann-migration` 的 `c3fb7e6` 完成。唯一下一步为 **[M06d3：used 的公开与内部入口接入](plan.md#下一步m06d3)**，尚未开始。本步提交用 `git log -1 -- docs/mla/status.md` 定位。

目标契约与实现结构见 [design](design.md)，步骤及完成条件见 [plan](plan.md)。下表只描述当前公开可调用能力，独立 attention 不代表公开 used 已开放。

## 当前可调用能力

共同边界为 **910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、连续布局、无梯度**；允许非零 storage offset。Q 使用完整分配段。BSND 每样本 KV 容量与紧凑 TND 的 `max_seqlen_k` 均为 0..16384；紧凑 TND 总 Q/KV 各不超过 `INT32_MAX`，总 KV 可以超过 16384。混合 BSND cache 总容量为 64 位 `B*Sk`。完整字段、地址和资源界限见 [design 6.2](design.md#62-调度字段与边界)。

| 入口 | 输入与长度 | 调度记录 |
| --- | --- | --- |
| `flash_attn_func` | BSND 整容量 | 内部 v1：`int32[401]` |
| `flash_attn_varlen_func` | 紧凑 TND Q/KV；两组同设备连续 `int32[B+1]` 前缀，max 参数为每段上界 | 每次内部 v3：`int32[405+4B]`；无公开显式复用参数 |
| `flash_attn_with_kvcache`，BSND Q | 只读 BSND cache；KV 长度为 None、`0..Sk` 整数或同设备连续 `int32[B]` | 无长度且无记录走 v1；否则内部 v2 或显式 v2：`int32[401+B]` |
| `flash_attn_with_kvcache`，TND Q | TND Q/qv + BSND K/V；必须给出 Q 前缀和 `max_seqlen_q` 上界；KV 长度同上 | 内部 v4 或显式 v4：`int32[406+5B]` |
| `get_scheduler_metadata` | 必须给设备连续 `int32[B]` KV 长度；不接受 None/整数 | 无 Q 前缀生成 v2，有 Q 前缀生成混合 v4 |

- **调用选项：** 仅 `num_splits=1`；func/varlen 默认 1，kvcache/metadata 默认 0，须显式传 1。无 mask，`softcap=0`、`pack_gqa=None`、`sm_margin=0`、`attention_chunk=0`；metadata 的 `has_softcap=False`。scale 默认 `1/sqrt(576)`，显式值须在 FP32 有限范围，记录不固化 scale。
- **输出：** O 与输入同 dtype，LSE 为 FP32。BSND O/LSE 为 `[B,Sq,Hq,512]` / `[B,Hq,Sq]`，TND 为 `[Tq,Hq,512]` / `[Hq,Tq]`。请求 LSE 时 func/varlen 返回二项，kvcache 返回四项；未 split 的两个 accumulator 为同设备空 tensor。
- **记录校验：** 前缀/长度值留在设备检查；显式 v2/v4 逐项匹配当次长度，v4 还匹配 Q 前缀。记录可克隆或使用连续偏移视图，损坏/过期记录由消费者 Trap 拒绝。空输出仍校验记录和长度。
- **空值：** 空 batch/query 保留空 shape；非零容量中的零 KV 样本 `O=0、LSE=-inf`，全局空 KV 为 `O=0、LSE=+inf`。

**仍拒绝：** varlen `seqused_q/k`、Q-used、非连续外层 stride、PA、causal/window、自动或多 split、192/128 prefill，以及 KV 追加、内置 rotary 等范围外组合。已有 BSND cache 的实际 KV 长度通过 `cache_seqlens` 支持，内部通用前向用 `seqused_k` 传递。内部通用混合入口仍拒绝 `max_seqlen_k`。fake/compile 已覆盖，完整 graph replay 留在 M16；目标包含 910C，生产平台检查仍只开放 910B。

以上入口与拒绝边界对应 [Python 包装/fake](../../flash_attn_npu_3/flash_attn_npu_interface.py)、[C++ 通用入口](../../csrc/ascend910/flash_attn_npu_3/flash_api.cpp)、[专用前向](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp) 及 [静态校验](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h)；完整生产分派表见 [design 6.5](design.md#65-生产接入与构建)。

## 独立 used attention

v5 为 **`int32[410+8B]`、version/kind=5/5**，保存完整调度、两轴布局/总容量/max 上界、显式 used 标记，以及每样本 64 位物理起点、分配长度和实际长度；协议见 [design 5.7](design.md#57-v5分配与-used-记录)。既有 AICPU 生成与 `ReadUsedMetadata` / `LoadUsedMetadata` 校验不变：每个 Cube/Vector 核检查当次四数组、跨度、端点和资源，失败时 Trap，只有校验通过后才允许写 attention 输出或分配事件。前缀/used 不做 D2H。

新 [`MlaUsed`](../../csrc/ascend910/flash_attn_npu_3/mla/kernel/mla_used.hpp) 接通 **BSND Q/KV、TND Q/KV、TND Q + BSND KV**。`MlaNonPaged` 的 v5 分支按 used 限制 M/S2 计算，TND 按分配起点寻址，BSND 保留 64 位 `b*capacity`；缩短 used 不移动后续样本。Block/Tile 数值与事件顺序复用现有实现。非空容量中零 KV 的有效行 `O=0、LSE=-inf`；全局空 KV 按固定 FA3 定义初始化整个输出分配区为 `O=0、LSE=+inf`。其他未使用 query 槽位不承诺输出值。

[`独立 attention probe`](../../tests/mla_used_attention_probe.cpp) 在创建 TPipe、同步及写输出前校验 v5，真实调度仅来自设备记录；测试中清空 host 调度不影响迁移输出。偏移视图、只读四数组别名、metadata clone、长度更新/重生成、跨 stream 和 allocator 生命周期已覆盖。

**当前 used attention 仅由独立 probe 调用，公开入口仍为 v1–v4，继续明确拒绝 used。** v5 AICPU 生成器已链接生产扩展，新 attention 实例尚未接入生产 launcher；本步未增加公共参数。

## 最近闭环证据

归档目录为 **`build/mla-m06d2c-used-attention/`**，总索引 `acceptance.json`；`*-command.json` / `*-exit.json` 保存命令、环境与退出码，`source-snapshots/` 和 `production-final/` 保存最终源码及构建产物。本步 **1,611 项去重 pytest 通过，无跳过项**：

| 验收面 | 本次结果 | 证据（相对归档目录） |
| --- | --- | --- |
| 上一步最小复查 | v5 CPU 5 项、910B 8 项；sanitizer、U01/T01、四数组更新、跨流及 stale-used Trap | `baseline-cpu.xml`、`baseline-device.xml`、`baseline-cpu-driver-exits.json` |
| used attention | 主套件 167 项；三布局、双 dtype、U01/T01、缺省/显式 used、M/KV 边界及 16384、零 Q/KV、poison、非零 offset、scale、长度更新、别名/clone/stream/allocator | `attention-final.xml`、`attention-smoke/{fp16,bf16}/accuracy.jsonl`、`producer-final/` |
| 全局空 KV 分配区 | 补充 12 项；双 dtype、三布局、部分/全零 Q-used，整个 O/LSE 分配区的固定 FA3 语义 | `attention-empty.xml`、`producer-empty/` |
| 真实 Trap（包含在主套件中） | 双 dtype 各 16 项；旧协议、布局/标记/跨度、四数组过期、错计数/对齐、端点/workspace、生成失败和空容量；诊断拒绝不写 O/LSE/workspace 哨兵 | `attention-smoke/*/trap-*/runtime-error.json` 及同目录命令/日志 |
| 固定源调度与输出 | 19 组调度：原生非 FD 14 / 禁止 split 映射 5；有效 O 和非空行 LSE 与同调度源 Kernel 逐元素一致，再验独立双数学 reference；零 KV 单独验 LSE 符号 | `attention-final-tmp/test_source_schedules0/attention*.json`、`source-final/`、`attention-smoke/` |
| v1–v4 metadata 回归 | 548 项；419 项 CPU 消费者与 129 项 AICPU 生成，含 sanitizer/源字段/空值/生命周期 | `legacy-metadata.xml`、`legacy-metadata-tmp/`、`legacy-metadata-driver-exits.json` |
| 普通 FA3 / 公开边界 | 839 项普通 metadata、compile、签名和 MLA 契约；另 13 项 varlen/混合 used 拒绝 | `ordinary-regressions.xml`、`ordinary-nodes.json`、`used-public-boundary.xml` |
| 构建 / 质量 | 19 项构建测试；双 dtype probe 的实际设备依赖、生产增量重建与加载核验；质量钩子和 diff 检查通过 | `build-tests.xml`、`attention-smoke/*/*-device.d`、`production-build-manifest.json`、`quality-staged.log` |

CPU 最小复查的两个持续驱动及旧 metadata 的十七个驱动均正常退出，ASan/UBSan/LSan 无报告。首轮 `attention-smoke` 的 46 项已通过；最终主套件复用核验过哈希的相同二进制。没有精度、Trap、构建或回归遗留失败。

固定源 TND 的累计长度同时决定起点与实际长度，测试侧从原始逻辑输入独立压紧有效段；BSND 轴保留物理容量，源 header 不改动。迁移侧保留分配间隙和 poison，数学 reference 按原始分配段长/used 独立切片。源组件共用已差分核验的基本块、核数和端点，仅比较官方定义区域；该源适配不代表原生 CANN 的公开 used 分派。

## 组件性能与限制

同卡同进程交替比较固定源非 FD 与迁移 Kernel，FP16/BF16 各 **14 组固定用例**；seed=20261010，预热 5 次，五轮各 20 个 wall 样本与 10 个 profiler 样本。每 dtype 保存 2,800 个 wall、1,400 个 Kernel 样本及逐例中位数。输入/输出/workspace 预分配，源 TND 压紧在计时前完成；probe 调用包含 ctypes、stream 登记和同步，迁移 Kernel 包含完整 v5 校验。

| dtype | 源 Kernel 中位数范围 | 迁移 Kernel 中位数范围 | 逐例迁移/源比值范围 | 源 / 迁移 probe wall 范围 |
| --- | --- | --- | --- | --- |
| FP16 | 24.60–948.31 μs | 30.31–953.08 μs | 1.004–2.915 | 225.14–1188.23 / 230.29–1189.95 μs |
| BF16 | 25.36–948.86 μs | 30.28–954.54 μs | 1.004–2.913 | 218.00–1164.91 / 220.54–1166.40 μs |

全部逐例结果见 `performance-{fp16,bf16}/summary.json`，原始样本为同目录 `wall.json` / `kernel-samples.json` 和 profiler 原始 CSV。最高比值均为 43 样本、零计算成本的混合布局用例；源 TND 使用压紧的 Q，迁移侧保留完整分配及 v5 校验。**本步有组件开销，不声称不劣化；这些数据不作为公开 E2E。** 公开 used 的适用 E2E 留在 M06d3；最近公开混合 E2E 仍为 `build/mla-m06c3c-kvcache/performance-acceptance.json`。

两 dtype 的 torch_npu profiler **framework flow / step / framework API DB 关联导出失败**，错误保存在 `profiler-diagnostics.json` 和 `performance-*/profiler/*/logs/`，未记为完整 framework trace 通过。组件计时使用独立调用顺序与原始 CANN `op_summary`：每 dtype 的 1,400 条 `kernel_details.csv` 在数量、名称、设备、时间戳和时长上逐项匹配，未使用失败的 framework 关联结果。

## 环境与构建

接手时核验上一闭环 **1,841 个归档文件、242 个源码依赖、25 个对象和扩展哈希**。本步 Python/NPU、sanitizer、编译和实机验收按既有方式在沙箱外执行；证据见 `precheck.json`。CATLASS 和只读 CANN 源工作区前后一致。

| 项目 | 本次实测记录 |
| --- | --- |
| Python / PyTorch | `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`、torch_npu `2.7.1` |
| CANN / 编译器 | `/usr/local/Ascend/cann-9.0.0-beta.2`；bisheng clang 15.0.5 |
| 芯片 | 八张 Ascend910B3；used/性能用 7，全局空 KV 补查和普通 FA3 用 4，旧 metadata 用 5 |
| CATLASS / 只读 CANN 源 | `769cd40a8716b28650b6bebb08db4834eea4462f` / `ca18aee610729ca5270a44e2d84c751bc687fbf0`；CATLASS 干净，CANN 原有 67 条删除/未跟踪记录保留 |
| 生产重建 | 共享 Kernel 增加 used 模板分支；六个既有 MLA 实例按设备依赖重编译，25 个对象内容哈希均不变；242 个源码依赖归档 |
| 实际加载扩展 | `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `21ab42b621b3c2597208e4e180cc378f27308f4ed3682b919eaddcacc067d902`；与本次链接产物一致 |

## 最小复现与交接

完整命令见 `attention-final-command.json`、`attention-empty-command.json`、`legacy-metadata-command.json`、`ordinary-regressions-command.json` 和 `performance-{fp16,bf16}-command.json`。以下复查会从源码重建独立 probe，覆盖有效行/更新/生命周期、全局空 KV 和 stale-used Trap；先检查健康空闲卡并设置 `MLA_DEVICE`，Trap 保持独立子进程：

```bash
ASCEND_RT_VISIBLE_DEVICES=${MLA_DEVICE:?set MLA_DEVICE} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MLA_USED_ATTENTION_DIR=build/mla-m06d2c-used-attention/recheck-attention MLA_USED_METADATA_DEVICE_DIR=build/mla-m06d2c-used-attention/recheck-producer python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_used_attention.py -k 'each_length_update or global_empty_fills_allocation or alias_clone or cross_stream_allocator or (runtime_trap and stale_uq)' --basetemp=build/mla-m06d2c-used-attention/recheck-tmp
```

Kernel/Block/Tile 或输出变化时运行完整 attention 套件及双 dtype 组件性能；共享头/生产构建变化时先运行 `FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 python3 setup.py build_ext --inplace`，核对加载路径和依赖，再补受影响回归。质量命令为 `PYTHONPATH=/tmp/mla-m01-precommit:$PYTHONPATH PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache ci/code_quality.sh --staged`，保留 CANN PYTHONPATH。只暂存本步文件，保留 `kernel_meta/` 未跟踪产物。

唯一下一步为 M06d3；先核验源码、构建与环境并复查本步证据。公开 used、910C 实机和完整 graph replay 尚未验收。
