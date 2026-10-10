# MLA 当前状态

本次为 **2026-10-10 文档核验**，基于 `feat/mla-cann-migration` 的 `59c6f72`；功能进度保持 **M06d1：实际 Q/KV 使用长度 CPU 归一与调度**。唯一下一步为 **[M06d2a：used 记录协议与独立 AICPU 生成](plan.md#下一步m06d2a)**，尚未开始。当前文档提交用 `git log -1 -- docs/mla/status.md` 定位，上一功能闭环为 `59c6f72`。

目标契约与实现结构见 [design](design.md)，步骤及完成条件见 [plan](plan.md)。下表只描述当前可调用能力，不把独立 CPU 组件或未来目标计作公开支持。

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

## 独立 used 组件

M06d1 的 [`ParseUsedLengths` / `BuildUsedSchedule`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_used_schedule.h) 仅为 CPU 组件，实现位于 [`mla_tnd_schedule.cpp`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_schedule.cpp)。支持 BSND Q/KV、紧凑 TND Q/KV、TND Q + BSND KV；独立保存两轴的 64 位物理起点、分配长度与实际 used，缩短 used 不移动后续样本，缺省使用完整分配段。

分配、used、计数、字段、地址和资源分别完整校验；失败不改写输出，不保留输入指针。基本块/workspace 依据分配几何，任务/成本/执行块上界依据实际长度；显式 used 与缺省 TND 累计尾值保留不同的源数组语义，详见 [design 6.2.7](design.md#627-分配长度与-used-的-cpu-归一)。**尚无 used metadata 协议、AICPU 发射、设备消费者或 attention 输出**；共享源码被 v3/v4 AICPU 编译不等于 used 已接入设备。

## 最近闭环证据

### 本次文档核验

归档目录为 `build/mla-docs-review-20261010/`，静态检查结果与命令见 `doc-checks.json`，证据复核见 `evidence-check.json`，质量检查见 `quality-staged.log` / `quality-staged-exit.json`。

本次对照入口/校验、v1–v4 头文件、调度/used、构建、测试清单和固定官方源码快照，修正 v2/v4 分派、混合前向、记录克隆/scale 和返回值说明；合并共同协议规则，压缩历史验收过程，保持范围与步骤状态不变。修改前核对上一索引的 **6,074 个归档文件、238 个源码依赖、24 个对象及扩展哈希**，均匹配。修改后 237 个依赖不变，Python 包装文件仅 docstring 改变，去除 docstring 后 AST 一致；归档、对象和扩展哈希仍匹配。

静态检查通过：140 个本地链接/锚点、25 张表、60 个步骤的顺序与完成分组、4 个固定官方签名及性能矩阵数量一致。**未重建、未加载扩展、未运行 CPU/NPU 功能或性能测试**。下列功能结果沿用已核实归档，不作为本次重跑结果。

### 上一功能验收：M06d1

2026-10-10 的归档为 **`build/mla-m06d1-used/`**，总索引 `acceptance.json`；`*-command.json` / `*-exit.json` 保存命令、环境和退出码。最终 **2,813 项去重 pytest 通过**，分布如下：

| 验收面 | 归档结果 | 证据（相对归档目录） |
| --- | --- | --- |
| used CPU 归一/调度 | 306 项；U01/T01、可选 used、零容量/零 used、越界/溢出、64 位物理起点 | `used-verified.xml`、`used-verified/` |
| 独立固定源差分 | 6,187 组；原生非 FD 3,579，FA3 禁止 split 映射 2,608；另 288 组旧布局字段一致 | `used-differential-summary.json` |
| 既有调度/metadata | 861 项；dense/BSND/TND/混合、v1..v4、v3/v4 消费；另两项重复边界启用消费者 sanitizer | `cpu-regression-acceptance.json`、`cpu-verified.xml`、`consumer-sanitized.xml` |
| 普通 FA3/接口边界 | 839 项；普通 metadata、compile、官方签名、MLA 契约和拒绝边界 | `ordinary-regressions.xml`、`ordinary-nodes.json` |
| 910B 已有 TND/混合链路 | 792 项；双 dtype、前缀更新、stream/allocator/别名/compile，未开放的 used 仍拒绝 | `device-regressions.xml`、`device-nodes.json` |
| 构建/产物 | 15 项；传递依赖失效、实例/隔离/缓冲；链接对象与扩展核验 | `build-tests.xml`、`production-build-manifest.json` |

CPU 回归原轮次虽有 861 项断言通过，但两个驱动因 fork 子进程继承 stdin 管道而等待 EOF 超时，总退出码为 1；不能将该轮直接记为通过。归档持续进程工具修复子进程管道关闭后，完整补跑受影响 dense/环境检查 94 项和共用源 oracle 的 v1 metadata 22 项，四个驱动正常退出，无 sanitizer 报告；节点、请求数及输入/输出摘要核对一致，故汇总为 `passed_after_focused_exit_rechecks`。重复用例不增加总计；早期中断/失败轮次保留在归档，不计作通过。完整诊断见 `cpu-exit-diagnosis.json`。

M06d1 未修改 attention Kernel / Block / Tile，未重跑性能。最近公开混合 E2E 仍为 `build/mla-m06c3c-kvcache/performance-acceptance.json`：双 dtype 各 12 例，保留原生 FD/空 KV 分类，适用七例的复用、生成加 R=1/4、逐次内部生成均达标；不作为 used 能力证据。

## 环境与构建

以下为 **上一功能验收环境**。本次仅核对磁盘产物和固定版本，不宣称当前设备空闲或运行环境已重新验收；下次开发/实测前须重新检查。

| 项目 | 记录 |
| --- | --- |
| Python / PyTorch | `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`、torch_npu `2.7.1` |
| CANN / 编译器 | `/usr/local/Ascend/cann-9.0.0-beta.2`；bisheng clang 15.0.5 |
| 芯片与执行环境 | 八张 Ascend910B3；上一轮设备/构建/完整 sanitizer 在沙箱外执行，沙箱内驱动初始化失败 |
| CATLASS / 只读 CANN 源 | `769cd40a8716b28650b6bebb08db4834eea4462f` / `ca18aee610729ca5270a44e2d84c751bc687fbf0`；CATLASS 干净，CANN 原有 67 条删除/未跟踪记录保留 |
| 生产重建 | M06d1 仅改变 `mla_tnd_metadata.o`、`mla_tnd_bsnd_metadata.o`；其余 22 个对象不变，238 个源码依赖已归档 |
| 上次实际加载扩展 | `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `3fca88dd5acbda6317fe019721a9f2fc3a5bf1f7733360e6ca5581a80bce2f43` |

## 最小复现与交接

文档检查可运行 `python3 build/mla-docs-review-20261010/check_docs.py`。下列为上一功能闭环的最小 CPU/设备复查命令，使用新的输出目录；完整 sanitizer 矩阵按 `used-verified-command.json` 和 `cpu-verified-command.json` 使用归档修正后的 `pytest_persistent.py`，EOF 问题的补查命令见 `cpu-exit-recheck-command.json` / `metadata-exit-recheck-command.json`。

```bash
MLA_SCHEDULE_SANITIZE=1 ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_used_schedule.py -k 'u01_t01_physical_spans or source_array_semantics_and_capacity_geometry or per_sample_bounds_and_required_prefixes or wide_addresses_and_active_tile_bound' --basetemp=build/mla-used-cpu-recheck
```

源码/构建变化时重建扩展；已有设备链路的最小复查如下，先检查健康空闲卡并设置 `MLA_DEVICE`：

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 python3 setup.py build_ext --inplace
ASCEND_RT_VISIBLE_DEVICES=${MLA_DEVICE:?set MLA_DEVICE} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MLA_TND_BSND_METADATA_DEVICE_DIR=build/mla-used-legacy-producer-recheck MLA_TND_BSND_ATTENTION_DIR=build/mla-used-legacy-attention-recheck python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_mixed_public.py -k 'entries_empty_and_returns or none_integer_lengths or updates_streams_allocator_and_aliases or (source_accuracy and None and spec0)' --basetemp=build/mla-used-legacy-public-recheck
```

质量命令为 `PYTHONPATH=/tmp/mla-m01-precommit:$PYTHONPATH PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache ci/code_quality.sh --staged`，保留 CANN PYTHONPATH。只暂存本步文件，保留 `kernel_meta/` 未跟踪产物。下次先核验源码、构建和运行环境，按受影响范围重跑最小检查，再推进 M06d2a；生成、消费和独立 attention 仍分属 M06d2a/b/c。
