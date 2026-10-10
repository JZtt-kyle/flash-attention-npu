# MLA 当前状态

最近功能闭环为 **M06d3b2：内部混合 Q-used 接入**（2026-10-10，分支 `feat/mla-cann-migration`）。唯一下一步为 **[M06e1：BSND 合法外层 stride](plan.md#下一步m06e1)**，尚未开始。

本页是当前能力与交接入口；契约、协议和架构见 [design](design.md)，迭代顺序与完成条件见 [plan](plan.md)。本次基于 `0279155`，当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。

## 当前可调用能力

共同边界为 **910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、连续布局、无梯度**。Q/K 的末维为 64，qv/V 为 512；允许连续视图的非零 storage offset。BSND 每样本 KV 容量与 TND 的 `max_seqlen_k` 均为 0..16384；TND 总 Q/KV 各不超过 `INT32_MAX`，总 KV 可以超过 16384。其余字段、地址和资源限制见 [design 6.2](design.md#62-调度字段与边界)。

### 公开入口

长度 tensor 均须同设备、连续、int32；B 为 batch。前缀为 `[B+1]`，实际长度为 `[B]`。

| 入口 | 输入与长度 | metadata 路径 |
| --- | --- | --- |
| `flash_attn_func` | BSND Q/KV，使用整容量 | 每次内部 v1 |
| `flash_attn_varlen_func` | TND Q/KV，两组分配前缀；max 参数是每段分配上界；可选 `seqused_q/k` | 任一 used 存在时每次内部 v5，否则 v3；无显式复用参数 |
| `flash_attn_with_kvcache`，BSND Q | Q 整容量，只读 BSND cache；`cache_seqlens` 为 None（整容量）、`0..Sk` 整数或长度 tensor | 无长度且无记录走 v1，否则内部 v2 或显式 v2 |
| `flash_attn_with_kvcache`，TND Q | Q 使用完整分配段，须给 Q 前缀和 `max_seqlen_q`；只读 BSND cache，KV 长度同上 | 内部 v4 或显式 v4 |
| `get_scheduler_metadata` | 必须给 KV 长度 tensor，不接受 None/整数；MLA 维度为 `headdim=64, headdim_v=512` | 无 Q 前缀生成 v2，有 Q 前缀生成混合 v4 |

协议精确字数集中见 [design 第 5 节](design.md#5-metadata-设计)。公开显式生成/复用只开放 v2/v4，v3/v5 在前向内部逐次生成。

### 内部入口与独立组件

| 层次 | 已接通边界 |
| --- | --- |
| `_flash_attn_forward` / C++ `fwd`，BSND Q/KV | `seqused_q` 存在时使用 v5，可选 `seqused_k`；无 Q-used 沿用 v1/v2；Q-used 与显式 metadata 组合拒绝 |
| 同一内部入口，TND Q/KV | 前缀、max、可选 used 及 v3/v5 选择与公开 varlen 相同；拒绝显式 metadata |
| 同一内部入口，TND Q + BSND KV | 无 Q-used 沿用 v4；有 `seqused_q` 时每次内部 v5，可选 `seqused_k`；Q-used 与显式 metadata、`max_seqlen_k` 仍拒绝 |
| v5 独立生成、消费者与 attention probe | BSND、TND、混合三布局均已验收；生产接入见上述入口 |

入口依据为 [Python 包装/fake](../../flash_attn_npu_3/flash_attn_npu_interface.py)、[C++ 通用分派](../../csrc/ascend910/flash_attn_npu_3/flash_api.cpp)、[专用前向](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp) 和 [静态校验](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h)。生产 BSND/TND/混合 used 共用 `UsedForward`、v5 生成/校验与 `MlaUsed`，数值 Block / Tile 沿用原路径。

### 调用约束与输出

- **选项：** 只支持 `num_splits=1`。func/varlen/内部前向默认 1；kvcache/metadata 默认 0，须显式传 1。仅无 mask、`softcap=0`、`pack_gqa=None`、`sm_margin=0`、`attention_chunk=0`；metadata 的 `has_softcap=False`。scale 默认 `1/sqrt(576)`，显式值须在 FP32 有限范围，记录不固化 scale。
- **输出：** O 与输入同 dtype，LSE 为 FP32。BSND O/LSE 为 `[B,Sq,Hq,512]` / `[B,Hq,Sq]`，TND 为 `[Tq,Hq,512]` / `[Hq,Tq]`。请求 LSE 时 func/varlen 返回二项，kvcache 返回四项；后两项是同设备空 accumulator，dtype 随 PyTorch 默认浮点 dtype。
- **长度与记录：** 长度值在设备检查，不做 D2H。显式 v2/v4 须匹配当次 KV 长度，v4 还匹配 Q 前缀。损坏/过期记录由消费者 Trap 拒绝，空输出仍校验。v5 完整检查两轴分配及 `0<=used[b]<=本轴分配段长`，即使另一轴 used 为零；缺省 used 使用整个分配段，缩短 used 不移动后续样本起点。
- **空值与有效区：** 空 batch/query 保留空 shape；非零容量中的零 KV 样本有效行 `O=0、LSE=-inf`；全局空 KV 初始化整个输出分配区为 `O=0、LSE=+inf`。其他未使用 query 槽位不承诺输出值。

**待迁移能力仍明确拒绝：** 非连续外层 stride、PA、causal/window、自动或多 split、192/128 prefill、非零 softcap 与非默认调度选项。KV 追加、内置 rotary 等范围外组合也拒绝，完整边界见 [design 第 4 节](design.md#4-功能边界)。fake/compile 已有覆盖，完整 graph replay 留在 M16；目标包含 910C，生产平台检查仍只开放 910B。

## 本步实现与验收证据

M06d3b2 归档为 **`build/mla-m06d3b2-mixed-used/`**，以下路径均相对该目录。`*-command.json` / `*-exit.json` 保存完整命令、环境和退出码；`acceptance.json` 为总索引，`source-snapshots/` / `production-final/` 保存源码和构建产物，全部验收证据记录哈希。

- 内部 `_flash_attn_forward` / C++ `fwd` 的混合 Q-used 进入 `TndBsndUsedForward` → `UsedForward`，复用已有 v5 生成器、消费者与 TND used launcher。Q 起点来自分配前缀，KV 起点始终为 64 位 `b*Sk`，缩短 used 不压紧任一轴。
- 未增加公开参数或私有 kvcache op 参数；公开混合 kvcache 仍使用 v4。Q-used 与任何显式 metadata 组合拒绝，混合 `max_seqlen_k` 仍拒绝。
- Kernel / Block / Tile、metadata 协议、CATLASS 与 CANN 源未改。29 个生产链接对象中仅 `flash_api.o`、`mla_forward.o` 改变；实际加载扩展与链接产物、251 个源码依赖和全部对象已核对。

| 验收面 | 结果 | 证据 |
| --- | --- | --- |
| 上一步最小复查 | 184 项通过；双 dtype 固定源/迁移 probe 与 v5 producer 从源码重建 | `baseline.xml`、`baseline-attention/`、`baseline-producer/` |
| 内部混合 Q-used | 359 项通过；双 dtype 有效行/poison padding、源/数学精度、空容量/零 used、前缀/used 更新、offset/alias/clone、跨流与 allocator 生命周期、真实/fake/C++/compile、静态边界；含双 dtype 各 15 项真实 Trap | `production-tests.xml`、`production/`、`production-source/` |
| 普通 FA3 / 接口 | 833 项通过；普通 metadata、compile、签名及 MLA 契约 | `ordinary-regressions.xml`、`ordinary-nodes.json` |
| 既有 MLA 入口 | 1624 项通过；完整 BSND/TND used，公开混合 v4、varlen 和 kvcache 的相关入口/空值/更新/生命周期回归 | `legacy-public.xml`、`legacy-nodes.json` |
| 构建 | 19 项通过；实例生成、header/AICPU/设备依赖及 sanitizer | `build-tests.xml`、`production-build-manifest.json` |

**合计 2,835 项去重 pytest 通过，无失败或跳过项**；184 项 baseline 已包含于后续回归，不重复累加。

主精度矩阵每 dtype 25 组，分别通过 Python 内部入口和原始 C++ 调用，重复三次；非空有效行 O/LSE 与固定源逐元素一致，零 KV 的 LSE 符号按 FA3 单独检查。独立数学对照的最大 O 绝对误差为 FP16 0.0009765625、BF16 0.015625，最大 LSE 绝对误差均为 9.537e-7；没有放宽既定精度阈值。记录见 `baseline-attention/{fp16,bf16}/production-mixed-used-accuracy.jsonl`。

### 性能口径与限制

本步无公开 FA3 等价入口，性能仅作**内部调用与同调度固定源 Kernel 对照**，不作公开 E2E 不劣化结论。双 dtype 各 9 个预先固定用例，包含混合 T01、非均匀/零长度、M/S2 尾块、16384 KV、43 batch、缺省 KV-used、显式整段 used 与多块 query。双方同输入、基本块、核数、端点；生产每次调用均生成 v5，源 probe 预分配缓冲区。

在 7 号 910B 同进程交替测量，CPU 亲和性 128..135，预热 5 次，五轮各 20 个 wall 样本和各 10 个 profiler 样本。同期旧入口回归位于 6 号卡。下表为每例中位数的范围，逐例值与全部原始样本均保存。

| dtype | 生产/固定源 Kernel 比值 | 逐次 v5 生成（μs） | 完整内部调用 wall（μs） |
| --- | --- | --- | --- |
| FP16 | 0.992–2.549 | 24.871–37.991 | 430.925–1299.860 |
| BF16 | 1.008–2.679 | 23.281–38.241 | 431.485–1318.590 |

**组件仍有开销，不声称纯 Kernel 不劣化。** 逐例生产/源 Kernel、生成及双方内部 wall 见 `performance-{fp16,bf16}/summary.json`；其 `wall.json`、`kernel-samples.json` 和 `profiler/` 为原始证据。公开 varlen used 的最近 E2E 结论仍来自 M06d3b1，见 `build/mla-m06d3b1-varlen-used/performance-acceptance.json`，不能将本步内部计时替代该口径。

profiler 两 dtype 均出现 `fwk_api_db` 解析失败，**未记为完整 framework trace 通过**。设备计时按调用顺序核对原始 CANN `op_summary` 与 `kernel_details.csv`，每 dtype 1,350 条任务逐项一致，其中 450 条为 v5 生成；诊断见 `profiler-diagnostics.json` 与 `device-trace-audit.json`。此前 4/5 号卡固定源 probe 的 AICore 超时根因仍未定，未认定硬件故障，本次未使用这两张卡；历史证据在 M06d3b1 的 `timeout-analysis.json` 和 `timeout-evidence/`。

## 本次实测环境

| 项目 | 本步记录 |
| --- | --- |
| Python / PyTorch | `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`、torch_npu `2.7.1` |
| CANN / 编译器 | `/usr/local/Ascend/cann-9.0.0-beta.2`；bisheng clang 15.0.5 |
| 芯片 | 八张 Ascend910B3；预检所用 6/7 号卡健康空闲，新入口/性能用 7，普通 FA3 与既有入口回归用 6 |
| CATLASS / 只读 CANN 源 | `769cd40a8716b28650b6bebb08db4834eea4462f` / `ca18aee610729ca5270a44e2d84c751bc687fbf0`；CATLASS 干净，CANN 原有工作区差异保留 |
| 实际加载扩展 | `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `277e55d03ca46c77a2b1691ebc8b6b2de11b2c3bc8298ef13a214977d13acd73` |

预检核对上步 2,659 个归档文件、251 个源码依赖、29 个对象与原扩展哈希，均一致，见 `precheck.json`。实际构建命令见 `production-build-command.json`；重建后的加载路径和对象变化见 `production-build-manifest.json`。功能与性能检查期间生产源码及扩展哈希未变。原有未跟踪 `kernel_meta/` 保留。

## 最小复现与交接

先检查健康空闲卡，设置 `MLA_DEVICE` 并加载上述 CANN 的 `set_env.sh`。共享头或生产源码改变时先执行 `FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 python3 setup.py build_ext --inplace` 并核对实际加载扩展。以下从源码重建独立 probe，覆盖混合 T01、空值、前缀/used 更新、生命周期及空 KV 坏 Q-used 的设备拒绝：

```bash
ASCEND_RT_VISIBLE_DEVICES=${MLA_DEVICE:?set MLA_DEVICE} \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MLA_USED_ATTENTION_REUSE=0 \
MLA_USED_ATTENTION_DIR=build/mla-m06d3b2-mixed-used/recheck-attention \
MLA_USED_METADATA_DEVICE_DIR=build/mla-m06d3b2-mixed-used/recheck-producer \
MLA_MIXED_USED_DIR=build/mla-m06d3b2-mixed-used/recheck-production \
python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_mixed_used.py \
  -k '(production_accuracy and spec0) or shapes_empty_and_defaults or fullgraph_prefix or alias_clone or updates_preserve or cross_stream_allocator_lifetime or (runtime_trap and empty_k_bad_q)' \
  --basetemp=build/mla-m06d3b2-mixed-used/recheck-tmp
```

完整命令见 `production-tests-command.json`、`ordinary-regressions-command.json`、`legacy-public-command.json`、`build-tests-command.json` 和 `performance-{fp16,bf16}-command.json`。本步未运行 910C 实机或完整 graph replay，后者仍在 M16。

文档链接/锚点、表格、固定官方签名、协议常量与步骤/性能矩阵一致性检查通过，见 `docs-check.json`。`git diff --check` 与适用质量钩子通过，见 `quality-staged.log`；首轮格式化重写另保留在 `quality-initial.log`。复现命令如下，质量工具沿用 `/tmp` 安装与缓存，保留 CANN 的 PYTHONPATH：

```bash
python3 build/mla-m06d3b2-mixed-used/check_docs.py
PYTHONPATH=/tmp/mla-m01-precommit${PYTHONPATH:+:$PYTHONPATH} \
PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache ci/code_quality.sh --staged
python3 build/mla-m06d3b2-mixed-used/acceptance.py
git diff --check
```

后续只推进 plan 中的 M06e1；本步未开始外层 stride 开发。
