# MLA 当前状态

最近闭环为 **M06e1a：BSND stride 描述与地址校验**（2026-10-10，分支 `feat/mla-cann-migration`）。本步是独立 CPU 组件；现有生产能力仍为 M06d3b2，非连续输入尚未开放。唯一下一步为 **[M06e1b：BSND 独立 stride attention](plan.md#下一步m06e1b)**。

本页是当前能力与交接入口；契约、协议和架构见 [design](design.md)，迭代顺序与完成条件见 [plan](plan.md)。本次基于 `3b8be96`，当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。

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
| BSND stride 地址组件 | 独立 CPU 校验四路输入的 64 位 stride、storage offset 和完整分配视图边界；尚未接入设备或前向 |
| v5 独立生成、消费者与 attention probe | BSND、TND、混合三布局均已验收；生产接入见上述入口 |

生产入口依据为 [Python 包装/fake](../../flash_attn_npu_3/flash_attn_npu_interface.py)、[C++ 通用分派](../../csrc/ascend910/flash_attn_npu_3/flash_api.cpp)、[专用前向](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp) 和 [静态校验](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h)。生产 BSND/TND/混合 used 共用 `UsedForward`、v5 生成/校验与 `MlaUsed`，数值 Block / Tile 沿用原路径。

### 调用约束与输出

- **选项：** 只支持 `num_splits=1`。func/varlen/内部前向默认 1；kvcache/metadata 默认 0，须显式传 1。仅无 mask、`softcap=0`、`pack_gqa=None`、`sm_margin=0`、`attention_chunk=0`；metadata 的 `has_softcap=False`。scale 默认 `1/sqrt(576)`，显式值须在 FP32 有限范围，记录不固化 scale。
- **输出：** O 与输入同 dtype，LSE 为 FP32。BSND O/LSE 为 `[B,Sq,Hq,512]` / `[B,Hq,Sq]`，TND 为 `[Tq,Hq,512]` / `[Hq,Tq]`。请求 LSE 时 func/varlen 返回二项，kvcache 返回四项；后两项是同设备空 accumulator，dtype 随 PyTorch 默认浮点 dtype。
- **长度与记录：** 长度值在设备检查，不做 D2H。显式 v2/v4 须匹配当次 KV 长度，v4 还匹配 Q 前缀。损坏/过期记录由消费者 Trap 拒绝，空输出仍校验。v5 完整检查两轴分配及 `0<=used[b]<=本轴分配段长`，即使另一轴 used 为零；缺省 used 使用整个分配段，缩短 used 不移动后续样本起点。
- **空值与有效区：** 空 batch/query 保留空 shape；非零容量中的零 KV 样本有效行 `O=0、LSE=-inf`；全局空 KV 初始化整个输出分配区为 `O=0、LSE=+inf`。其他未使用 query 槽位不承诺输出值。

**待迁移能力仍明确拒绝：** 非连续外层 stride、PA、causal/window、自动或多 split、192/128 prefill、非零 softcap 与非默认调度选项。KV 追加、内置 rotary 等范围外组合也拒绝，完整边界见 [design 第 4 节](design.md#4-功能边界)。fake/compile 已有覆盖，完整 graph replay 留在 M16；目标包含 910C，生产平台检查仍只开放 910B。

## 本步实现与验收证据

M06e1a 归档为 **`build/mla-m06e1a-bsnd-layout/`**，以下路径均相对该目录。`*-command.json` / `*-exit.json` 保存完整命令、环境和退出码；`acceptance.json` 为总索引，`source-snapshots/` 保存本步源码。原始输入、驱动输出、编译命令、二进制及源码哈希均归档。

- 新增 [`mla_bsnd_layout.h/.cpp`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_bsnd_layout.h)，以整个分配视图检查 storage 和有符号 64 位字节界限，乘加前检查溢出，失败保留原输出。Q/qv/K/V 分别构造描述，实际 `data_ptr()` 已含 offset，设备侧不得重复相加。
- 地址组件接受末维连续、非负外层 stride，包括 BNSD、切片、只读广播/重叠；空视图不访问 storage，单例轴不窄化未使用 stride。详见 [design 6.2.8](design.md#628-bsnd-输入-stride-与地址校验)。这些是组件能力，不代表前向入口开放。
- M06e1 拆为 CPU 地址校验、独立设备执行、生产接入三个提交。本步未修改 Kernel / Block / Tile、公共/fake/C++ 入口、metadata、setup、CATLASS 或 CANN 源；新组件由测试驱动直接编译，尚未链接生产扩展。

| 验收面 | 结果 | 证据 |
| --- | --- | --- |
| 上一步证据与生产构建 | 2,307 个归档文件、251 个源码依赖、29 个对象及加载/链接扩展哈希一致；上步 2,835 项结果与最小复查命令已复核 | `baseline-audit.json`、`verify_baseline.py`、`baseline-final.log` |
| CPU 地址组件 | 82 项通过；双 dtype 真实视图地址、四路独立 offset、U01 poison、广播/重叠、空/单例轴、65536/uint32/int64 边界、失败不修改输出；含 1,200 组大整数随机对照 | `cpu-final.xml`、`cpu-final/` |
| ASan / UBSan / LeakSanitizer | 82 项通过；沙箱外从源码重建，保留泄漏检测，无 sanitizer 报告 | `sanitizer-final.xml`、`sanitizer-final/` |
| 既有入口边界 | 120 项通过；910B 双 dtype 真实/fake/原始 C++ 的支持边界及 BSND Q-used stride 拒绝 | `public-boundary.xml`、`public-boundary-command.json` |

**合计 202 项去重 pytest 通过**；sanitizer 的 82 项为同一 CPU 矩阵重验，不重复累加。

CPU 初测 82 项通过；增加大 offset 边界并格式化后重建，最终仍为 82 项通过。沙箱中的首次 sanitizer 在首例退出，错误为 `LeakSanitizer does not work under ptrace`，保留 `sanitizer.log` 和 `sanitizer/`，未计为通过。沙箱外复测结果见上表。

本步未改公共入口或 metadata，未重跑普通 FA3 全套；未改设备数值路径，未运行新的 stride NPU 精度或性能，也未重建生产扩展。源 ND→NZ stride 字段与 64 位输入 stride 间的逐行/逐矩阵回退仍需 M06e1b 设备验收。既有完整功能与性能证据仍在 `build/mla-m06d3b2-mixed-used/acceptance.json`；公开 varlen used 的 E2E 证据仍在 `build/mla-m06d3b1-varlen-used/performance-acceptance.json`，本步没有新增性能结论。

## 本次实测环境

| 项目 | 本步记录 |
| --- | --- |
| Python / PyTorch | `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`、torch_npu `2.7.1` |
| CANN / 编译器 | `/usr/local/Ascend/cann-9.0.0-beta.2`；bisheng clang 15.0.5；CPU 驱动使用 g++，版本与完整编译参数见 `environment.json`、各驱动目录的 `compile-command.json` |
| 芯片 | 八张 Ascend910B3；预检健康状态为 OK，7 号卡空闲，用于既有入口边界回归 |
| CATLASS / 只读 CANN 源 | `769cd40a8716b28650b6bebb08db4834eea4462f` / `ca18aee610729ca5270a44e2d84c751bc687fbf0`；CATLASS 干净，CANN 原有工作区差异未变 |
| 实际加载扩展 | `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `277e55d03ca46c77a2b1691ebc8b6b2de11b2c3bc8298ef13a214977d13acd73`，沿用已核验生产构建 |

`environment.json` 保存版本、芯片状态及实际扩展路径/哈希；加载时 CANN 的版本/tiling 注册警告保留在 `environment.log`，进程正常退出。CPU 检查设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，只验证 CPU 视图；910B 回归加载上述 CANN 环境。原有未跟踪 `kernel_meta/` 保留。

此前 4/5 号卡固定源 probe 的 AICore 超时根因仍未定，本次未使用这两张卡；历史证据见 `build/mla-m06d3b1-varlen-used/timeout-analysis.json` 和 `timeout-evidence/`。上步 profiler 的 `fwk_api_db` 解析失败仍只按原始设备任务表验收，不记为完整 framework trace 通过。

## 最小复现与交接

新 CPU 组件的最小闭环不需 CANN/NPU。sanitizer 同一命令增加 `MLA_BSND_LAYOUT_SANITIZE=1`、`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` 与 `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`；若沙箱不支持 LeakSanitizer，须在允许其线程检查的环境运行，不能把环境失败计为通过。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
MLA_BSND_LAYOUT_DIR=build/mla-m06e1a-bsnd-layout/recheck \
python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_bsnd_layout.py \
  --basetemp=build/mla-m06e1a-bsnd-layout/recheck-tmp
```

完整 sanitizer 和既有入口边界复现见 `sanitizer-final-command.json` 与 `public-boundary-command.json`。后续修改共享头或生产源码后必须重建扩展并核对加载哈希，再按 plan 验证受影响路径；上步混合入口的完整最小命令保存在 `build/mla-m06d3b2-mixed-used/source-snapshots/docs/mla/status.md`。

文档链接/锚点、步骤拆分、官方签名、协议与来源核对结果保存在 `docs-check.json`；最终质量检查与证据索引分别为 `quality-staged.log`、`acceptance.json`。复现命令如下：

```bash
python3 build/mla-m06e1a-bsnd-layout/verify_baseline.py
python3 build/mla-m06e1a-bsnd-layout/check_docs.py
PYTHONPATH=/tmp/mla-m01-precommit${PYTHONPATH:+:$PYTHONPATH} \
PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache ci/code_quality.sh --staged
python3 build/mla-m06e1a-bsnd-layout/acceptance.py
git diff --check
```

后续只推进 M06e1b；生产 stride 接入在 M06e1c，TND/混合 query stride 在 M06e2。
