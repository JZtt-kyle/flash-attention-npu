# MLA 当前状态

最近功能闭环为 **M06d3b1：公开 varlen used 接入**，提交 `e461dc9`（2026-10-10，分支 `feat/mla-cann-migration`）。本次基于该提交核验并整理三份文档，未推进功能步骤。唯一下一步仍为 **[M06d3b2：内部混合 Q-used 接入](plan.md#下一步m06d3b2)**，尚未开始。

本页是当前能力与交接入口；目标契约、协议和架构见 [design](design.md)，迭代顺序与完成条件见 [plan](plan.md)。最近文档提交可用 `git log -1 -- docs/mla/status.md` 定位，不能将它当作最近功能提交。

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
| 同一内部入口，TND Q + BSND KV | Q 整段、KV 实际长度与 v4 已接通；Q-used 和 `max_seqlen_k` 仍拒绝 |
| v5 独立生成、消费者与 attention probe | BSND、TND、混合三布局均已验收；混合 Q-used 尚无生产入口 |

入口依据为 [Python 包装/fake](../../flash_attn_npu_3/flash_attn_npu_interface.py)、[C++ 通用分派](../../csrc/ascend910/flash_attn_npu_3/flash_api.cpp)、[专用前向](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp) 和 [静态校验](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h)。生产 BSND/TND used 共用 `UsedForward`、v5 生成/校验与 `MlaUsed`，数值 Block / Tile 沿用原路径。

### 调用约束与输出

- **选项：** 只支持 `num_splits=1`。func/varlen/内部前向默认 1；kvcache/metadata 默认 0，须显式传 1。仅无 mask、`softcap=0`、`pack_gqa=None`、`sm_margin=0`、`attention_chunk=0`；metadata 的 `has_softcap=False`。scale 默认 `1/sqrt(576)`，显式值须在 FP32 有限范围，记录不固化 scale。
- **输出：** O 与输入同 dtype，LSE 为 FP32。BSND O/LSE 为 `[B,Sq,Hq,512]` / `[B,Hq,Sq]`，TND 为 `[Tq,Hq,512]` / `[Hq,Tq]`。请求 LSE 时 func/varlen 返回二项，kvcache 返回四项；后两项是同设备空 accumulator，dtype 随 PyTorch 默认浮点 dtype。
- **长度与记录：** 长度值在设备检查，不做 D2H。显式 v2/v4 须匹配当次 KV 长度，v4 还匹配 Q 前缀。损坏/过期记录由消费者 Trap 拒绝，空输出仍校验。v5 完整检查两轴分配及 `0<=used[b]<=本轴分配段长`，即使另一轴 used 为零；缺省 used 使用整个分配段，缩短 used 不移动后续样本起点。
- **空值与有效区：** 空 batch/query 保留空 shape；非零容量中的零 KV 样本有效行 `O=0、LSE=-inf`；全局空 KV 初始化整个输出分配区为 `O=0、LSE=+inf`。其他未使用 query 槽位不承诺输出值。

**待迁移能力仍明确拒绝：** 混合 Q-used、非连续外层 stride、PA、causal/window、自动或多 split、192/128 prefill、非零 softcap 与非默认调度选项。KV 追加、内置 rotary 等范围外组合也拒绝，完整边界见 [design 第 4 节](design.md#4-功能边界)。fake/compile 已有覆盖，完整 graph replay 留在 M16；目标包含 910C，生产平台检查仍只开放 910B。

## 最近功能闭环证据

M06d3b1 归档为 **`build/mla-m06d3b1-varlen-used/`**，以下路径均相对该目录。`acceptance.json` 为总索引，`*-command.json` / `*-exit.json` 保存命令、环境和退出码，`source-snapshots/` / `production-final/` 保存源码和构建产物。

| 验收面 | 最终结果 | 证据 |
| --- | --- | --- |
| 上一步最小复查 | 60 项；从源码重建双 dtype probe 与 v5 producer，检查 BSND used 精度/空值/更新/生命周期 | `baseline.xml`、`baseline-attention/`、`baseline-producer/` |
| 公开 varlen used | 468 项；真实/fake/私有 op/C++、双 dtype 源/数学精度、有效区/空值、四数组更新、compile/生命周期；含两 dtype 各 12 项真实 Trap | `production-tests.xml`、`production/`、`production-source/`、`production-tmp/` |
| 普通 FA3 / 接口 | 833 项；普通 metadata、compile、签名和 MLA 契约 | `ordinary-regressions.xml`、`ordinary-nodes.json` |
| 既有 MLA 入口 | 563 项；完整 BSND used 及 varlen/混合/kvcache 的相关入口回归 | `legacy-public-final.xml` |
| 构建 | 19 项；生成实例与 AICPU/header/设备依赖失效检查 | `build-tests.xml`、`production-build-manifest.json` |

以上合计 **1,883 项去重 pytest 通过，无跳过项**；60 项 baseline 与后续套件重叠，不再累加。质量结果另见 `quality-staged.log`。首轮失败记录保留，不计入最终通过结果。

### 性能结论与限制

公开对照为固定源原生 FIA 与 [varlen used 驱动](../../tests/mla_varlen_used_profile.py)，双 dtype 各 10 个固定用例。每 dtype 的原生分派为 5 例非 FD、1 例空 KV、4 例 FD；适用子集为 0、2、6、7、8、9。双方预热 5 次、五轮各 20 个 wall 样本，同卡同进程交替测量，CPU 亲和性 128..135。R=4 表示连续四次前向，**每次均包含 v5 生成**；源侧必要的压紧和输出恢复计入 E2E。

| dtype | 适用 R=1 E2E 比值范围 | 适用 R=4 E2E 比值范围 | 同调度组件 Kernel 比值范围 |
| --- | --- | --- | --- |
| FP16 | 0.505–0.966 | 0.387–0.808 | 1.016–1.883 |
| BF16 | 0.535–0.973 | 0.411–0.780 | 1.013–3.030 |

比值均为生产/固定源。适用公开用例逐例满足不超过 1；**组件仍有开销，不声称纯 Kernel 不劣化**。四例原生 FD 全部保留，单列为未迁移路径；长 KV 例 3 的 R=1/4 比值为 FP16 1.646/1.577、BF16 2.043/2.126，不计作同路径性能通过。

逐例调用段、生成、Kernel、转换与均摊值见 `performance-acceptance.json`，分派证明见 `native-dispatch-proof.json`，原始样本见 `performance-{fp16,bf16}-card7/`。同输入/基本块/核数/端点的双 dtype 各 10 例组件对照及 v5 生成耗时见 `component-{fp16,bf16}/`；其预分配源 probe 不代表公开 E2E。验收口径见 [design 第 7 节](design.md#7-验收口径)。

保留两项环境/工具限制，接手实机验证时需注意：

- 4/5 号卡曾在独立 CANN probe 出现 AICore 超时，根因未定，不能认定硬件故障。6 号卡该例连续 20 轮、60 次前向及最终 563 项旧入口回归通过；新入口与最终性能在 7 号卡通过。详见 `timeout-analysis.json`、`timeout-evidence/`，失败运行保存在 `legacy-public.xml`、`timeout-recheck.xml` 及对应日志。
- profiler 有 step_trace DB 警告及组件 framework DB 关联导出失败，未记为完整 framework trace 通过。最终计时由调用顺序和原始 CANN `op_summary` 核对；每 dtype 公开 3,700 条、组件 1,500 条设备任务与 `kernel_details.csv` 逐项匹配。详见 `profiler-diagnostics.json`；早期性能失败及解析修正后的最终采样分别归档。

## 验收环境与本次核验

下表是 **M06d3b1 的实测环境**，不是本次重新探测或加载扩展的结果。本次只做静态文档核验；未运行构建、Python/NPU 功能测试或性能采样。

| 项目 | 最近功能验收记录 |
| --- | --- |
| Python / PyTorch | `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`、torch_npu `2.7.1` |
| CANN / 编译器 | `/usr/local/Ascend/cann-9.0.0-beta.2`；bisheng clang 15.0.5 |
| 芯片 | 八张 Ascend910B3；新入口/性能用 7，普通 FA3 和最终旧入口回归用 6；4/5 的超时证据如上 |
| CATLASS / 只读 CANN 源 | `769cd40a8716b28650b6bebb08db4834eea4462f` / `ca18aee610729ca5270a44e2d84c751bc687fbf0`；CATLASS 干净，CANN 原有工作区差异见 `precheck.json` |
| 生产构建 | 29 个链接对象、251 个源码依赖；最终精度、回归和性能期间源码及扩展哈希不变 |
| 当时实际加载扩展 | `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `b88c02a9f9e0a70df9a38f014c12bb31bcd8e7406cac60d6adcb72bfc1462fa8` |

本次文档核验（2026-10-10）已复核归档索引中的 **2,659 个文件**、当前 251 个源码依赖、29 个对象及扩展/链接产物哈希，全部一致；固定官方 FA3 的八个快照文件也与归档清单一致。依据源码修正 v5 接入/生命周期、补齐 `MlaUsed` 架构与 14 例三布局 used 组件矩阵，并区分组件、生产入口和未来目标。文档核验材料在 `build/mla-doc-audit-20261010/`；工作区原有 `kernel_meta/` 保留。

## 最小复现与交接

功能复验前重新检查环境、健康空闲卡和实际加载扩展，设置 `MLA_DEVICE` 并加载上述 CANN 的 `set_env.sh`。以下沿用上一步最小命令，从源码重建独立 probe，检查 T01、空值、四数组更新/生命周期和空 KV 坏长度 Trap；**本次仅核对命令与已有测试/归档，未执行它**。

```bash
ASCEND_RT_VISIBLE_DEVICES=${MLA_DEVICE:?set MLA_DEVICE} \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MLA_USED_ATTENTION_REUSE=0 \
MLA_USED_ATTENTION_DIR=build/mla-m06d3b1-varlen-used/recheck-attention \
MLA_USED_METADATA_DEVICE_DIR=build/mla-m06d3b1-varlen-used/recheck-producer \
MLA_VARLEN_USED_DIR=build/mla-m06d3b1-varlen-used/recheck-production \
python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_varlen_used.py \
  -k '(production_accuracy and spec0) or shapes_empty_and_defaults or four_array_updates or alias_clone or cross_stream_allocator_lifetime or (runtime_trap and empty_k)' \
  --basetemp=build/mla-m06d3b1-varlen-used/recheck-tmp
```

完整功能命令见功能归档的 `production-tests-command.json`、`ordinary-regressions-command.json`、`legacy-public-final-command.json`、`performance-{fp16,bf16}-card7-command.json` 和 `component-{fp16,bf16}-command.json`。共享头或生产源码变化后先用 `FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 python3 setup.py build_ext --inplace` 重建，核对加载路径、设备依赖及对象哈希，再补受影响回归。Kernel/输出变化须重跑相关源/数学精度与双 dtype 性能。

本次文档检查的复现命令如下；质量工具沿用既有 `/tmp` 安装与缓存，保留 CANN 的 PYTHONPATH：

```bash
python3 build/mla-doc-audit-20261010/check_evidence.py
python3 build/mla-doc-audit-20261010/check_docs.py
PYTHONPATH=/tmp/mla-m01-precommit${PYTHONPATH:+:$PYTHONPATH} \
PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache ci/code_quality.sh --staged
git diff --check
```

本次文档链接/锚点、表格、官方签名、协议常量、步骤一致性及复现命令静态检查通过；`git diff --check` 和适用质量钩子通过。结果见本次文档核验目录的 `evidence-check.json`、`docs-check.json`、`quality-staged.log`。唯一后续功能步骤为 M06d3b2，完成条件以 plan 为准；910C 实现兼容和完整 graph replay 尚未验收。
