# MLA 当前状态

最近闭环为 **M06e1c：BSND stride 生产接入**（2026-10-10，分支 `feat/mla-cann-migration`）。BSND 四路外层 stride 已接通整容量、实际 KV 长度及内部 Q-used；唯一下一步为 **[M06e2：TND 合法外层 stride](plan.md#下一步m06e2)**。

本页是当前能力与交接入口；契约、协议和架构见 [design](design.md)，迭代顺序与完成条件见 [plan](plan.md)。本次基于 `c1db9a2aea82762a05ab7dd15b1acea049b87fe0`；当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。

## 当前可调用能力

生产共同边界为 **910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、无梯度**。Q/K 的末维为 64，qv/V 为 512；BSND 支持末维连续、外层非负 stride，包括 BNSD、切片、广播、重叠及四路独立 storage offset；TND 与混合 query 仍要求输入连续。BSND 每样本 KV 容量与 TND 的 `max_seqlen_k` 均为 0..16384；TND 总 Q/KV 各不超过 `INT32_MAX`，总 KV 可以超过 16384。其余字段、地址和资源限制见 [design 6.2](design.md#62-调度字段与边界)。

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
| BSND stride | 四路 64 位 stride/offset 已接入生产 v1/v2/v5；发射前检查完整分配视图的 storage/字节边界，metadata 不绑定 stride，不复制完整 cache |
| v5 独立生成、消费者与 attention probe | BSND、TND、混合三布局均已验收；生产接入见上述入口 |

生产入口依据为 [Python 包装/fake](../../flash_attn_npu_3/flash_attn_npu_interface.py)、[C++ 通用分派](../../csrc/ascend910/flash_attn_npu_3/flash_api.cpp)、[专用前向](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp) 和 [静态校验](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h)。生产 BSND/TND/混合 used 共用 `UsedForward`、v5 生成/校验与 `MlaUsed`，数值 Block / Tile 沿用原路径。

### 调用约束与输出

- **选项：** 只支持 `num_splits=1`。func/varlen/内部前向默认 1；kvcache/metadata 默认 0，须显式传 1。仅无 mask、`softcap=0`、`pack_gqa=None`、`sm_margin=0`、`attention_chunk=0`；metadata 的 `has_softcap=False`。scale 默认 `1/sqrt(576)`，显式值须在 FP32 有限范围，记录不固化 scale。
- **输出：** O 与输入同 dtype，LSE 为 FP32。BSND O/LSE 为 `[B,Sq,Hq,512]` / `[B,Hq,Sq]`，TND 为 `[Tq,Hq,512]` / `[Hq,Tq]`。请求 LSE 时 func/varlen 返回二项，kvcache 返回四项；后两项是同设备空 accumulator，dtype 随 PyTorch 默认浮点 dtype。
- **长度与记录：** 长度值在设备检查，不做 D2H。显式 v2/v4 须匹配当次 KV 长度，v4 还匹配 Q 前缀。损坏/过期记录由消费者 Trap 拒绝，空输出仍校验。v5 完整检查两轴分配及 `0<=used[b]<=本轴分配段长`，即使另一轴 used 为零；缺省 used 使用整个分配段，缩短 used 不移动后续样本起点。
- **空值与有效区：** 空 batch/query 保留空 shape；非零容量中的零 KV 样本有效行 `O=0、LSE=-inf`；全局空 KV 初始化整个输出分配区为 `O=0、LSE=+inf`。其他未使用 query 槽位不承诺输出值。

**待迁移能力仍明确拒绝：** TND/混合 query 的非连续输入、PA、causal/window、自动或多 split、192/128 prefill、非零 softcap 与非默认调度选项。KV 追加、内置 rotary 等范围外组合也拒绝，完整边界见 [design 第 4 节](design.md#4-功能边界)。fake/compile 已有覆盖，完整 graph replay 留在 M16；目标包含 910C，生产平台检查仍只开放 910B。

## 最近闭环与证据

本步证据目录为 **`build/mla-m06e1c-bsnd-stride/`**，以下路径均相对该目录。`acceptance.json` 记录最终源码、构建、测试和原始性能文件哈希；`*-command.json`、`*.log`、`*-exit.json` 保存命令、环境、日志与退出码。`final-sources/` 保存源码快照，`final-staged.patch` 保存提交差异。

- 生产 host 复用 `BuildBsndTensorLayout`，分别校验 Q/qv/K/V 的完整 storage 与有符号字节界限，并检查真实跨度的指针溢出；used 为零也不缩小校验范围。launcher 依据四路连续性选择既有 `HasStrides` 模板，指针不重复加 offset。
- BSND 整容量 v1、实际 KV 长度/公开 metadata 复用 v2、内部 Q-used v5 均消费原视图。metadata 协议、数值 Block / Tile、TND/混合实现及源调度不变；未修改 CATLASS 或 CANN 源。
- 真实/fake/C++/compile 接受范围一致；Dynamo 跟踪时避免包装器中的 `storage_offset()` 与符号 `math.isfinite()`，完整 storage/scale 检查保留在 fake/C++ 算子内。已编译调用更换 stride/长度/scale，以及运行时 storage 失效均有覆盖。

| 验收面 | 结果 | 证据 |
| --- | --- | --- |
| 接手基线 | 上步独立 stride 110 项通过；旧生产 250 份源码依赖、29 个对象与实际加载扩展一致 | `environment.*`、`baseline.*`、`baseline-archive-restored.json` |
| CPU 地址/构建/性能工具 | 112 项通过，无失败/跳过，保留 ASan/UBSan | `cpu.*` |
| 生产 stride | 278 项通过，无失败/跳过；双 dtype、公开/原始 C++、v1/v2/v5、空值、广播/重叠、65535/65536/65537、真实 >8 GiB storage、fake/compile、metadata/scale 复用与流生命周期 | `production-reviewed.*`、`accuracy.jsonl`、`final-probes/`、`producer/` |
| 加强后的补验 | 149 项通过，无失败/跳过；逻辑字节数仍可放下而真实 stride 跨度越界的拒绝、源/生产相同发射核数、metadata 复用 | `production-supplement.*`、`supplement/` |
| 连续 MLA 回归 | 462 项通过，无失败/跳过；dense、BSND 实际长度、BSND used 精度与输入拒绝 | `continuous.*`、`continuous-nodes.json`、`reg-*/` |
| 普通 FA3 与入口回归 | 1,189 项通过，无失败/跳过；签名/分派/metadata 契约、普通 BSND/TND、kvcache/FD、softcap/window、compile 和混合 stride 拒绝 | `ordinary-regressions.*`、`ordinary-nodes.json` |
| 构建与实际加载 | 独立探针和生产重建成功；253 份源码依赖、30 个对象及实际加载扩展核对一致 | `production-build-final.*`、`production-build-manifest.json`、`production-final/`、`environment-final.*` |
| 源分派与性能 | 双 dtype 固定 9 例生产 Kernel、10 例公开 E2E；适用 7 例各项 E2E 达标 | `native-dispatch-proof.json`、`performance-acceptance.json`、`component-{fp16,bf16}/`、`public-{fp16,bf16}/` |
| 提交前质量与文档 | 22 个本步文件通过适用质量检查和差异检查；链接/锚点、表格、公开签名、五版协议及固定矩阵核验通过 | `quality-staged.*`、`docs-closed.*`、`docs-check.json` |

本步五组测试共 **2,041 个去重用例通过**，149 项补验不重复计数；上步 110 项基线另列。生产精度对照固定源和独立高/低精度双 reference，零 KV 的源/FA3 LSE 符号差异单独检查。14 项独立进程负例确认损坏/过期 metadata、非法 used 在设备真实 Trap，无长度 D2H。

初轮发现的 Dynamo 校验限制已修复；负例中的 NPU `Storage.resize_` 会触发异步复制，已改为只改主机 storage 字节元数据、并在 finally 恢复的测试探针。旧 producer 哈希和缺失 fixture 也已纠正。失败轮保留于 `production-initial.*`、`production-current.*`、`production-compile-fix.*`、`compile-focused.*`、`production-final.*` 和 `storage-fixture-diagnosis.json`，未记为通过。

### 公开 metadata E2E

用例在采样前固定于 [STRIDE_CASES](../../tests/mla_profile_cases.py)。双方接收相同物理视图；H2D 和一次性视图准备对称排除，原生 CANN 每次 AsStrided/BroadcastTo 转换保留在 E2E 内。7 号卡采样前后健康且只有本次测量进程；固定 CPU 0..7、预热 5 次、五轮交替采样，每例每调用段 20 个同步 wall 样本。每 dtype 保存 7,000 个 wall 样本和 3,250 条设备任务，其中 2,250 条为 CANN 转换；FA3 没有输入转换任务。

下表为中位数的 **FA3 / 固定原生 CANN** 比值。复用单列一次预生成记录；“生成+R”从公开 metadata 生成开始，包含 R 次前向，CANN 完成同样 R 次调用。FP16/BF16 的适用七例三项均 `<=1.0`，最高分别为 **0.957 / 0.968**。绝对耗时、生成与隐式前向、转换耗时和逐轮样本见各 dtype 的 `profile.json`、`wall-samples.json`。

| 例 / 布局（B,Sq,Hq,Sk） | FP16 复用 / 生成+1 / 生成+4 | BF16 复用 / 生成+1 / 生成+4 | 分派 |
| --- | --- | --- | --- |
| 0 / contiguous (2,1,8,257) | 0.677 / 0.957 / 0.584 | 0.661 / 0.968 / 0.591 | 适用，通过 |
| 1 / bnsd (2,7,16,513) | 0.620 / 0.827 / 0.524 | 0.595 / 0.798 / 0.484 | FD，单列 |
| 2 / independent (2,2,128,1025) | 0.602 / 0.770 / 0.509 | 0.550 / 0.716 / 0.471 | FD，单列 |
| 3 / independent (2,7,16,33) | 0.604 / 0.831 / 0.501 | 0.565 / 0.786 / 0.467 | 适用，通过 |
| 4 / independent (2,3,8,16384) | 2.299 / 2.402 / 3.287 | 2.090 / 2.204 / 2.867 | FD，单列 |
| 5 / matrix_65536 (1,7,8,33) | 0.634 / 0.871 / 0.515 | 0.571 / 0.787 / 0.475 | 适用，通过 |
| 6 / head_65536 (1,7,8,33) | 0.607 / 0.860 / 0.520 | 0.595 / 0.823 / 0.478 | 适用，通过 |
| 7 / broadcast (2,7,8,33) | 0.564 / 0.786 / 0.518 | 0.565 / 0.800 / 0.482 | 适用，通过 |
| 8 / overlap (2,7,8,33) | 0.589 / 0.833 / 0.517 | 0.587 / 0.824 / 0.498 | 适用，通过 |
| 9 / bnsd (2,7,8,33) | 0.597 / 0.837 / 0.502 | 0.531 / 0.747 / 0.473 | 适用，通过 |

第 1/2/4 例的原生 FD 根据固定源 CPU 调度及实际二进制 tiling key 双重核验后单列，不按耗时筛选；第 4 例明显慢于原生 FD，保留完整结果。双方同路径的验收仍使用严格 1.0 门槛。首次误选混合布局源包导致入口 key 缺失，证据为 `native-wrong-vendor-*`；改用已核验的 BSND 包后双 dtype 全部成功。首次性能报告仅识别 Slice，遗漏实际 AsStrided/BroadcastTo 任务；首轮完整数据保留于 `public-copy-parser-initial-fp16/`，修正统计后重新完成双 dtype 采样。

### 生产 Kernel 对照

固定 [生产 PERFORMANCE_CASES](../../tests/test_flash_attn_npu_v3_mla_bsnd_stride_production.py) 共 9 例；双方逻辑输入、非 FD 调度和发射核数一致。源侧连续物化在计时前完成，物理布局不同；生产 wall 含逐次内部 metadata 生成，源 probe 使用预分配缓冲区，因此此表不替代公开 E2E。每 dtype 保存 1,800 个 wall 样本、900 个 attention 样本和 450 个生产 AICPU 任务。下表单位为 μs，均为中位数。

| 例 / 协议 / 布局（B,Sq,Hq,Sk） | FP16 Kernel 迁移 / 源（比值） | BF16 Kernel 迁移 / 源（比值） |
| --- | --- | --- |
| 0 / v1 / contiguous (2,1,8,257) | 24.740 / 24.191 (1.023) | 25.191 / 24.180 (1.042) |
| 1 / v1 / bnsd (2,7,16,513) | 56.211 / 55.491 (1.013) | 57.551 / 56.741 (1.014) |
| 2 / v2 / independent (2,2,128,1025) | 84.722 / 87.311 (0.970) | 89.711 / 87.632 (1.024) |
| 3 / v5 / independent (2,7,16,33) | 43.061 / 22.991 (1.873) | 44.191 / 22.530 (1.961) |
| 4 / v5 / bnsd (5,9,171,1025) | 79.741 / 72.542 (1.099) | 88.332 / 75.001 (1.178) |
| 5 / v5 / independent (2,3,8,16384) | 853.008 / 948.089 (0.900) | 866.928 / 961.569 (0.902) |
| 6 / v1 / matrix_65536 (1,7,8,33) | 33.510 / 13.270 (2.525) | 32.910 / 13.291 (2.476) |
| 7 / v1 / head_65536 (1,7,8,33) | 26.271 / 12.950 (2.029) | 26.351 / 13.370 (1.971) |
| 8 / v1 / broadcast (2,7,8,33) | 38.621 / 15.700 (2.460) | 38.811 / 16.170 (2.400) |

大 stride/广播等输入的组件开销仍高于源侧连续输入；Kernel 比值范围为 FP16 **0.900..2.525**、BF16 **0.902..2.476**，不宣称组件不劣化。每例 AICPU、内部 wall、stride/offset、完整调度和原始任务见 `component-{fp16,bf16}/summary.json`、`wall.json`、`kernel-samples.json` 及 `profiler/`。

profiler 仍报 `fwk_api_db` / ACL→NPU flow 解析失败，未将 host 关联分析记为通过。四组性能的 `kernel_details.csv` 均与原始 `op_summary_*.csv` 按名称、设备、起始时间和耗时逐条核对，再按同步发射顺序关联用例；wall 独立测量。

## 环境与已知限制

Python `/usr/local/python3.11.15/bin/python3` 3.11.15，torch `2.7.1+cpu`，torch_npu `2.7.1`，CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，bisheng clang 15.0.5，g++ 11.4；8 张 Ascend910B3，本步仅使用 7 号卡。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f` 干净；只读 CANN 源 `ca18aee610729ca5270a44e2d84c751bc687fbf0` 的原有差异保留，原有未跟踪 `kernel_meta/` 未纳入提交。

最终实际加载路径为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 **`d3eefbee09f643dc6fe6db2c89d0055db218a3d3b30f47b9a9422b8cedd9eea1`**。相对基线仅新增/改变 BSND 地址对象、四个 BSND dispatch 对象、`mla_forward.o` 和 `flash_api.o`；普通 FA3 与 TND 对象哈希不变。构建中的子模块初始化曾因只读配置告警，但编译/链接成功；复现应设置 `FLASH_ATTN_SKIP_SUBMODULE_INIT=1`，使用已核验的 CATLASS。

原生 CANN 使用 `build/mla-m02b-validation/opp-dense/vendors/mla_ca18aee_transformer/`；host 库、设备二进制及编译来源见 `native-package.json`，实际加载见 `public-*/environment.json`，实际注册/tiling key 见 `native-dispatch-proof.json`。未将安装包路径或源码 commit 单独视为运行溯源。

沙箱内无法可靠读取 NPU 型号，实际加载及实机检查在获批的沙箱外执行；sanitizer 保持启用。910C 与完整 graph replay 未在本步验收；此前其他卡源 probe 超时根因仍未确认，本步未使用那些卡。

## 唯一下一步与复现

下一步仅为 **[M06e2：TND 合法外层 stride](plan.md#下一步m06e2)**，进入实现前先拆成独立验收的小步。本轮提交后停止，不开始下一步。

最小生产精度复现（使用已核验的当前生产扩展，加载器检查独立探针源码/二进制哈希）：

```bash
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
MLA_STRIDE_DIR=build/mla-m06e1c-bsnd-stride/final-probes \
MLA_STRIDE_PRODUCER_DIR=build/mla-m06e1c-bsnd-stride/producer MLA_STRIDE_REUSE=1 \
MLA_PRODUCTION_STRIDE_DIR=build/mla-m06e1c-bsnd-stride/recheck \
python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_bsnd_stride_production.py
```

CPU、连续和普通 FA3 的完整参数见 `cpu-command.json`、`continuous-command.json`、`ordinary-regressions-command.json`；构建参数见 `production-build-final-command.json`。公开性能复现使用新输出目录，BF16 将下面两处 `fp16` 改为 `bf16`：

```bash
MLA_CANN_VENDOR="$PWD/build/mla-m02b-validation/opp-dense/vendors/mla_ca18aee_transformer" \
ASCEND_CUSTOM_OPP_PATH="$PWD/build/mla-m02b-validation/opp-dense/vendors/mla_ca18aee_transformer" \
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
MLA_FORWARD_ARTIFACT_DIR=build/mla-m06e1c-bsnd-stride/recheck-public-fp16 \
MLA_FORWARD_DTYPE=fp16 MLA_FORWARD_METADATA_REUSE=1 MLA_FORWARD_BSND_STRIDES=1 \
taskset -c 0-7 python3 -m tests.mla_forward_profile
```

组件性能命令见 `component-{fp16,bf16}-command.json`；分派取证另加 `MLA_FORWARD_DISPATCH_ONLY=1`，完整日志开关见 `native-{fp16,bf16}-command.json`，不在正式计时中开启。
