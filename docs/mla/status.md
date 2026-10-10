# MLA 当前状态

最近闭环为 **M06e1b：BSND 独立 stride attention**（2026-10-10，分支 `feat/mla-cann-migration`）。四路 64 位 stride 已接通独立设备组件；生产能力仍为 M06d3b2，非连续输入尚未开放。唯一下一步为 **[M06e1c：BSND stride 生产接入](plan.md#下一步m06e1c)**。

本页是当前能力与交接入口；契约、协议和架构见 [design](design.md)，迭代顺序与完成条件见 [plan](plan.md)。本次基于 `70aae25054b9db3ec71c15e08d433930ce370bf0`；当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。

## 当前可调用能力

生产共同边界为 **910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、连续布局、无梯度**。Q/K 的末维为 64，qv/V 为 512；允许连续视图的非零 storage offset。BSND 每样本 KV 容量与 TND 的 `max_seqlen_k` 均为 0..16384；TND 总 Q/KV 各不超过 `INT32_MAX`，总 KV 可以超过 16384。其余字段、地址和资源限制见 [design 6.2](design.md#62-调度字段与边界)。

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
| BSND stride 独立组件 | 四路 64 位 stride/offset 接入 v1/v2/v5；CPU 地址、双 dtype 910B 源/数学精度、连续路径回归及组件性能对照均已完成，生产入口未接入 |
| v5 独立生成、消费者与 attention probe | BSND、TND、混合三布局均已验收；生产接入见上述入口 |

生产入口依据为 [Python 包装/fake](../../flash_attn_npu_3/flash_attn_npu_interface.py)、[C++ 通用分派](../../csrc/ascend910/flash_attn_npu_3/flash_api.cpp)、[专用前向](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp) 和 [静态校验](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h)。生产 BSND/TND/混合 used 共用 `UsedForward`、v5 生成/校验与 `MlaUsed`，数值 Block / Tile 沿用原路径。

### 调用约束与输出

- **选项：** 只支持 `num_splits=1`。func/varlen/内部前向默认 1；kvcache/metadata 默认 0，须显式传 1。仅无 mask、`softcap=0`、`pack_gqa=None`、`sm_margin=0`、`attention_chunk=0`；metadata 的 `has_softcap=False`。scale 默认 `1/sqrt(576)`，显式值须在 FP32 有限范围，记录不固化 scale。
- **输出：** O 与输入同 dtype，LSE 为 FP32。BSND O/LSE 为 `[B,Sq,Hq,512]` / `[B,Hq,Sq]`，TND 为 `[Tq,Hq,512]` / `[Hq,Tq]`。请求 LSE 时 func/varlen 返回二项，kvcache 返回四项；后两项是同设备空 accumulator，dtype 随 PyTorch 默认浮点 dtype。
- **长度与记录：** 长度值在设备检查，不做 D2H。显式 v2/v4 须匹配当次 KV 长度，v4 还匹配 Q 前缀。损坏/过期记录由消费者 Trap 拒绝，空输出仍校验。v5 完整检查两轴分配及 `0<=used[b]<=本轴分配段长`，即使另一轴 used 为零；缺省 used 使用整个分配段，缩短 used 不移动后续样本起点。
- **空值与有效区：** 空 batch/query 保留空 shape；非零容量中的零 KV 样本有效行 `O=0、LSE=-inf`；全局空 KV 初始化整个输出分配区为 `O=0、LSE=+inf`。其他未使用 query 槽位不承诺输出值。

**待迁移能力仍明确拒绝：** 非连续外层 stride、PA、causal/window、自动或多 split、192/128 prefill、非零 softcap 与非默认调度选项。KV 追加、内置 rotary 等范围外组合也拒绝，完整边界见 [design 第 4 节](design.md#4-功能边界)。fake/compile 已有覆盖，完整 graph replay 留在 M16；目标包含 910C，生产平台检查仍只开放 910B。

## 最近闭环与证据

本步证据目录为 **`build/mla-m06e1b-bsnd-stride/`**，以下路径均相对该目录；完整来源、测试、构建与原始性能文件哈希见 `acceptance.json`。命令、环境、日志及退出码保存为 `*-command.json`、`*.log`、`*-exit.json`；`final-sources/` 保存本次源码快照，`final-staged.patch` 保存提交差异。

- `MlaBsndInputLayouts` 保存 Q/qv/K/V 四份 64 位描述，`HasStrides` 实例沿既有 Kernel / Cube Block / Copy Tile 使用真实 batch/row/head stride。保留源 S1G 顺序、任务环、数值与同步，V 的布局同时用于内容 QK 和 PV，无完整 cache 复制；metadata 不保存 stride。
- 恢复逐行/逐矩阵 ND→NZ 回退，覆盖零 stride、独立 head/row stride 及组合边界。910B 的多矩阵字段为 uint16，源 `>65536` 判断在 65536 处截断；本步改为 `>=65536`，见 [design 6.2.8](design.md#628-bsnd-输入-stride-与地址校验)。首次 FP16 在此处失败（15 通过、1 失败），原始证据保留于 `fp16-initial.*`、`initial-probes/`、`initial-sources/`；修复后的最终源码通过完整双 dtype 验收。
- 独立驱动发射前检查四个完整视图的 shape/storage/stride，设备消费真实 v1/v2/v5。v1/v2 驱动明确拒绝全局空容量，空容量初始化仍属既有生产 host 分支；空/单例 stride 的独立设备覆盖使用 v5。
- 未修改公共/fake/C++ 入口、metadata 协议、Vector / MMAD Tile、setup、CATLASS 或 CANN 源。生产扩展因共享头重建并完成回归，默认模板仍使用连续布局；本步不开放生产 stride。

| 验收面 | 结果 | 证据 |
| --- | --- | --- |
| 恢复与上步基线 | 中断前 12 份源码、4,025 个归档文件哈希一致；此前上步归档与生产基线核验保留 | `resume-evidence-audit.json`、`prior-archive.json`、`baseline.*` |
| CPU 地址与构建回归 | 101 项通过，无失败/跳过；保留 ASan/UBSan 检查 | `cpu-verified.xml`、`cpu-verified-command.json` |
| 最终独立 stride 精度 | 110 项通过，无失败/跳过；双 dtype、v1/v2/v5、U01/poison、BNSD/独立偏移、广播/重叠、空/单例、65535/65536/65537 和组合回退 | `stride-reviewed.*`、`final-probes/` |
| 连续布局与生产 MLA | 994 项通过，无失败/跳过；dense、BSND 实际长度、三布局 used、TND/混合公开源/数学精度、输入拒绝及 dense 生命周期 | `continuous.xml`、`continuous-nodes.json`、`continuous-*` 构建目录 |
| 普通 FA3 与入口回归 | 833 项通过，无失败/跳过；签名/分派/契约、BSND/TND、kvcache、flash decode、softcap/window、显式 metadata 和 compile | `ordinary-regressions.xml`、`ordinary-nodes.json` |
| 构建与实际加载 | 最终独立探针和生产构建成功；250 个源码依赖、29 个对象与实际加载扩展一致；相对基线只改变 10 个 MLA dispatch 对象 | `build-reviewed.*`、`production-build-reviewed.*`、`production-reviewed-manifest.json`、`environment-resumed.json`、`production-reviewed/` |
| 组件性能 | 固定 9 例 × 双 dtype，全部采样完成；每 dtype 900 个 Kernel 样本与 1,800 个 wall 样本 | `performance-{fp16,bf16}/`，逐例结果见下 |
| 提交前质量与文档 | 本步 12 个文件通过适用质量检查与差异检查；文档链接/锚点、表格、四个公开签名、五版协议与固定矩阵核验通过 | `quality-staged.*`、`docs-closed.*`、`docs-check.json` |

上述四组测试共 **2,038 个去重用例通过**。最终 stride 精度中的非空有效行 O/LSE 与固定源逐元素一致，且通过独立高/低精度双 reference；零 KV 的 FA3/source LSE 符号差异单独检查。20 项大地址用例使用真实超过 8 GiB 的 NPU storage，最大 8,590,091,492 字节；重复执行、输入只读、输出/workspace 哨兵、metadata 不变、stride/scale 复用及无效视图发射前拒绝均有覆盖。首轮修复后的 104 项保留为 `stride-final.*`，不替代最终 110 项。

### 组件性能结果

用例在采样前固定于 [PERFORMANCE_CASES](../../tests/test_flash_attn_npu_v3_mla_bsnd_stride.py)。双方使用相同逻辑输入与非 FD 调度；源侧独立物化连续输入在计时前完成，输入物理布局不同。7 号卡采样前后健康且只有本次测量进程；双方固定 CPU 0..7，预热 5 次，五轮每例/每侧 20 个同步 wall 样本、10 个 profiler Kernel 样本。

表中耗时单位为 μs，均为中位数，顺序为 **迁移 / 源**；括号为 Kernel 比值。完整形状、used 长度、stride/offset、精度记录、逐轮原始样本、版本和分派见各 dtype 的 `summary.json`、`wall.json`、`kernel-samples.json` 及 `profiler/`。

| 例（B,Sq,Hq,Sk） | FP16 Kernel | BF16 Kernel | FP16 probe wall | BF16 probe wall |
| --- | --- | --- | --- | --- |
| v1 连续 (2,1,8,257) | 25.651 / 24.331 (1.054) | 25.431 / 24.650 (1.032) | 215.455 / 214.320 | 221.510 / 218.495 |
| v1 BNSD (2,7,16,513) | 55.401 / 54.181 (1.023) | 55.861 / 54.791 (1.020) | 246.645 / 246.330 | 250.665 / 250.835 |
| v2 独立布局 (2,2,128,1025) | 72.892 / 73.811 (0.988) | 73.221 / 74.992 (0.976) | 265.950 / 268.855 | 283.885 / 286.410 |
| v5 U01 (2,7,16,33) | 18.810 / 11.680 (1.610) | 19.121 / 11.780 (1.623) | 216.235 / 216.315 | 231.015 / 232.625 |
| v5 BNSD/单核 (5,9,171,1025) | 134.642 / 125.763 (1.071) | 135.493 / 126.723 (1.069) | 345.905 / 336.005 | 351.510 / 345.645 |
| v5 长 KV (2,3,8,16384) | 834.417 / 938.509 (0.889) | 835.997 / 937.219 (0.892) | 1057.150 / 1156.885 | 1065.915 / 1174.445 |
| v1 matrix=65536 (1,7,8,33) | 33.270 / 13.250 (2.511) | 33.861 / 13.321 (2.542) | 216.950 / 207.560 | 239.025 / 225.845 |
| v1 head=65536 (1,7,8,33) | 26.500 / 13.280 (1.995) | 26.561 / 13.230 (2.008) | 215.810 / 206.680 | 220.000 / 215.515 |
| v1 广播 (2,7,8,33) | 37.671 / 16.041 (2.348) | 35.501 / 15.940 (2.227) | 221.640 / 203.185 | 232.490 / 217.620 |

Kernel 比值为 FP16 **0.889..2.511**、BF16 **0.892..2.542**；U01、大 stride 和广播用例比源侧连续输入慢，全部保留并明确记录。此步完成的是组件对照，**不宣称不劣化，也不替代公开 metadata E2E**；生产接入时仍执行 [design 第 7 节](design.md#7-验收口径) 的适用门槛。

profiler 仍报 `fwk_api_db` / ACL→NPU flow 解析失败，未将该 host 关联分析记为通过。每个 dtype 的 900 条 `kernel_details.csv` 已与原始 `op_summary_*.csv` 按 Kernel 名、设备、起始时间和耗时逐项核对，再按同步发射顺序关联用例；本步 Kernel 数据采用这些设备证据，wall 独立测量。

## 环境与已知限制

Python `/usr/local/python3.11.15/bin/python3` 3.11.15，torch `2.7.1+cpu`，torch_npu `2.7.1`，CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，bisheng clang 15.0.5，g++ 11.4；8 张 Ascend910B3，本步只使用 7 号卡。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f` 干净；只读 CANN 源 `ca18aee610729ca5270a44e2d84c751bc687fbf0` 的原有差异保留，原有未跟踪 `kernel_meta/` 未纳入提交。

最终实际加载路径为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 **`10dda4f3ef71de9ca4c9b8325561fdf9f22c53301deb74d48be1ca5768b94116`**；加载、链接产物、对象及依赖核对见 `environment-resumed.json`。`environment.json` 与 `production-build-manifest.json` 分别保留开发前和首次重建记录，最终源码验收使用 `final-probes/` 与最终重建扩展。

早期沙箱内的 LeakSanitizer/ptrace 失败和 NPU 型号读取失败分别保留在 `cpu-final.*`、`production-audit.*`，已由沙箱外检查解决，没有关闭 sanitizer。中断前自动权限审核超时未创建测试进程，恢复后必要实机检查均已执行。此前 4/5 号卡源 probe 超时根因仍未确认，本步未使用这两张卡；910C 实机与完整 graph replay 未在本步验收。

## 唯一下一步与复现

下一步仅为 **[M06e1c：BSND stride 生产接入](plan.md#下一步m06e1c)**，本轮提交后停止。接入真实/fake/C++/compile、长度/metadata 更新与流生命周期，并完成适用公开 E2E 和普通 FA3 回归；TND/混合 stride 留在 M06e2。

最小独立精度复现（加载器核对源码与二进制哈希，归档构建保持不变时复用）：

```bash
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
MLA_STRIDE_DIR=build/mla-m06e1b-bsnd-stride/final-probes \
MLA_STRIDE_PRODUCER_DIR=build/mla-m06e1b-bsnd-stride/producer MLA_STRIDE_REUSE=1 \
python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_bsnd_stride.py
```

完整连续和普通 FA3 参数见 `continuous-command.json`、`ordinary-regressions-command.json`；CPU 检查见 `cpu-verified-command.json`。性能复现使用新输出目录，BF16 将两个 `fp16` 改为 `bf16`：

```bash
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
taskset -c 0-7 python3 -m tests.mla_bsnd_stride_profile \
  build/mla-m06e1b-bsnd-stride/final-probes/fp16 \
  build/mla-m06e1b-bsnd-stride/producer/build-manifest.json \
  build/mla-m06e1b-bsnd-stride/recheck-performance-fp16
```
