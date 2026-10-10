# MLA 当前状态

最近闭环为 **M06e2a：TND CPU 地址校验**（2026-10-10，分支 `feat/mla-cann-migration`）。新增独立 TND stride 描述与完整 storage 校验；生产能力仍停在 M06e1c，非连续 TND/混合输入仍拒绝。唯一下一步为 **[M06e2b：TND 与混合布局独立 stride attention](plan.md#下一步m06e2b)**。

本页是当前能力与交接入口；契约、协议和架构见 [design](design.md)，迭代顺序与完成条件见 [plan](plan.md)。本次基于 `4e83d327ee8b4a74a585d072949886e541dffa77`；当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。

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
| TND stride CPU 组件 | 单 tensor 的 64 位 token/head stride 与完整 storage 校验；四路独立视图、分配前缀/used/BSND cache 地址组合已验收，未接入设备或生产 |
| v5 独立生成、消费者与 attention probe | BSND、TND、混合三布局均已验收；生产接入见上述入口 |

生产入口依据为 [Python 包装/fake](../../flash_attn_npu_3/flash_attn_npu_interface.py)、[C++ 通用分派](../../csrc/ascend910/flash_attn_npu_3/flash_api.cpp)、[专用前向](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp) 和 [静态校验](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h)。生产 BSND/TND/混合 used 共用 `UsedForward`、v5 生成/校验与 `MlaUsed`，数值 Block / Tile 沿用原路径。

### 调用约束与输出

- **选项：** 只支持 `num_splits=1`。func/varlen/内部前向默认 1；kvcache/metadata 默认 0，须显式传 1。仅无 mask、`softcap=0`、`pack_gqa=None`、`sm_margin=0`、`attention_chunk=0`；metadata 的 `has_softcap=False`。scale 默认 `1/sqrt(576)`，显式值须在 FP32 有限范围，记录不固化 scale。
- **输出：** O 与输入同 dtype，LSE 为 FP32。BSND O/LSE 为 `[B,Sq,Hq,512]` / `[B,Hq,Sq]`，TND 为 `[Tq,Hq,512]` / `[Hq,Tq]`。请求 LSE 时 func/varlen 返回二项，kvcache 返回四项；后两项是同设备空 accumulator，dtype 随 PyTorch 默认浮点 dtype。
- **长度与记录：** 长度值在设备检查，不做 D2H。显式 v2/v4 须匹配当次 KV 长度，v4 还匹配 Q 前缀。损坏/过期记录由消费者 Trap 拒绝，空输出仍校验。v5 完整检查两轴分配及 `0<=used[b]<=本轴分配段长`，即使另一轴 used 为零；缺省 used 使用整个分配段，缩短 used 不移动后续样本起点。
- **空值与有效区：** 空 batch/query 保留空 shape；非零容量中的零 KV 样本有效行 `O=0、LSE=-inf`；全局空 KV 初始化整个输出分配区为 `O=0、LSE=+inf`。其他未使用 query 槽位不承诺输出值。

**待迁移能力仍明确拒绝：** TND/混合 query 的非连续输入、PA、causal/window、自动或多 split、192/128 prefill、非零 softcap 与非默认调度选项。KV 追加、内置 rotary 等范围外组合也拒绝，完整边界见 [design 第 4 节](design.md#4-功能边界)。fake/compile 已有覆盖，完整 graph replay 留在 M16；目标包含 910C，生产平台检查仍只开放 910B。

## 最近闭环与证据

本步证据目录为 **`build/mla-m06e2a-tnd-layout/`**，以下路径均相对该目录。`acceptance.json` 保存源码、独立构建、测试与证据哈希；`*-command.json`、`*.log`、`*-exit.json` 保存命令、环境和结果。`final-sources/` 保存本步源码快照，`final-staged.patch` 保存提交差异。

- 新增 [`BuildTndTensorLayout`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_layout.h)，将 TND 视图映射为单 batch 的 BSND 视图，复用已有逐次乘加溢出和 storage 检查。描述只保存相对 `data_ptr()` 的 `tokenStride/headStride/spanElements`；storage offset 不重复加到设备地址。
- 接受末维连续、外层非负 stride，包括 NTD、切片、广播、重叠、空 token 和单例大 stride。非空时检查整个分配区与 offset 的有符号 64 位字节界限，不能因 used 缩短或为零而缩小范围；失败不修改输出描述。
- [`CPU 验收`](../../tests/test_flash_attn_npu_v3_mla_tnd_layout.py) 使用独立 Python 大整数 oracle 和真实 FP16/BF16 PyTorch 标量地址。四路 Q/qv/K/V 分别带 stride/offset，组合 T01、分配间隙、单轴/双轴/全零 used 与混合 BSND cache；从现有分配解析器取得起点，再与原始分配长度独立核对，检查 poison 槽位和最后一个分配字节越界。
- 本步未修改 Kernel、Block、Tile、metadata、公共入口或生产构建；新组件仅链接独立 CPU 驱动，CANN/CATLASS 源均未修改。

| 验收面 | 结果 | 证据 |
| --- | --- | --- |
| 接手环境与上步闭环 | 上步最终验收退出码全为 0；253 份生产源码依赖、30 个对象及实际加载扩展哈希一致 | `environment.*`；上步 `build/mla-m06e1c-bsnd-stride/` |
| 新增 TND CPU 地址 | 101 项通过，双 dtype 真实视图、64/512 维、65535/65536/65537、uint32/有符号字节边界、1,200 组随机大整数及组合地址 | `cpu.xml`、`cpu.log`、`cpu-tnd/` |
| 受影响 CPU 回归 | BSND 地址 82 项、固定 CANN 源 used 调度差分 306 项通过 | `cpu.xml`、`cpu-bsnd/`、`cpu-used/` |
| ASan/UBSan 批量重放 | TND 1,375 个、BSND 1,408 个地址描述，20 组分配/used 组合；6,187 组调度各自通过迁移、固定源无 split 与源原生分派重放；stderr 为空，输出与已核对 golden 逐字节一致 | `sanitizer.*`、`sanitized/report.json`、`sanitized/*-input.txt`、`sanitized/*-output.txt` |
| 提交前质量与文档 | 本步 6 个文件的适用质量钩子、差异检查及三份文档的链接/锚点、表格、步骤和源码一致性检查通过 | `quality-staged.*`、`docs.*`、`docs-check.json` |

CPU pytest 共 **489 项通过，无失败或跳过**；sanitizer 重放不重复计入 pytest 数量。重放使用同一源码重新构建，保留 ASan/UBSan 和泄漏检查；普通 CPU 测试负责真实 PyTorch 地址/数值及独立 oracle 比对，sanitizer 对保存的相同输入批量重放，减少重复进程启动成本。

初次环境脚本使用了不存在的 bisheng 路径，已改为实际 PATH 中的编译器；证据为 `environment-initial.*`。首轮 sanitizer 在沙箱内因 LeakSanitizer 无法检查进程而失败，保留于 `cpu-sandbox.*`；沙箱外逐用例运行在 38 项通过后主动中断，改为完整 CPU 验证加批量 sanitizer，保留于 `cpu-interrupted.*`。这些轮次均未记为完整通过。

最新生产精度/性能闭环仍为 `4e83d32`（M06e1c），证据在 `build/mla-m06e1c-bsnd-stride/acceptance.json`、`performance-acceptance.json` 及双 dtype `component-*/`、`public-*/`。其 2,041 个去重用例和适用七例公开 E2E 通过；大 stride/广播的组件开销、原生 FD 长 KV 对照退化及 profiler host 关联解析限制仍保留，不能据此宣称所有组件不劣化。本轮未重跑 NPU 精度、性能或普通 FA3 实机回归：生产源码、构建产物和加载环境未变化，本步只新增独立 CPU 地址组件。

## 环境与已知限制

Python `/usr/local/python3.11.15/bin/python3` 3.11.15，torch `2.7.1+cpu`，torch_npu `2.7.1`，CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，bisheng clang 15.0.5，g++ 11.4。8 张 Ascend910B3，型号、健康与占用保存在 `environment.json`；接手时 7 号卡存在进程，本步只做环境加载检查，验收计算均在 CPU。

CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f` 干净；只读 CANN 源 `ca18aee610729ca5270a44e2d84c751bc687fbf0` 的原有差异保留，原有未跟踪 `kernel_meta/` 未纳入提交。

实际加载扩展仍为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 **`d3eefbee09f643dc6fe6db2c89d0055db218a3d3b30f47b9a9422b8cedd9eea1`**；本步没有重建生产扩展。后续构建应使用已核验的 CATLASS 并设置 `FLASH_ATTN_SKIP_SUBMODULE_INIT=1`。沙箱内 NPU 型号检查不可靠，LeakSanitizer 进程检查受限；本轮相应检查均在获批的沙箱外完成。910C、完整 graph replay 和 TND stride 设备执行仍未验收。

## 唯一下一步与复现

下一步仅为 **[M06e2b：TND 与混合布局独立 stride attention](plan.md#下一步m06e2b)**，覆盖 v3/v4/v5 和双 dtype 源/数学精度、代表性组件性能及连续/普通 FA3 回归；生产接入留在 M06e2c。本轮提交后停止，不开始下一步。

完整 CPU 验收复现（新输出目录）：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
MLA_TND_LAYOUT_DIR=build/mla-m06e2a-tnd-layout/recheck-tnd \
MLA_BSND_LAYOUT_DIR=build/mla-m06e2a-tnd-layout/recheck-bsnd \
MLA_USED_SCHEDULE_ARTIFACT_DIR=build/mla-m06e2a-tnd-layout/recheck-used \
python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_tnd_layout.py \
  tests/test_flash_attn_npu_v3_mla_bsnd_layout.py \
  tests/test_flash_attn_npu_v3_mla_used_schedule.py
```

逐用例 sanitizer 复现可在同一命令前增加 `MLA_TND_LAYOUT_SANITIZE=1 MLA_BSND_LAYOUT_SANITIZE=1 MLA_SCHEDULE_SANITIZE=1`，须在允许 LeakSanitizer 进程检查的环境执行，耗时较长。本次批量重放的准确命令见 `sanitizer-command.json`，输入、golden、编译命令、二进制及哈希保存在 `sanitized/`，重放脚本为 `replay_sanitized.py`。

上步最小生产精度命令仍有效，使用已核验的生产扩展和独立探针：

```bash
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
MLA_STRIDE_DIR=build/mla-m06e1c-bsnd-stride/final-probes \
MLA_STRIDE_PRODUCER_DIR=build/mla-m06e1c-bsnd-stride/producer MLA_STRIDE_REUSE=1 \
MLA_PRODUCTION_STRIDE_DIR=build/mla-m06e2a-tnd-layout/recheck-production \
python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_bsnd_stride_production.py
```
