# MLA 当前状态

本轮为 **文档核验与整理**（2026-10-10），核验代码为 `2a8cd38`，分支 `feat/mla-cann-migration`。功能进度不变：生产已完成 **M06e1c（BSND stride）**，独立组件已完成 **M06e2a（TND CPU 地址校验）**。唯一下一步为 **[M06e2b：TND 与混合布局独立 stride attention](plan.md#下一步m06e2b)**。

本页维护当前能力、证据、环境差异和下一步；目标契约与架构见 [design](design.md)，步骤与完成条件见 [plan](plan.md)。当前文档提交可用 `git log -1 -- docs/mla/status.md` 定位，功能闭环提交见下方证据表。

## 当前可调用能力

共同边界为 **910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、无梯度**。Q/K 末维为 64，qv/V 为 512；四路输入须同 dtype、同 NPU 设备。BSND 每样本 KV 容量与 TND 的 `max_seqlen_k` 均为 0..16384；TND 总 Q/KV 各不超过 `INT32_MAX`，总 KV 可以超过 16384。其他字段、地址与资源限制见 design [6.2](design.md#62-调度字段与边界)、[6.3](design.md#63-长度与布局调度)、[6.4](design.md#64-输入-stride-与地址校验)。

### 布局与 stride

| 输入布局 | 生产支持 | 独立组件边界 |
| --- | --- | --- |
| BSND Q/KV | 四路末维连续、外层非负 stride；含 BNSD、切片、广播、重叠、独立 storage offset；完整 storage/字节界限在发射前校验 | CPU 校验、设备 attention 和生产接入均完成，覆盖 v1/v2/v5 |
| TND Q/KV | Q、qv、K、V 四路均须连续 | TND stride 只完成 CPU 地址校验，尚未链接生产或接入设备 |
| TND Q + BSND KV | Q、qv、K、V 四路均须连续，包含 BSND cache | CPU 已验证 TND 与 BSND 独立地址描述的组合；混合非连续输入仍拒绝 |

stride 与 storage offset 不进入 metadata；`data_ptr()` 已包含 offset。CPU 地址成立、metadata 能表达布局、生产入口可调用是不同层次。

### 公开入口

长度 tensor 均须同设备、连续、int32；B 为 batch，分配前缀为 `[B+1]`，实际长度为 `[B]`。公开显式生成/复用只开放 v2/v4，精确字数见 [metadata 协议](design.md#5-metadata-设计)。

| 入口 | 输入与长度 | metadata 路径 |
| --- | --- | --- |
| `flash_attn_func` | BSND Q/KV，使用整容量 | 每次内部 v1 |
| `flash_attn_varlen_func` | TND Q/KV，两组分配前缀；max 参数是每段分配上界；可选 `seqused_q/k` | 任一 used 存在时每次内部 v5，否则 v3；无显式复用参数 |
| `flash_attn_with_kvcache`，BSND Q | Q 整容量，只读 BSND cache；`cache_seqlens` 为 None（整容量）、`0..Sk` 整数或长度 tensor | 无长度且无记录走 v1，否则内部 v2 或显式 v2 |
| `flash_attn_with_kvcache`，TND Q | Q 使用完整分配段，须给 Q 前缀和 `max_seqlen_q`；只读 BSND cache，KV 长度同上 | 内部 v4 或显式 v4 |
| `get_scheduler_metadata` | 必须给 KV 长度 tensor，不接受 None/整数；MLA 维度为 `headdim=64, headdim_v=512` | 无 Q 前缀生成 v2，有 Q 前缀生成混合 v4 |

### 内部入口

以下为 `_flash_attn_forward` / C++ `fwd`，不增加公共签名。三种布局的 used 分支共用 `UsedForward`、v5 生成/校验与 `MlaUsed`。

| 输入 | 记录选择与边界 |
| --- | --- |
| BSND Q/KV | 有 `seqused_q` 时每次内部 v5，可选 `seqused_k`；无 Q-used 沿用 v1/v2；Q-used 与显式 metadata 组合拒绝 |
| TND Q/KV | 前缀、max、可选 used 及 v3/v5 选择与公开 varlen 相同；拒绝显式 metadata |
| TND Q + BSND KV | 无 Q-used 沿用 v4；有 `seqused_q` 时每次内部 v5，可选 `seqused_k`，复用 TND used launcher；Q-used 与显式 metadata 组合拒绝，混合入口始终拒绝 `max_seqlen_k` |

核验入口为 [Python 包装/fake](../../flash_attn_npu_3/flash_attn_npu_interface.py)、[C++ 通用分派](../../csrc/ascend910/flash_attn_npu_3/flash_api.cpp)、[专用前向](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp) 和 [静态校验](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h)；launcher 与 Kernel 映射见 [design 第 6 节](design.md#6-代码结构)。

### 调用约束与输出

- **选项：** 只支持 `num_splits=1`。func/varlen/内部前向默认 1；kvcache/metadata 默认 0，调用时须显式传 1。仅无 mask、`softcap=0`、`pack_gqa=None`、`sm_margin=0`、`attention_chunk=0`；metadata 的 `has_softcap=False`，不接受数值 `softcap/softmax_scale` 关键字。前向 scale 默认 `1/sqrt(576)`，显式值须在 FP32 有限范围，记录不固化 scale。
- **输出：** O 与输入同 dtype，LSE 为 FP32。BSND O/LSE 为 `[B,Sq,Hq,512]` / `[B,Hq,Sq]`，TND 为 `[Tq,Hq,512]` / `[Hq,Tq]`。请求 LSE 时 func/varlen 返回二项，kvcache 返回四项；后两项是同设备空 accumulator，dtype 随 PyTorch 默认浮点 dtype。
- **长度与复用：** 长度值在设备检查，不做 D2H。显式 v2/v4 须匹配当次 KV 长度，v4 还匹配 Q 前缀；损坏/过期记录由消费者 Trap 拒绝，空输出仍校验。v5 完整检查两轴分配与 used，即使另一轴 used 为零；缺省 used 使用整段，缩短 used 不移动后续样本起点。生命周期见 [design 5.7](design.md#57-stream-与内存生命周期)。
- **空值与有效区：** 空 batch/query 保留空 shape；非零容量中的零 KV 样本有效行 `O=0、LSE=-inf`；全局空 KV 初始化整个输出分配区为 `O=0、LSE=+inf`。其他未使用 query 槽位不承诺输出值。

**尚未开放：** TND/混合路径任一输入非连续、PA、causal/window、自动或多 split、192/128 prefill、非零 softcap 与非默认调度选项。KV 追加、内置 rotary 等范围外组合也明确拒绝，完整范围见 [design 第 4 节](design.md#4-功能边界)。fake/compile 已有覆盖，完整 graph replay 留在 M16；910C 尚未开放或实机验收。

## 最近闭环与证据

路径均相对仓库根目录，位于忽略的 `build/`，不随 Git 分发。本轮只核验源码、文档与已有证据，**未重跑 CPU 数值、NPU 精度、性能或普通 FA3 实机测试**。

| 闭环 | 核验结果 | 证据入口 |
| --- | --- | --- |
| 本轮文档核验，基于 `2a8cd38` | 对照 Python/C++、构建清单、固定 FA3 快照、CANN Git blob；检查协议字数、性能矩阵、步骤、链接/锚点、表格与复现命令语法 | `build/mla-docs-audit-20261010/docs-check.json`、`quality.log` |
| M06e2a，`2a8cd38` | CPU 489 项通过，无失败/跳过：TND 地址 101、BSND 地址 82、used 调度 306；ASan/UBSan 批量重放通过 | `build/mla-m06e2a-tnd-layout/acceptance.json`、`cpu.xml`、`sanitized/report.json` |
| 最新生产闭环 M06e1c，`4e83d32` | 2,041 个去重用例通过；双 dtype 公开 stride 矩阵中适用的 7 例，其 metadata 复用和生成加 R=1/4 E2E 均达标 | `build/mla-m06e1c-bsnd-stride/acceptance.json`、`performance-acceptance.json`、`component-*/`、`public-*/` |

M06e2a 的 1,139 份归档证据哈希、其四个 CPU 组件/测试文件及归档时两份文档与该提交一致；最终 CPU、sanitizer、文档、质量检查和验收退出码均为 0。sanitizer 覆盖 TND/BSND 地址 1,375/1,408 组、分配/used 组合 20 组、调度 6,187 组（迁移、固定源无 split、源原生分派分别重放），不重复计入 pytest 数量。初轮编译器路径错误、沙箱 LeakSanitizer 失败和中断轮次保留在该目录的 `environment-initial.*`、`cpu-sandbox.*`、`cpu-interrupted.*`，未记为通过。

生产性能结论仅限上述适用调用段。大 stride/广播的组件退化、原生 FD 长 KV 对照退化，以及 profiler host 关联解析限制仍存在；原生 FD 的 3 例保留数据并单列，不能据此宣称所有组件或所有原生路径不劣化。

## 环境与已知限制

最近环境实测来自 `build/mla-m06e2a-tnd-layout/environment.json`：Python `/usr/local/python3.11.15/bin/python3` 3.11.15，torch `2.7.1+cpu`，torch_npu `2.7.1`，CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，bisheng clang 15.0.5，g++ 11.4，8 张 Ascend910B3。本轮未重新读取设备健康/占用，不能把归档中卡的占用情况当作当前状态。

归档实际加载扩展为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 为 `d3eefbee09f643dc6fe6db2c89d0055db218a3d3b30f47b9a9422b8cedd9eea1`；本轮仅复核磁盘文件哈希一致，未重新加载或构建扩展。M06e2a 归档核验了 253 份生产源码依赖、30 个对象；TND CPU 地址组件不在生产构建清单。

CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f` 干净；只读 CANN 源 `ca18aee610729ca5270a44e2d84c751bc687fbf0` 的已有差异与本仓未跟踪 `kernel_meta/` 均保留。后续构建使用已核验的 CATLASS 并设置 `FLASH_ATTN_SKIP_SUBMODULE_INIT=1`。沙箱内 NPU 型号与 LeakSanitizer 进程检查受限；下一次实机验证须重新核查环境、占用和扩展加载来源。固定官方网页本轮未能在线获取，接口核对使用已归档的 `d15f1531...` 源码快照与版本记录。

## 唯一下一步与复现

下一步仅为 **[M06e2b](plan.md#下一步m06e2b)**；TND/混合 stride 的生产接入留在 M06e2c。本轮提交后停止。

本轮文档检查可在仓库根目录复现，脚本与报告保留在证据目录：

```bash
python3 build/mla-docs-audit-20261010/check_docs.py
git diff --check
```

M06e2a CPU 验收复现（另设输出目录，保留原证据）：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
MLA_TND_LAYOUT_DIR=build/mla-m06e2a-tnd-layout/recheck-tnd \
MLA_BSND_LAYOUT_DIR=build/mla-m06e2a-tnd-layout/recheck-bsnd \
MLA_USED_SCHEDULE_ARTIFACT_DIR=build/mla-m06e2a-tnd-layout/recheck-used \
python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_tnd_layout.py \
  tests/test_flash_attn_npu_v3_mla_bsnd_layout.py \
  tests/test_flash_attn_npu_v3_mla_used_schedule.py
```

sanitizer 的批量重放命令、输入和构建哈希见同目录 `sanitizer-command.json`、`sanitized/`；逐用例方式可在 CPU 命令前增加 `MLA_TND_LAYOUT_SANITIZE=1 MLA_BSND_LAYOUT_SANITIZE=1 MLA_SCHEDULE_SANITIZE=1`，须允许 LeakSanitizer 进程检查。生产最小精度复现见 `build/mla-m06e1c-bsnd-stride/production-reviewed-command.json`，完整连续/普通 FA3 回归见同目录 `continuous-command.json`、`ordinary-regressions-command.json`；执行前重新选空闲卡及新输出目录，不能直接沿用归档的设备占用假设。
