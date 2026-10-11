# MLA 当前状态

核对日期：2026-10-11；分支：`feat/mla-cann-migration`；接手基线：`82d3809`。本轮完成 **M07d3：TND Q 的独立 PA attention**，3,832 项去重 pytest 通过，完成双 dtype 固定源 PA/non-FD 性能采样。最近生产接入仍为 M06e2c2；唯一下一步为 [M07e1：公开 PA metadata](plan.md#下一步m07e1)。

本页回答“现在能用什么、证据在哪里、下一步是什么”。接口、协议和验收方法查 [design](design.md)，步骤与完成条件查 [plan](plan.md)。本轮闭环提交用 `git log -1 -- docs/mla/status.md` 定位。

## 当前可调用能力

共同边界：**910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、无梯度、无 mask、`num_splits=1`**。Q/K 末维为 64，qv/V 为 512，四路同 dtype、同 NPU 设备。BSND 单样本 KV 容量及 TND 的 `max_seqlen_k` 为 0..16384；非分页 TND 总 Q/KV 各不超过 `INT32_MAX`，总 KV 可超过 16384。其他字段和资源界限见 [design 6.2–6.3](design.md#62-调度字段与边界)。

| 层次 | 已接通能力 | 记录与边界 |
| --- | --- | --- |
| 公开前向 | `flash_attn_func`，BSND Q/KV | 整容量，内部 v1 |
| 公开前向 | `flash_attn_varlen_func`，TND Q/KV | 两轴分配前缀；无 used 时内部 v3，任一 `seqused_q/k` 存在时内部 v5 |
| 公开只读 cache | `flash_attn_with_kvcache`，BSND Q + BSND KV | 整容量或实际 KV 长度；无长度且无记录走 v1，否则 v2；支持显式 v2 复用 |
| 公开只读 cache | `flash_attn_with_kvcache`，TND Q + BSND KV | Q 使用完整分配段；内部生成或显式复用 v4 |
| 公开 metadata | `get_scheduler_metadata` | 仅 v2（BSND）和 v4（混合）；生成器须接收 KV 长度 tensor |
| 内部前向 | BSND、TND、混合布局的 Q-used | 内部 v5；BSND/混合 Q-used 无公开等价入口，不开放显式 v5 复用 |
| 独立 PA 组件 | CPU 地址/调度、v6 CPU/AICPU 生成、CPU/910B 消费者、910B PA Copy Tile | 公开 **PA 仍明确拒绝**；组件与公开接入分别验收 |
| 独立 PA attention | BSND/TND Q，含分配与内部 Q-used；Q/qv 与 K/V 独立 stride | 仅独立 probe；公开 PA metadata 和前向尚未接通 |

三种非分页布局均支持四路独立的末维连续、外层非负 stride，包括 BNSD/NTD 归一视图、切片、广播、重叠及 offset；发射前检查完整 storage。显式 v2/v4 可跨合法视图复用，仍须匹配当次长度和 v4 Q 前缀。完整入口分派见 [design 6.7.1](design.md#671-非分页入口与记录选择)。

调用时最容易混淆的约束：

- kvcache/metadata 的 `num_splits` 默认 0，MLA 须显式传 **1**；metadata 的 `qkv_dtype` 默认 BF16，须与前向输入一致。func/varlen 默认 `num_splits=1`。
- 仅接受默认 mask/调度选项和 `softcap=0`；生成器使用 `has_softcap=False`，拒绝数值 `softcap/softmax_scale` 兼容关键字。前向 scale 默认 `1/sqrt(576)`，显式值须在 FP32 有限范围；完整限制见 [design 6.7.2](design.md#672-校验输出与发射)。
- 长度 tensor 须同设备、连续、int32；kvcache 另接受 None/整数。动态值由设备检查，无长度 D2H；空输出也不绕过记录校验，损坏或过期记录由消费者 Trap 拒绝。
- 请求 LSE 时 func/varlen 返回二项，kvcache 返回四项。非零容量中的零 KV 有效行为 `O=0、LSE=-inf`；全局空 KV 初始化分配区为 `O=0、LSE=+inf`；其他未使用 query 槽位不承诺值。shape、accumulator 与生命周期见 [design 6.7.2](design.md#672-校验输出与发射) 和 [5.8](design.md#58-stream-与内存生命周期)。

计划内尚未开放：PA、causal/window、自动或多 split、192/128 prefill、非零 softcap 与非默认调度选项。fake/compile 已有覆盖，完整 graph replay 留在 M16；910C 尚未开放或实机验收。KV 追加、内置 rotary、量化、backward 等排除项见 [design 第 4 节](design.md#4-功能边界)。

## 本轮闭环与证据

新增 [`MlaPagedTnd`](../../csrc/ascend910/flash_attn_npu_3/mla/kernel/mla_paged.hpp)，与 `MlaPagedBsnd` 共用 PA 包装层、Copy 和既有 attention 任务环。TND Q/qv 从 v6 读取分配起点，按各自 token/head stride 寻址；Q-used 不移动后续样本，KV 长度保持非累计。O/LSE 使用 `[Tq,Hq,512]` / `[Hq,Tq]`；每核在 `TPipe`、同步和输出/workspace 写入前完成 v6、当前长度及所有有效页检查。公共入口仍不发射 PA，边界见 [design](design.md#独立-bsndtnd-pa-attention)。

证据保存在忽略目录 `build/mla-m07d3-tnd-paged-attention/`，以下路径相对此目录。保留命令、日志、XML、源码快照、固定源、实际加载库、输入/O/LSE/reference tensor 和性能原始数据；闭环汇总由 `check_acceptance.py` 生成 `acceptance.json`。这些本机证据不随 Git 分发，迁机须复制目录。

| 核验 | 结果与依据 |
| --- | --- |
| 前置闭环 | 编辑前核验 M07d2 的 3,122 项历史证据、差异及最小质量检查通过；不计作本轮新测试。`prerequisite.json` |
| 本轮功能 | **3,832 项去重 pytest 通过，无失败/跳过**：TND PA 752、BSND PA 698、CPU 地址/消费者 596、设备消费者 274、PA Copy 300、非分页 BSND 82 与 TND/混合 232、普通 FA3/入口 822、非分页生产 76；九套 XML、日志与 `*-command.json` |
| 全页大小 O/LSE | 双 dtype、全部 64 种 P，各覆盖连续及 K/V 单独或同时 padding。非空有效行 O/LSE 与未修改的固定源 PA/non-FD Kernel 逐元素一致，同时通过 FP64/低精度数学 reference。源 Q 独立压紧有效段，数学侧按原始分配/used 和页表逐 token 索引，不读取 metadata。`probes/*/{accuracy.jsonl,accuracy/*.pt}` |
| 任务环与地址 | 覆盖非均匀分配/used、零 Q/KV、端点外零尾部、KV=16384、多 M/S2 块、全核发射、QK 两半、scale=0/负值、online max 变化、65535/65536/65537、NTD/广播/重叠/offset；真实 >uint32 的 Q/qv token/head、K/V 页与行地址、页表行距通过。宽地址实际分配约 8–16 GiB；`device.xml` |
| 空值与拒绝 | 合法空 query/表/池通过，零 KV 有效行 `O=0、LSE=-inf`；空 Q/used 不跳过有效页检查，完整 Q storage 不按 used 缩短。损坏记录、stale 前缀/used/KV、非法页号和资源经每核诊断检查，O/LSE/workspace 哨兵未改写；双 dtype **20 次 release 真实 Trap**，`probes/*/tnd-error-*/runtime-error.json` |
| stream 与复用 | 同流更新分配前缀、Q-used/KV 长度并重算；metadata 跨 cache/stride/scale 复用；两条 worker stream、独立 workspace 与 allocator 回收通过。探针记录全部异步输入和 workspace 的 stream 生命周期 |
| 构建与加载 | 最终格式化后的双 dtype probe 重新构建后执行全套，每种核对 952 个设备依赖及固定 Git blob。非分页回归库重建；生产重编 10 个 MLA 实例，核对实际加载库、32 个对象、260 个仓库源文件和 6,186 个依赖。`probes/*/{build-manifest,loaded,oracle-source-manifest}.json`、`production-load.json`、`production-final/` |
| 组件性能 | 双 dtype 各 12 例完成；迁移/源 Kernel 中位比 FP16 **1.0085–1.1534**、BF16 **1.0052–1.1500**；wall 比分别 **0.9935–1.0287**、**1.0032–1.0235**。固定矩阵/分派、每 dtype 2,400 个 wall 与 1,200 个设备样本、原始 CSV 均保留，见 `performance/comparison.json` |
| 回归边界 | BSND PA 重跑全页大小及任务环/地址/空值等专项，未重跑其旧 12 例性能矩阵和 release Trap。非分页重跑源/数学及生产三布局生命周期，普通 FA3 包含 metadata/func/varlen/cache/FD/compile；未重跑普通 FA3 全套、旧消费者 Trap、旧耗时或公开 MLA E2E。历史证据不能算本轮新运行 |
| 提交质量 | 文档链接/锚点、表格/命令语法、源码/对象/加载库哈希、差异及适用质量钩子检查；仅本步 11 个文件提交。`docs-check.json`、`quality-staged.log`、`source-snapshots.json` |

**性能口径：** 双 dtype 各 12 例，P=16/112/128/256/528/1024，短/长 KV、单/多 Q、分配间隙、零 Q/KV、乱序/尾页和独立 K/V padding。矩阵与源分派在采样前固化，每例先复核源/双数学精度；5 次预热，五轮各 20 对 wall、10 对设备样本。两边使用相同逻辑输入、M/S2 块、核数和端点；迁移计时包含 v6/当前长度/全部有效页检查，metadata 生成、分配/上传及源侧物化/有效 Q 压紧在计时外。K/V 原生页间距保留，不能宣称原生 FIA host 接受所有视图。

两种 dtype 的全部 12 例 Kernel 均慢于固定源，不能宣称 Kernel 逐例不劣化。独立 probe wall 含 ctypes、发射和同步，**不是公开 PA E2E**；每种 dtype 各 6 例原生 FD 分派单列，FD 能力仍留在 M10。公开 metadata E2E 门槛沿用 [design 7](design.md#7-验收口径)。

两种 dtype 的 profiler 仍报告 `fwk_api_db` / host flow 解析错误，保留于 `profile.log`；采样器逐条核对 `kernel_details.csv` 与原始 `op_summary_*.csv` 的名称、时间、设备和数量，只采用已核对的设备耗时，不声称 host 关联解析通过。本轮功能测试及构建命令均成功；接手环境探测首次用了错误的 Python 接口模块名，修正后核对实际扩展成功，不计作测试通过项。

## 最近功能闭环与限制

以下是历史完整验收，本轮实际重跑范围见上表；不能用新子集代替历史全套。路径相对仓库根目录。

| 功能闭环 | 结果与证据 |
| --- | --- |
| M07d2 BSND PA attention，`82d3809` | `build/mla-m07d2-paged-attention/acceptance.json`：3,122 项去重 pytest；双 dtype 12 例 Kernel 中位比为 FP16 1.0114–1.1386、BF16 1.0097–1.1515，均慢于固定源 |
| M07d1 PA Copy，`139b992` | `build/mla-m07d1-paged-copy/acceptance.json`：1,401 项去重 pytest，全部页大小的双 dtype Copy、64 位地址与同步；组件 Kernel 中位比 FP16 0.9985–1.0036、BF16 0.9949–1.0041 |
| M07c3 独立 v6 消费者，`9f9aeb9` | `build/mla-m07c3-paged-consumer/acceptance.json`：2,273 项去重 pytest；CPU/910B 各 2,552 组源差分，58 次 release Trap，2,622,720 次 sanitizer 调用 |
| 最近生产接入 M06e2c2，`538b8e5` | `build/mla-m06e2c2-mixed-stride-production/acceptance.json`：4,659 项去重 pytest，完成三种非分页布局 stride 接入 |
| M06e2c2 公开性能 | 同目录 `performance-acceptance.json`：双 dtype 各 8 个适用公开用例，R=1/4 的复用、生成加前向、内部生成共 96 段逐例达标；FA3/CANN E2E 为 FP16 0.284–0.623、BF16 0.284–0.675。各 3 个原生 FD 用例留待 M10 |

**仍保留的限制：** M06e2c2 组件 Kernel 比值为 FP16 0.898–1.888、BF16 0.901–2.458，两种 dtype 均有 6 例中的 4 例慢于固定源。当前 PA attention 覆盖独立 BSND/TND Q；公开 PA metadata/前向与公开 PA E2E 尚待验收。910C 和完整 graph replay 未作本轮实机验收。

## 环境与构建

本轮 Python `/usr/local/python3.11.15/bin/python3` 3.11.15，torch `2.7.1+cpu`、torch_npu `2.7.1`，CANN `/usr/local/Ascend/cann-9.0.0-beta.2`；接手时 8 张 910B3 健康，物理卡 6 空闲，进程内设备 0。见 `environment.json` 与 `npu-before.log`；采样前后健康/占用记录见 `performance/{fp16,bf16}/npu-{before,after}.log`。

CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f` 且干净；只读 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0` 原有 67 项差异、本仓原有未跟踪 `kernel_meta/` 均保留。接手和提交前状态分别保存在 `*-status-before.txt` / `*-status-final.txt`。

生产扩展实际路径为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`。接手 SHA256 `a34d76dee5d754ec34e9f7a100c81f7226ce23db1aef6f4a0f077a926dfc9a63`；本轮构建文件 SHA256 `cd44a4c5caae26478a1a8407250bc2d60219018714111c15f2a36d985cf49404`，实际加载已由 `production-load.json` 核对。生产仍只实例化非分页 Kernel，不因重编开放公开 PA。独立 probe 每种 dtype 核对 952 个设备依赖，加载时验证当前源码/依赖及库哈希。

继续使用非 login shell；CPU pytest 设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，NPU 在沙箱外执行。质量工具使用 `/tmp/mla-m01-precommit` 及其 cache；设备任务串行，版本/tiling 注册警告保留在日志中。长 KV 的双数学 CPU reference 每例约 90 秒，不能误记为 Kernel 耗时。

## 唯一下一步与复现

下一步仅为 **[M07e1](plan.md#下一步m07e1)**：公开 `get_scheduler_metadata(page_size=P)` 的 BSND/TND Q 接入、真实/fake/C++/compile 一致性、长度更新/流生命周期和普通 FA3 回归。公开 PA 前向仍拒绝，分别留在 M07e2/M07e3；本轮提交后停止。

本轮最小闭环复核（仓库根目录，非 login shell，不发射 NPU）：

```bash
python3 build/mla-m07d3-tnd-paged-attention/check_acceptance.py
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache \
bash ci/code_quality.sh --staged
git diff --check
git diff --cached --check
```

构建和完整功能命令见本轮 `build_probes.py`、`build_regression.py`、`run_checks.py`、`run_remaining.py` 与 `*-command.json`。性能命令为 `python3 -m tests.mla_paged_attention_profile <probes> <producer/build-manifest.json> <新性能目录> tnd`。当前独立库复用设置 `MLA_PAGED_ATTENTION_DIR` 为本轮 `probes/`、`MLA_PAGED_ATTENTION_REUSE=1`；fixture 会核对当前源码与依赖哈希。全新复现换新 `MLA_PAGED_ATTENTION_DIR`、`MLA_PAGED_METADATA_DEVICE_DIR`、`MLA_PAGED_SCHEDULE_DIR`，不启用 reuse，串行构建/执行并保留既有证据。源码、环境或构建变化后重跑受影响检查，历史证据核验不能替代新运行。
