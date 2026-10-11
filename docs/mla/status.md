# MLA 当前状态

核对日期：2026-10-11；分支：`feat/mla-cann-migration`；接手基线：`139b992`。本轮完成 **M07d2：BSND Q 的独立 PA attention**，3,122 项去重 pytest 通过，完成双 dtype 固定源 PA/non-FD 性能采样。最近生产接入仍为 M06e2c2；唯一下一步为 [M07d3：TND Q 的独立 PA attention](plan.md#下一步m07d3)。

本页回答“现在能用什么、证据在哪里、下一步是什么”。接口、协议和验收方法查 [design](design.md)，步骤与完成条件查 [plan](plan.md)。本轮闭环提交用 `git log -1 -- docs/mla/status.md` 定位。

## 当前可调用能力

共同边界：**910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、无梯度、无 mask、`num_splits=1`**。Q/K 末维为 64，qv/V 为 512，四路同 dtype、同 NPU 设备。BSND 单样本 KV 容量及 TND 的 `max_seqlen_k` 为 0..16384；TND 总 Q/KV 各不超过 `INT32_MAX`，总 KV 可超过 16384。其他字段和资源界限见 [design 6.2–6.3](design.md#62-调度字段与边界)。

| 层次 | 已接通能力 | 记录与边界 |
| --- | --- | --- |
| 公开前向 | `flash_attn_func`，BSND Q/KV | 整容量，内部 v1 |
| 公开前向 | `flash_attn_varlen_func`，TND Q/KV | 两轴分配前缀；无 used 时内部 v3，任一 `seqused_q/k` 存在时内部 v5 |
| 公开只读 cache | `flash_attn_with_kvcache`，BSND Q + BSND KV | 整容量或实际 KV 长度；无长度且无记录走 v1，否则 v2；支持显式 v2 复用 |
| 公开只读 cache | `flash_attn_with_kvcache`，TND Q + BSND KV | Q 使用完整分配段；内部生成或显式复用 v4 |
| 公开 metadata | `get_scheduler_metadata` | 仅 v2（BSND）和 v4（混合）；生成器须接收 KV 长度 tensor |
| 内部前向 | BSND、TND、混合布局的 Q-used | 内部 v5；BSND/混合 Q-used 无公开等价入口，不开放显式 v5 复用 |
| 独立 PA 组件 | CPU 地址/调度、v6 CPU/AICPU 生成、CPU/910B 消费者、910B PA Copy Tile | 公开 **PA 仍明确拒绝**；组件与公开接入分别验收 |
| 独立 PA attention | BSND Q，含分配与内部 Q-used；K/V 独立 stride | 仅独立 probe；TND PA attention、公开 metadata 和前向尚未接通 |

三种非分页布局均支持四路独立的末维连续、外层非负 stride，包括 BNSD/NTD 归一视图、切片、广播、重叠及 offset；发射前检查完整 storage。显式 v2/v4 可跨合法视图复用，仍须匹配当次长度和 v4 Q 前缀。完整入口分派见 [design 6.7.1](design.md#671-非分页入口与记录选择)。

调用时最容易混淆的约束：

- kvcache/metadata 的 `num_splits` 默认 0，MLA 须显式传 **1**；metadata 的 `qkv_dtype` 默认 BF16，须与前向输入一致。func/varlen 默认 `num_splits=1`。
- 仅接受默认 mask/调度选项和 `softcap=0`；生成器使用 `has_softcap=False`，拒绝数值 `softcap/softmax_scale` 兼容关键字。前向 scale 默认 `1/sqrt(576)`，显式值须在 FP32 有限范围；完整限制见 [design 6.7.2](design.md#672-校验输出与发射)。
- 长度 tensor 须同设备、连续、int32；kvcache 另接受 None/整数。动态值由设备检查，无长度 D2H；空输出也不绕过记录校验，损坏或过期记录由消费者 Trap 拒绝。
- 请求 LSE 时 func/varlen 返回二项，kvcache 返回四项。非零容量中的零 KV 有效行为 `O=0、LSE=-inf`；全局空 KV 初始化分配区为 `O=0、LSE=+inf`；其他未使用 query 槽位不承诺值。shape、accumulator 与生命周期见 [design 6.7.2](design.md#672-校验输出与发射) 和 [5.8](design.md#58-stream-与内存生命周期)。

计划内尚未开放：PA、causal/window、自动或多 split、192/128 prefill、非零 softcap 与非默认调度选项。fake/compile 已有覆盖，完整 graph replay 留在 M16；910C 尚未开放或实机验收。KV 追加、内置 rotary、量化、backward 等排除项见 [design 第 4 节](design.md#4-功能边界)。

## 本轮闭环与证据

新增 [MlaPagedBsnd](../../csrc/ascend910/flash_attn_npu_3/mla/kernel/mla_paged.hpp)，把 v6 消费、PA Copy 和既有 attention 任务环接通。`MlaTaskRing` 保留非分页的 `MlaNonPaged` 别名；Cube 通过编译期 PA 分支读取真实 cache，复用既有 Vector、缓冲和事件顺序。每核在 TPipe、同步及输出/workspace 写入前检查当次长度、cache 描述和全部有效页。独立 PA 空 KV 有效行使用 `O=0、LSE=-inf`，包含合法空表/池；公共入口仍不发射 PA。边界见 [design](design.md#独立-bsnd-pa-attention)。

证据保存在忽略目录 `build/mla-m07d2-paged-attention/`，以下路径相对此目录。保留命令、日志、XML、源码快照、固定源、实际加载库、输入/输出/reference tensor 和性能原始数据；闭环汇总由 `check_acceptance.py` 生成 `acceptance.json`。这些本机证据不随 Git 分发，迁机须复制该目录。

| 核验 | 结果与依据 |
| --- | --- |
| 前置闭环 | 编辑前核验 M07d1 的 1,401 项历史证据、差异及适用质量检查通过；不计作本轮新测试。`prerequisite.json` |
| 本轮功能 | **3,122 项去重 pytest 通过，无失败/跳过**：新 PA attention 740、CPU 地址/消费者 596、设备 v6 消费者 274、PA Copy 300、非分页独立 BSND 82 与 TND/混合 232、普通 FA3/入口 822、非分页生产 76；九套 `*.xml`、日志与 `*-command.json` |
| 全页大小 O/LSE | 双 dtype、全部 64 种 P，各覆盖连续及 K/V 单独或同时 padding。非空有效行 O/LSE 与未修改的固定源 PA/non-FD Kernel 逐元素一致，同时通过 FP64 与低精度 CPU reference。数学侧直接按原始页表逐 token 索引，不读取 metadata；`probes/*/accuracy/*.pt`、`release/*/accuracy/*.pt` 与 `accuracy.jsonl` |
| 任务环与地址 | 覆盖 KV=16384、多 M/S2 块、端点外零尾部、全核发射、Q-used/poison、QK 两半、scale=0/负值、online max 变化、65535/65536/65537、广播/重叠、offset；真实 >uint32 的 Q/qv 地址、K/V 页与行地址、页表行距均通过。宽地址实际分配约 8–16 GiB；`device.xml`、`device-rest.xml` |
| 空值与拒绝 | 空 query、零 KV、空表/池、小行数以及 stale 长度、损坏记录/页号、完整 storage 拒绝通过；零 query/used 不跳过有效页检查。双 dtype **18 次 release 真实 Trap**；诊断模式核对各核状态及 O/LSE/workspace 哨兵未改写。`release/*/error-*/runtime-error.json` |
| stream 与复用 | 同流更新长度并重算、metadata 跨 cache/stride/scale 复用、两条 worker stream 与 allocator 回收通过；探针发射记录全部异步输入/workspace 的 stream 生命周期。对应 XML 与源码测试 |
| 最终构建 | 自动格式化前首轮 714 项通过；编译输入的 C++ token / Python AST 等价证据和原始库保留在 `pre-format-sources/`、`probes/`、`format-equivalence.json`。格式化后重新构建 `release/`，196 项补充测试通过；固定性能矩阵使用该最终库，每例先重验源/双数学精度。每种 dtype 核验 951 个设备依赖文件，源头文件来自固定 Git blob |
| 组件性能 | 双 dtype 各 12 例完成；迁移/源 Kernel 中位比为 FP16 **1.0114–1.1386**、BF16 **1.0097–1.1515**；wall 比为 **1.0069–1.0476**、**1.0043–1.0324**。逐例结果见 `performance/comparison.json`；固化矩阵/分派、双 dtype `wall.json` 与原始 profiler CSV 均保留 |
| 受影响生产回归 | 共享 Kernel/Block 的改动已重编生产 10 个 MLA 实例；三种非分页布局的源/数学与生命周期、普通 FA3 metadata/func/varlen/cache/FD/compile 均通过。未重跑普通 FA3 全套、旧消费者 release Trap、旧耗时或公开 MLA E2E；历史结果不能当作本轮新运行 |
| 提交质量 | 文档链接/锚点、表格/命令语法、源码/对象/加载库哈希、差异与适用质量钩子通过；仅本步 12 个文件进入提交。`docs-check.json`、`quality-staged.log` |

**性能口径：** 双 dtype 各 12 例，P=16/112/128/256/528/1024，短/长 KV、单/多 Q、乱序页、尾页、独立 K/V padding 和零长度；采样前改用足量物理页，长序列不折叠到小页池，并由采样器重验新输入精度。每例 5 次预热，五轮各 20 对 wall、10 对设备样本。两边使用相同逻辑输入、M/S2 块、核数和端点；迁移计时包含 v6/当次长度/全部有效页检查，生成 metadata、分配/上传和源侧视图物化在计时外。K/V 原生页间距保留，其他源 ABI 无法表示的 Q/cache 行/页表 stride 先独立物化，不能宣称原生 FIA host 接受所有视图。

两种 dtype 各 12 例均慢于固定源，不能宣称 Kernel 逐例不劣化。独立 probe wall 含 ctypes、发射与同步，**不是公开 PA E2E**；每种 dtype 各 6 个原生 FD 用例单列，FD 能力仍留在 M10。公开 metadata E2E 门槛和阶段边界沿用 [design 7](design.md#7-验收口径)。

两种 dtype 的 profiler 仍报告 `fwk_api_db` / host flow 解析错误，保留在 `profile.log`；采样器逐条核对 `kernel_details.csv` 与原始 `op_summary_*.csv` 的名称、时间、设备和数量，只采用已核对的设备耗时，不声称 host 关联解析通过。

开发期失败如实保留：`initial-regression-producer-build-failure/` 为两个 AICPU 构建线程共享 stdout 重定向引发的异常，已改串行重建成功；`initial-nonpaged-regression-failure/` 为并发 NPU 回归时的 AICPU `get kernel failed`。隔离的 2 项小测及随后串行的 314 项独立非分页回归均通过，未复现该加载错误；早期失败不计入通过数量。

## 最近功能闭环与限制

以下是历史完整验收，本轮实际重跑范围见上表；不能用新子集代替历史全套。路径相对仓库根目录。

| 功能闭环 | 结果与证据 |
| --- | --- |
| M07d1 PA Copy，`139b992` | `build/mla-m07d1-paged-copy/acceptance.json`：1,401 项去重 pytest，全部页大小的双 dtype Copy、64 位地址与同步；组件 Kernel 中位比 FP16 0.9985–1.0036、BF16 0.9949–1.0041 |
| M07c3 独立 v6 消费者，`9f9aeb9` | `build/mla-m07c3-paged-consumer/acceptance.json`：2,273 项去重 pytest；CPU/910B 各 2,552 组源差分，58 次 release Trap，2,622,720 次 sanitizer 调用 |
| 最近生产接入 M06e2c2，`538b8e5` | `build/mla-m06e2c2-mixed-stride-production/acceptance.json`：4,659 项去重 pytest，完成三种非分页布局 stride 接入 |
| M06e2c2 公开性能 | 同目录 `performance-acceptance.json`：双 dtype 各 8 个适用公开用例，R=1/4 的复用、生成加前向、内部生成共 96 段逐例达标；FA3/CANN E2E 为 FP16 0.284–0.623、BF16 0.284–0.675。各 3 个原生 FD 用例留待 M10 |

**仍保留的限制：** M06e2c2 组件 Kernel 比值为 FP16 0.898–1.888、BF16 0.901–2.458，两种 dtype 均有 6 例中的 4 例慢于固定源。当前 PA attention 仅覆盖独立 BSND Q；TND PA attention、公开 PA metadata/前向与公开 PA E2E 尚待验收。910C 和完整 graph replay 未作本轮实机验收。

## 环境与构建

本轮使用 Python `/usr/local/python3.11.15/bin/python3` 3.11.15，torch `2.7.1+cpu`、torch_npu `2.7.1`，CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，物理卡 6 的 Ascend910B3（进程内设备 0）。8 张卡在接手时健康，卡 6 无运行进程；采样前后健康/占用分别保存在 `performance/{fp16,bf16}/npu-before.log` 与 `npu-after.log`。

CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f` 且干净；只读 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0` 的原有 67 项差异、本仓原有未跟踪 `kernel_meta/` 均保留。详见 `environment.json`、`*-status-before.txt` 和 `*-status-final.txt`。

生产扩展实际加载路径为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，本轮重建并安装后的 SHA256 为 `a34d76dee5d754ec34e9f7a100c81f7226ce23db1aef6f4a0f077a926dfc9a63`；旧 `0e8f3c6e...` 库保存在 `baseline-production.so`。`production-load.json` 核对实际加载库、链接库、32 个对象、260 个仓库源文件和 6,186 个依赖；对象与库保存在 `production-final/`。生产包含共用头文件但仍只实例化非分页 Kernel，不能据重编产物推断公开 PA 已开放。独立 release 库的来源与实际加载哈希见 `release/*/{build-manifest,loaded,oracle-source-manifest}.json`。

继续使用非 login shell；CPU pytest 设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，NPU 在沙箱外执行，质量工具使用 `/tmp/mla-m01-precommit` 及其 cache。设备任务串行执行；版本/tiling 注册警告保留在日志中。下次开发/实机验证前重新检查设备和加载产物，不将本轮卡号视为持续可用保证。

## 唯一下一步与复现

下一步仅为 **[M07d3](plan.md#下一步m07d3)**：TND Q 的独立 PA attention，保留 Q 分配前缀起点与内部 Q-used、KV 非累计长度、源数值/事件/任务环，完成双 dtype 源/数学精度与同调度非 FD 性能。公开 PA metadata 和前向分别留在 M07e1/M07e2–e3；本轮提交后停止。

本轮最小闭环复核（仓库根目录，非 login shell，不发射 NPU）：

```bash
python3 build/mla-m07d2-paged-attention/check_acceptance.py
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache \
bash ci/code_quality.sh --staged
git diff --check
git diff --cached --check
```

完整构建和功能命令见本步 `run_checks.py`、`run_remaining.py` 与 `*-command.json`：`device`、`device-rest`、CPU/消费者/Copy/三种非分页/普通 FA3 回归和 `profile`。`probes/` 保存格式化前库，当前源复用须使用 `release/` 并设置 `MLA_PAGED_ATTENTION_REUSE=1`；fixture 会校验源码和依赖哈希。全新复现设置新的 `MLA_PAGED_ATTENTION_DIR`、`MLA_PAGED_METADATA_DEVICE_DIR`、`MLA_PAGED_SCHEDULE_DIR`，不启用 reuse，串行构建/执行，保留既有证据。源码、环境或构建变化后重跑受影响检查，历史证据核验不能替代新运行。
