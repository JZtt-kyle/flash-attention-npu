# MLA 当前状态

本轮完成 **M07b2：独立 PA CPU 调度**，基线 `402e154`，分支 `feat/mla-cann-migration`，日期 2026-10-11。M06 已完成，M07 完成到独立 CPU 地址与调度；**PA 仍明确拒绝**。唯一下一步是 **[M07c1：PA 协议与 CPU 序列化](plan.md#下一步m07c1)**。

本页只维护当前能力、最近证据、环境差异和下一步。接口、协议与架构查 [design](design.md)，步骤与完成条件查 [plan](plan.md)；当前文档闭环提交用 `git log -1 -- docs/mla/status.md` 定位。

## 当前可调用能力

共同边界为 **910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、无梯度、无 mask、`num_splits=1`**。Q/K 末维为 64，qv/V 为 512；四路输入同 dtype、同 NPU 设备。BSND 每样本 KV 容量与 TND 的 `max_seqlen_k` 均为 0..16384；TND 总 Q/KV 各不超过 `INT32_MAX`，总 KV 可以超过 16384。其他字段及资源上限见 [design 6.2–6.3](design.md#62-调度字段与边界)。

| 能力 | 当前接通范围 |
| --- | --- |
| BSND Q/KV | 公开 `flash_attn_func`；只读 `flash_attn_with_kvcache` 支持整容量或实际 KV 长度；内部 Q-used 走 v5 |
| TND Q/KV | 公开 varlen 支持两轴分配前缀及可选 `seqused_q/k`，每次内部生成 v3/v5，无显式 metadata 复用 |
| TND Q + BSND KV | 公开只读 kvcache 使用完整 Q 分配段，支持 v4 生成/复用；内部混合 Q-used 走 v5 |
| 三种布局的 stride | 均已生产接入；四路末维连续、外层非负 stride，含 BNSD/NTD、切片、广播、重叠和独立 offset；发射前校验完整 storage |
| 显式 metadata | 公开生成/复用仅 v2（BSND）和 v4（混合）；可跨合法视图复用，仍须匹配当次长度及 v4 的 Q 前缀 |
| PA | 独立 CPU 地址、有效页校验、页内分段及 BSND/TND Q 调度（含 Q-used）；未接入 metadata、设备 Copy、attention 或公开入口 |

入口参数与 v1–v5 选择统一见 [design 6.7.1](design.md#671-非分页入口与记录选择)，stride 与 PA CPU 组件见 [design 6.4](design.md#64-输入-stride-与地址校验)。调用时还须注意：

- **默认值：** func/varlen/内部前向的 `num_splits` 默认 1；kvcache/metadata 默认 0，MLA 调用须显式传 1。只接受 `causal=False`、`window_size=(-1,-1)`、`softcap=0`、`pack_gqa=None`、`sm_margin=0`、`attention_chunk=0` 和 `rotary_interleaved=True`。metadata 的 `has_softcap=False`，不接受数值 `softcap/softmax_scale` 兼容关键字。
- **长度与 scale：** 长度 tensor 必须同设备、连续、int32，值在设备检查，无 D2H。kvcache 另接受 None/整数长度；生成器必须接收 tensor。前向 scale 默认 `1/sqrt(576)`，显式值须在 FP32 有限范围，metadata 不保存 scale。
- **输出与空值：** O 同输入 dtype，LSE 为 FP32；请求 LSE 时 func/varlen 返回二项，kvcache 返回四项。非零容量中的零 KV 有效行 `O=0、LSE=-inf`；全局空 KV 初始化整个分配区为 `O=0、LSE=+inf`；其他未使用 query 槽位不承诺输出值。空输出仍校验记录和长度，损坏或过期记录由消费者 Trap 拒绝。shape、空 accumulator 和生命周期见 [design 6.7.2](design.md#672-校验输出与发射)、[5.7](design.md#57-stream-与内存生命周期)。

**计划内尚未开放：** PA、causal/window、自动或多 split、192/128 prefill、非零 softcap 与非默认调度选项。fake/compile 已有覆盖，完整 graph replay 留在 M16；910C 尚未开放或实机验收。KV 追加、内置 rotary、量化与 backward 等范围外组合也明确拒绝，完整范围见 [design 第 4 节](design.md#4-功能边界)。

## 本轮核验与证据

新增 `mla_paged_schedule.h` 的独立解析/调度接口，实现在 `mla_tnd_schedule.cpp` 中复用既有长度归一、SplitCore、无 split 端点与资源预算。逻辑容量 C 与源常规 PA 的页对齐 S2 分开保存，保留 Q 分配起点与 used；没有生产调用方，未修改 metadata、Kernel 或公开入口。

本轮证据位于忽略目录 `build/mla-m07b2-paged-schedule/`，下表未注明目录的文件均指此处；上一步核验副本为 `build/mla-m07b2-prerequisite/`，未覆盖旧证据。路径相对仓库根目录，不随 Git 分发。保存了固定源快照、编译命令、完整输入/输出、XML、sanitizer 重放及哈希清单，汇总为 `acceptance.json`。

| 核验项 | 结论与证据 |
| --- | --- |
| 上一步交接 | 开发前复跑 `402e154` 的最小文档核验；202 个链接、29 个表格、25 份固定源、5,867 份历史证据、534 个其余受跟踪文件与扩展哈希通过。`prerequisite-command.json` / `prerequisite.log`；历史功能结果不算本轮重跑 |
| PA CPU 与旧路径 | **1,131 个去重 pytest 通过，无失败或跳过**：PA 420、dense 93、BSND 92、TND 111、混合 109、used 306。`cpu.xml` / `cpu-command.json` 为 1,130 项；`empty-values.xml` 补 1 项；`capture.xml` 完整复跑最终 PA 420 项并采集全部驱动输入/输出 |
| 固定源字段差分 | 7,897 组，原生非 FD 3,196、FA3 无 split 映射 4,701；全部 64 种 P、1..26 AIC、C1V1/C1V2、BSND/TND Q、可选 Q-used、非均匀与零尾部。完整端点、FD 候选及 workspace 与固定源一致，独立标量任务覆盖/成本一致；`capture/paged/*-inputs.json`、`*-outputs.json` 与 `source-manifest.json` |
| 容量与边界 | P=112 的 448 基块、P=528 的 512 基块及跨页范围；C≤16384 且为 P 的整数倍，C 与源 S2 分离，KV 非累计；空分配/全零 KV、Q-used 为零仍验 KV、非法前缀/计数/页大小、>uint32 Q 起点、执行块/字段/workspace 溢出和失败不改输出。范围为 CPU 算术及调度，无 PA 设备结论 |
| Sanitizer | **ASan/UBSan/LeakSanitizer 批量重放通过**：迁移调度 8,047、迁移解析 98、固定源映射 7,897、固定源原生 7,897、迁移 SplitCore 原生 7,897，共 31,836 条；5 个进程退出 0、stderr 为空，输出 SHA256 与普通运行一致。`sanitized-replay/report.json`、`manifest.json`、完整 stdin/expected/actual；命令为 `sanitizer-replay-command.json` |
| 提交质量与来源 | 本步差异、文档链接/表格/步骤、源 blob 与未改文件、实际加载扩展及 CATLASS 核对；适用质量钩子与 `git diff --check`。`check_acceptance.py`、`checks.json`、`quality-staged.log`；仅提交本步 9 个文件 |

逐用例 sanitizer 在 10 项通过后因每进程 LeakSanitizer 扫描耗时主动中断，退出 2，**不计为闭环通过**；`sanitized.log`、`sanitized-command.json`、`sanitized-interruption.json` 保留。随后完整采集最终 420 项的全部驱动调用并批量重放，以上最终报告才是 sanitizer 闭环依据。首次质量钩子自动格式化后已复核；没有绕过钩子。

## 最近功能闭环与限制

以下是**此前闭环，本轮仅在前置检查中复核证据和来源**，不作为本轮新增测试结果。

| 闭环 | 结果与可复现证据 |
| --- | --- |
| M07b1 CPU 地址，`ccc5fe5` | `build/mla-m07b1-paged-layout/acceptance.json`：354 个去重 pytest 通过，无失败或跳过（PA 171、BSND 82、TND 101）；`cpu-command.json` / `cpu.xml` 保存完整命令及结果 |
| M07b1 sanitizer | 同目录 `sanitized-replay/report.json`：ASan/UBSan/LeakSanitizer 批量重放迁移驱动 2,953、固定源驱动 2,270 条输入，输出哈希相同，两进程退出 0、stderr 为空；复现脚本为 `replay_sanitized.py` |
| 最近生产接入：M06e2c2，`538b8e5` | `build/mla-m06e2c2-mixed-stride-production/acceptance.json`：4,659 个去重 pytest 通过，无失败或跳过；生产 448、受影响回归 2,684、普通 FA3 833、TND/独立 stride 694 |
| M06e2c2 公开性能 | 同目录 `performance-acceptance.json`：双 dtype 各 8 个适用公开用例，R=1/4 的复用、生成加前向、内部生成共 96 段逐例达标；FA3/CANN E2E 为 FP16 0.284–0.623、BF16 0.284–0.675。各 3 个原生 FD 用例单列，待 M10 |

**保留的限制：** M06e2c2 组件 Kernel 比值为 FP16 0.898–1.888、BF16 0.901–2.458，两种 dtype 均有 6 例中的 4 例慢于固定源，不能宣称 Kernel 全部不劣化。组件 profiler 的 `fwk_api_db` 解析错误保留，设备 CSV 已核验，不据此声称 host 关联可用。BSND/TND stride 的公开 E2E 证据分别在 `build/mla-m06e1c-bsnd-stride/`、`build/mla-m06e2c1-tnd-stride-production/`。

M07b1 的 CPU Copy trace 只证明地址和分段，不能代替 PA 设备/attention/E2E；>uint32 cache 地址只作算术检查。M07b2 的全零 KV 单校验核/零 workspace 也不定义 PA 空行 O/LSE；空表/空池及设备行为继续由后续步骤验收。

## 环境与构建

本轮实际导入核验：Python `/usr/local/python3.11.15/bin/python3` 3.11.15、torch `2.7.1+cpu`、torch_npu `2.7.1`，CANN `/usr/local/Ascend/cann-9.0.0-beta.2`；CATLASS 固定为 `769cd40a8716b28650b6bebb08db4834eea4462f` 且干净。`environment.json` 与 `npu-smi.log` 确认 8 张 Ascend910B3 健康，查询时 1–7 卡无运行进程，0 卡已有占用；本步未运行 NPU 功能、精度或性能测试。

实际加载扩展 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so` 的 SHA256 仍为 `38582e6e4deb2eed122a38e5c1da06cd6d5386b4d00bc405f21e1cf469a4a00e`，与 M07b1 加载及 M06e2c2 构建记录一致。本步只重编独立 CPU 驱动，**未重建生产扩展，该 `.so` 不包含新增 PA 调度**。共享源码中的新 PA 入口没有生产调用方；旧 CPU 调度以本轮重编结果回归。

初次沙箱内扩展导入因驱动设备查询失败退出 1，`npu-smi` 因 DCMI 初始化失败退出 187；沙箱外只读查询成功，见 `environment-sandbox-failures.json` 与最终环境记录。CPU pytest 设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`；sanitizer 在允许 LeakSanitizer ptrace 的环境执行，未关闭 leak 检查。

只读 CANN 源仍为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，原有 67 项差异与本仓未跟踪 `kernel_meta/` 保留。使用非 login shell；质量检查沿用 `/tmp/mla-m01-precommit` 与 `/tmp/mla-m01-precommit-cache`，未修改系统安装。

## 唯一下一步与复现

下一步仅为 **[M07c1](plan.md#下一步m07c1)**：新 PA 协议与 CPU 序列化；固化版本、字数、调用 context 和长度快照，明确逻辑 C 与源 S2，独立解码/损坏/失败清零及旧协议回归；本轮提交后停止。

本轮最小闭环复核（依赖本机证据，不重新导入扩展或发射 NPU）：

```bash
python3 build/mla-m07b2-paged-schedule/check_acceptance.py
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache \
bash ci/code_quality.sh --staged
git diff --check
git diff --cached --check
```

重跑功能验证用本轮 `cpu-command.json` 的六套 CPU pytest 命令；最终 PA 套件为 420 项。编译命令和固定 Git blob 在对应驱动目录，sanitizer 完整重放命令见 `sanitizer-replay-command.json` 与 `replay_sanitized.py`。后续修改使用新的证据目录，避免覆盖旧结果；先核对源码/二进制/输入哈希再复现。生产/性能证据仍见 M06e2c2，不以本轮 CPU 结果代替设备验收。
