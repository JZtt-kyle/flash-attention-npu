# MLA 当前状态

本轮完成 **M07c1：PA 协议与 CPU 序列化**，基线 `8fb46d7`，分支 `feat/mla-cann-migration`，日期 2026-10-11。M06 已完成，M07 完成到独立 v6 CPU 记录；**PA 仍明确拒绝**。唯一下一步是 **[M07c2：独立 PA AICPU 生成器](plan.md#下一步m07c2)**。

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
| PA | 独立 CPU 地址、调度和 v6 序列化（含 BSND/TND Q-used）；未接入 AICPU、设备消费者、Copy、attention 或公开入口 |

入口参数与 v1–v5 选择统一见 [design 6.7.1](design.md#671-非分页入口与记录选择)，stride 与 PA CPU 组件见 [design 6.4](design.md#64-输入-stride-与地址校验)。调用时还须注意：

- **默认值：** func/varlen/内部前向的 `num_splits` 默认 1；kvcache/metadata 默认 0，MLA 调用须显式传 1。只接受 `causal=False`、`window_size=(-1,-1)`、`softcap=0`、`pack_gqa=None`、`sm_margin=0`、`attention_chunk=0` 和 `rotary_interleaved=True`。metadata 的 `has_softcap=False`，不接受数值 `softcap/softmax_scale` 兼容关键字。
- **长度与 scale：** 长度 tensor 必须同设备、连续、int32，值在设备检查，无 D2H。kvcache 另接受 None/整数长度；生成器必须接收 tensor。前向 scale 默认 `1/sqrt(576)`，显式值须在 FP32 有限范围，metadata 不保存 scale。
- **输出与空值：** O 同输入 dtype，LSE 为 FP32；请求 LSE 时 func/varlen 返回二项，kvcache 返回四项。非零容量中的零 KV 有效行 `O=0、LSE=-inf`；全局空 KV 初始化整个分配区为 `O=0、LSE=+inf`；其他未使用 query 槽位不承诺输出值。空输出仍校验记录和长度，损坏或过期记录由消费者 Trap 拒绝。shape、空 accumulator 和生命周期见 [design 6.7.2](design.md#672-校验输出与发射)、[5.7](design.md#57-stream-与内存生命周期)。

**计划内尚未开放：** PA、causal/window、自动或多 split、192/128 prefill、非零 softcap 与非默认调度选项。fake/compile 已有覆盖，完整 graph replay 留在 M16；910C 尚未开放或实机验收。KV 追加、内置 rotary、量化与 backward 等范围外组合也明确拒绝，完整范围见 [design 第 4 节](design.md#4-功能边界)。

## 本轮核验与证据

新增 `mla_paged_metadata.h/.cpp` 的独立 CPU 序列化，复用既有 PA 调度和共同 wire 编码。v6 为精确 `408+5B` 字，固化独立调用 context，区分 P、逻辑 C、源 S2，保存 Q 分配/used 与逐样本 KV 长度快照；不保存页表、物理页池或 stride。没有生产调用方，本步未修改调度实现、Kernel、公开入口或构建列表，详见 [design 5.8](design.md#58-v6nd-pa-的独立-cpu-记录)。

本轮证据位于忽略目录 `build/mla-m07c1-paged-metadata/`，下表未注明目录的文件均指此处。路径相对仓库根目录，不随 Git 分发；固定源、源码快照、编译命令、驱动全部输入/输出、XML、sanitizer 重放及哈希清单均保留，汇总为 `acceptance.json`。

| 核验项 | 结论与证据 |
| --- | --- |
| 上一步交接 | 开发前复跑 `8fb46d7` 的最小闭环核验；1,131 项历史测试证据、31,836 条历史 sanitizer、固定源、205 个链接、29 个表格、扩展哈希与 CATLASS 通过。`prerequisite-command.json` / `prerequisite.log`；历史结果不算本轮重跑 |
| 本轮功能 | **2,522 个去重 pytest 通过，无失败或跳过**：新 PA 协议 353、PA 调度 420、旧 v1–v5 CPU 协议/消费者 916、普通 FA3/入口契约 833。`cpu.xml`、`old-cpu.xml`、`ordinary.xml` 及对应 `*-command.json` / 日志 |
| v6 固定源差分 | **8,097 组**完整 wire 逐字一致：原生非 FD 3,614、FA3 无 split 映射 4,483。FP16/BF16 编码、全部 64 种 P、1..26 AIC、C1V1/C1V2、BSND/TND Q、可选 used、非均匀与零尾部；端点/FD/workspace 对照固定源，独立标量任务覆盖和成本一致。`paged/*-*.json`、`source/source-manifest.json` |
| 协议与缓冲边界 | 空分配/全零 KV、Q-used 为零仍验 KV、P/C/S2、前缀/used/长度更新、64 位 Q 起点和 workspace、字数/地址/字段/tile 溢出、四字节对齐、只读输入别名、输入/输出重叠、失败清零；测试解码器拒绝 3,344 次逐字损坏，旧消费者拒绝 v6 且输出不变。`paged/corruption.json`、`paged/buffers.json`、`paged/calls/` |
| Sanitizer | **ASan/UBSan/LeakSanitizer 32,506 条批量重放通过**：新 CPU 记录 8,214、源映射/原生各 8,097、迁移 SplitCore 原生 8,097、缓冲/旧消费者驱动 1；9 个进程退出 0、stderr 为空，输出 SHA256 一致。`sanitized-replay/report.json`、`manifest.json` 及完整 stdin/expected/actual；命令见 `sanitizer-replay-command.json` |
| 普通 FA3 | 在空闲 910B 卡 6 使用既有扩展，普通 metadata、func/varlen/kvcache、FP16/BF16、softcap/scale、真实/fake/C++ 签名及 compile 回归通过；`ordinary-command.json` / `ordinary.xml`。这是现有路径回归，不是 PA 设备验收 |
| 提交质量与来源 | 仅本步 8 个文件；适用质量钩子、`git diff --check`、文档链接/表格/步骤、固定源 blob、36 份编译源码快照、536 个其余受跟踪文件、实际扩展和 CATLASS 核对。`source-snapshot/`、`evidence-hashes.json`、`check_acceptance.py`、`quality-staged.log` |

初轮新协议 pytest 在 340 项通过后因测试预期错误失败 1 项，**不计为闭环通过**。极大输入计数使区间与输出重叠时，序列化器正确返回“不写输出”；该情形改由专用缓冲驱动检查无写入，同时检查不重叠的非法描述符产生失败记录。保留 `initial-cpu.log/xml`、`initial-cpu-command.json` 与 `initial-paged/`；最终完整重跑以 `cpu.xml` 为准。

损坏检查使用独立 Python 测试解码规范，严格核对全部源字段，包括诊断 `maxCost`；**尚无 PA 设备消费者或真实 Trap 结论**。C1V1 仅属 CPU 合成平台比较，实机普通回归为 910B C1V2。新代码只由独立 CPU 驱动编译，不将旧扩展的设备回归记为 v6 AICPU/attention 验收。

## 最近功能闭环与限制

以下为历史证据；本轮实际重跑项目以[上表](#本轮核验与证据)为准。

| 闭环 | 结果与可复现证据 |
| --- | --- |
| M07b2 CPU 调度，`8fb46d7` | `build/mla-m07b2-paged-schedule/acceptance.json`：1,131 项去重 pytest、7,897 组源字段差分、31,836 条 ASan/UBSan/LeakSanitizer；本轮另重跑 PA 调度 420 项 |
| M07b1 CPU 地址，`ccc5fe5` | `build/mla-m07b1-paged-layout/acceptance.json`：354 项 pytest；`sanitized-replay/report.json`：迁移 2,953、固定源 2,270 条 sanitizer 输入/输出一致 |
| 最近生产接入：M06e2c2，`538b8e5` | `build/mla-m06e2c2-mixed-stride-production/acceptance.json`：4,659 项去重 pytest；生产 448、受影响回归 2,684、普通 FA3 833、TND/独立 stride 694 |
| M06e2c2 公开性能 | 同目录 `performance-acceptance.json`：双 dtype 各 8 个适用公开用例，R=1/4 的复用、生成加前向、内部生成共 96 段逐例达标；FA3/CANN E2E 为 FP16 0.284–0.623、BF16 0.284–0.675。各 3 个原生 FD 用例单列，待 M10 |

**保留的限制：** M06e2c2 组件 Kernel 比值为 FP16 0.898–1.888、BF16 0.901–2.458，两种 dtype 均有 6 例中的 4 例慢于固定源，不能宣称 Kernel 全部不劣化。组件 profiler 的 `fwk_api_db` 解析错误保留，设备 CSV 已核验，不据此声称 host 关联可用。BSND/TND stride 的公开 E2E 证据分别在 `build/mla-m06e1c-bsnd-stride/`、`build/mla-m06e2c1-tnd-stride-production/`。

PA 的 CPU 地址/Copy trace、调度和 v6 wire 只证明对应 CPU 层，不代替设备 Copy、attention 或 E2E；>uint32 cache 地址仍仅作算术检查。全零 KV 的单校验核/零 workspace 不定义 PA 空行 O/LSE，空表/空池及设备行为由后续步骤验收。

## 环境与构建

本轮实际导入核验：Python `/usr/local/python3.11.15/bin/python3` 3.11.15、torch `2.7.1+cpu`、torch_npu `2.7.1`，CANN `/usr/local/Ascend/cann-9.0.0-beta.2`；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f` 且干净。`environment.json` / `npu-smi.log` 确认 8 张 Ascend910B3 健康，查询时 1–7 卡无运行进程，0 卡已有占用；本步仅在卡 6 运行既有路径回归，没有 PA NPU 精度或性能结果。

实际加载扩展 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so` 的 SHA256 仍为 `38582e6e4deb2eed122a38e5c1da06cd6d5386b4d00bc405f21e1cf469a4a00e`，与交接及 M06e2c2 构建一致。**未重建生产扩展，该 `.so` 不包含新增 v6 代码**；生产源码、AICPU 与实例未改，独立 CPU 驱动均由本轮源码编译。

沙箱内扩展导入因设备查询失败退出 1，`npu-smi` 因 DCMI 初始化失败退出 187；沙箱外只读查询成功，见 `environment-sandbox-failure.json`、`npu-smi-sandbox.log` 和最终环境记录。CPU pytest 使用 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`；sanitizer 在允许 LeakSanitizer ptrace 的环境执行，未关闭 leak 检查。CANN 导入的版本/tiling 注册警告保留在普通回归日志中。

只读 CANN 源仍为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，原有 67 项差异与本仓未跟踪 `kernel_meta/` 保留。使用非 login shell；质量检查沿用 `/tmp/mla-m01-precommit` 与 `/tmp/mla-m01-precommit-cache`，未修改系统安装。

## 唯一下一步与复现

下一步仅为 **[M07c2](plan.md#下一步m07c2)**：独立 PA AICPU 生成器，直接读取设备长度并复用 v6 CPU 序列化，覆盖 TND 总 Q 推导、失败清零、别名拒绝和异步生命周期；本轮提交后停止。

本轮最小闭环复核（依赖本机证据，不重新导入扩展或发射 NPU）：

```bash
python3 build/mla-m07c1-paged-metadata/check_acceptance.py
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache \
bash ci/code_quality.sh --staged
git diff --check
git diff --cached --check
```

完整功能命令见 `cpu-command.json`、`old-cpu-command.json`、`ordinary-command.json`；独立构建与固定源见 `paged/compile.json` 和 `source/`，sanitizer 复现见 `prepare_sanitizer.py` / `replay_sanitized.py` 及其命令记录。后续修改使用新的证据目录，避免覆盖本轮结果；先核对源码、二进制、输入哈希再复现。生产/性能证据仍以 M06e2c2 为准。
