# MLA 当前状态

本轮恢复并完成 **M07b1 后中断的文档核验与整理（纯文档）**，核验基线 `ccc5fe5`，分支 `feat/mla-cann-migration`，日期 2026-10-11。开发进度未推进：M06 已完成，M07 完成到独立 PA CPU 地址组件；**PA 仍明确拒绝**。唯一下一步是 **[M07b2：PA CPU 调度](plan.md#下一步m07b2)**。

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
| PA | 仅独立 CPU 地址、有效页校验与页内分段；未接入调度、metadata、设备 Copy、attention 或公开入口 |

入口参数与 v1–v5 选择统一见 [design 6.7.1](design.md#671-非分页入口与记录选择)，stride 与 PA CPU 组件见 [design 6.4](design.md#64-输入-stride-与地址校验)。调用时还须注意：

- **默认值：** func/varlen/内部前向的 `num_splits` 默认 1；kvcache/metadata 默认 0，MLA 调用须显式传 1。只接受 `causal=False`、`window_size=(-1,-1)`、`softcap=0`、`pack_gqa=None`、`sm_margin=0`、`attention_chunk=0` 和 `rotary_interleaved=True`。metadata 的 `has_softcap=False`，不接受数值 `softcap/softmax_scale` 兼容关键字。
- **长度与 scale：** 长度 tensor 必须同设备、连续、int32，值在设备检查，无 D2H。kvcache 另接受 None/整数长度；生成器必须接收 tensor。前向 scale 默认 `1/sqrt(576)`，显式值须在 FP32 有限范围，metadata 不保存 scale。
- **输出与空值：** O 同输入 dtype，LSE 为 FP32；请求 LSE 时 func/varlen 返回二项，kvcache 返回四项。非零容量中的零 KV 有效行 `O=0、LSE=-inf`；全局空 KV 初始化整个分配区为 `O=0、LSE=+inf`；其他未使用 query 槽位不承诺输出值。空输出仍校验记录和长度，损坏或过期记录由消费者 Trap 拒绝。shape、空 accumulator 和生命周期见 [design 6.7.2](design.md#672-校验输出与发射)、[5.7](design.md#57-stream-与内存生命周期)。

**计划内尚未开放：** PA、causal/window、自动或多 split、192/128 prefill、非零 softcap 与非默认调度选项。fake/compile 已有覆盖，完整 graph replay 留在 M16；910C 尚未开放或实机验收。KV 追加、内置 rotary、量化与 backward 等范围外组合也明确拒绝，完整范围见 [design 第 4 节](design.md#4-功能边界)。

## 本轮核验与证据

本轮只修改 design、plan、status。原核验证据保留在忽略目录 `build/mla-docs-audit-ccc5fe5/`；恢复核验与最终检查位于 `build/mla-docs-audit-ccc5fe5-resume-20261011/`，下表未注明目录的文件均指后者。路径均相对仓库根目录，不随 Git 分发；包含核验前文档、来源快照、哈希清单、检查脚本及结果。没有改源码、重建或加载扩展，也没有运行 CPU/NPU 功能、精度或性能测试。

| 核验项 | 结论与证据 |
| --- | --- |
| 上一步交接 | 修改文档前复跑 M07b1 的文档/交接检查；199 个本地链接、27 个表格及既有源码/扩展哈希通过，历史 XML 与最终 sanitizer 报告可核对。原目录的 `prerequisite.log`、`prerequisite-check.json`；其中测试数来自历史记录，不是本轮重跑 |
| 中断恢复 | 会话记录、Git HEAD/reflog 与进程核对确认无新提交、无遗留 MLA 构建/测试/提交进程；原 53 份核验产物及三份文档哈希匹配。`recovery.json`、`process-check.json`；此前提交审核超时未执行，不计为成功提交 |
| 文档与实际代码 | 修正混合 v4/v5 的 stride 限制和模板映射、TND stride 的过时待接入表述；补齐 PA CPU 接口与未进生产构建的边界。Python/fake、C++ 分派、前向、launcher、metadata 常量及测试清单逐项核对，见 `docs-check.json` |
| 固定来源与历史证据 | 25 份 CANN 固定 Git blob、官方 FA3 Python/C++ 归档及公开签名/默认值核对；M07b1 与 M06e2c2 共 5,867 份历史证据、534 个其余受跟踪文件及磁盘扩展哈希一致。`source-audit.json`、`signatures.json` |
| 结构与提交质量 | 调用分支集中到 design，plan 待办在前、历史在后，status 保留能力摘要与交接；恢复后重跑文档核验，202 个本地链接/锚点、29 个表格、78 个步骤、22 组性能清单、命令语法及适用质量钩子通过。`docs-check.json`、`quality-staged.log`、`diff-check-command.json`；汇总见 `acceptance.json` |

## 最近功能闭环与限制

以下是**此前闭环，本轮只复核证据和来源**，不作为本轮新增测试结果。

| 闭环 | 结果与可复现证据 |
| --- | --- |
| M07b1 CPU 地址，`ccc5fe5` | `build/mla-m07b1-paged-layout/acceptance.json`：354 个去重 pytest 通过，无失败或跳过（PA 171、BSND 82、TND 101）；`cpu-command.json` / `cpu.xml` 保存完整命令及结果 |
| M07b1 sanitizer | 同目录 `sanitized-replay/report.json`：ASan/UBSan/LeakSanitizer 批量重放迁移驱动 2,953、固定源驱动 2,270 条输入，输出哈希相同，两进程退出 0、stderr 为空；复现脚本为 `replay_sanitized.py` |
| 最近生产接入：M06e2c2，`538b8e5` | `build/mla-m06e2c2-mixed-stride-production/acceptance.json`：4,659 个去重 pytest 通过，无失败或跳过；生产 448、受影响回归 2,684、普通 FA3 833、TND/独立 stride 694 |
| M06e2c2 公开性能 | 同目录 `performance-acceptance.json`：双 dtype 各 8 个适用公开用例，R=1/4 的复用、生成加前向、内部生成共 96 段逐例达标；FA3/CANN E2E 为 FP16 0.284–0.623、BF16 0.284–0.675。各 3 个原生 FD 用例单列，待 M10 |

**保留的限制：** M06e2c2 组件 Kernel 比值为 FP16 0.898–1.888、BF16 0.901–2.458，两种 dtype 均有 6 例中的 4 例慢于固定源，不能宣称 Kernel 全部不劣化。组件 profiler 的 `fwk_api_db` 解析错误保留，设备 CSV 已核验，不据此声称 host 关联可用。BSND/TND stride 的公开 E2E 证据分别在 `build/mla-m06e1c-bsnd-stride/`、`build/mla-m06e2c1-tnd-stride-production/`。

M07b1 的 CPU Copy trace 只证明地址和分段，不能代替 PA 设备/attention/E2E；>uint32 cache 地址只作算术检查。早期 sanitizer 逐用例运行一次因沙箱 ptrace 限制失败、一次在 2 项通过后主动中断，均不计为通过；原日志和退出码保留在 M07b1 目录，闭环依据为最终批量重放报告。

## 环境与构建

本轮只作静态环境和来源核对：Python `/usr/local/python3.11.15/bin/python3` 3.11.15，已安装 torch/torch_npu distribution 均为 2.7.1，CANN `/usr/local/Ascend/cann-9.0.0-beta.2`；CATLASS 固定为 `769cd40a8716b28650b6bebb08db4834eea4462f` 且干净。M07b1 的最近实际导入记录为 torch `2.7.1+cpu`、torch_npu `2.7.1`；distribution 字符串不同不代表环境升级。

磁盘扩展 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so` 的 SHA256 仍为 `38582e6e4deb2eed122a38e5c1da06cd6d5386b4d00bc405f21e1cf469a4a00e`，与 M07b1 的加载记录和 M06e2c2 最终构建记录一致；**PA CPU 组件未进入该扩展**。本轮未查询 NPU 健康或占用；M07b1 曾确认 8 张 Ascend910B3 健康，该历史记录不代表下次测试时设备可用。

只读 CANN 源仍为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，原有 67 项差异与本仓未跟踪 `kernel_meta/` 保留。使用非 login shell；质量检查沿用 `/tmp/mla-m01-precommit` 与 `/tmp/mla-m01-precommit-cache`，未修改系统安装。

## 唯一下一步与复现

下一步仅为 **[M07b2](plan.md#下一步m07b2)**：独立 PA CPU 调度及固定源完整字段差分、sanitizer 和旧 CPU 路径回归；本轮提交后停止。

本轮最小文档闭环（依赖本机保存的证据目录；检查脚本不导入 torch、不发射 NPU）：

```bash
python3 build/mla-docs-audit-ccc5fe5-resume-20261011/check_docs.py
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache \
bash ci/code_quality.sh --staged
git diff --check
git diff --cached --check
```

后续变更使用新的证据目录，避免覆盖旧结果。CPU 复现见 M07b1 的 `cpu-command.json`、`sanitizer-replay-command.json`；sanitizer 依赖完整驱动输入及允许 LeakSanitizer ptrace 的环境。生产/性能复现见 M06e2c2 的 `production-build-command.json`、`production-audit-command.json`、`performance.py` 及原始 command JSON。进入开发或实机验证前重新检查环境、设备占用、构建和实际加载来源，再按影响范围重跑必要检查。
