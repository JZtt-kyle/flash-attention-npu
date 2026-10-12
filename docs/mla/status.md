# MLA 当前状态

核对日期：2026-10-12；分支：`feat/mla-cann-migration`；接手提交：`a70d0da`；最近功能闭环：`29ed04f`（M07e3c）。
**本轮完成 W05：清理无用产物、核对保留证据、修复外部无头脚本接续；W01–W05 已闭环。未启动 M08 开发，唯一下一步为 [M08a](plan.md#后续功能m08)。**

本页保存当前能力、最近闭环证据和环境差异；接口与架构见 [design](design.md)，实施顺序见 [plan](plan.md)。当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。

## 当前可调用能力

共同边界：**910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、无梯度、无 mask、`num_splits=1`**。Q/K 末维为 64，qv/V 为 512，四路同 dtype、同 NPU 设备；末维连续，合法外层 stride 按完整 storage 校验。每样本 KV 容量或 TND 段长上界为 16384，TND 总 KV 和 PA 物理页池不套用这个上限。

| 入口 | 已接通输入 | metadata 与长度 |
| --- | --- | --- |
| `flash_attn_func` | BSND Q/KV | 整容量，内部 v1 |
| `flash_attn_varlen_func` | TND Q/KV | 两组分配前缀；无 used 时内部 v3，任一 `seqused_q/k` 存在时内部 v5 |
| `flash_attn_with_kvcache` | BSND Q + BSND KV | 无长度且无记录走 v1，否则内部生成或显式复用 v2 |
| `flash_attn_with_kvcache` | TND Q + BSND KV | Q 完整分配段，内部生成或显式复用 v4 |
| `flash_attn_with_kvcache(page_table=...)` | BSND/TND Q + ND PA KV | Q 完整分配段，内部生成或显式复用 v6 |
| `get_scheduler_metadata` | BSND/TND Q 的只读 cache 调度 | 非 PA 生成 v2/v4；指定 `page_size` 生成 v6 |
| 内部 `fwd` / PA 专用算子 | 非 PA 三布局及 BSND/TND PA 的 Q-used | 分别内部生成 v5/v6；拒绝 Q-used 与显式记录复用 |

调用要点：

- kvcache 和 metadata 的 `num_splits` 默认 0，MLA 须显式传 **1**。生成器须指定 `headdim=64, headdim_v=512`，并让 `qkv_dtype` 匹配输入（默认 BF16）。完整选项见 [design 2.2](design.md#22-后端选项与返回值)。
- kvcache 的 KV 长度接受 None、整数或同设备连续 `int32[B]`；生成器只接受 tensor。TND cache 前向须给 `cu_seqlens_q` 和 `max_seqlen_q`。**公开 varlen 支持 Q-used；公开 kvcache 和 metadata 生成器没有 Q-used 参数。**
- PA 的 P 为 16..1024 的 16 倍数，逻辑容量 `C=页表列数×P` 在 0..16384；生成器的 `max_seqlen_k` 须等于 C。None 表示 C，整数须在 0..C，按逻辑 B 物化，不能用 Tq 或物理页数代替 B。
- 分配前缀决定物理起点，used 只限制有效范围。记录不绑定地址、stride、页表或 scale；每次前向核对当次长度、前缀和全部有效页。长度或前缀改变须重生成；输入与记录在异步执行期间须保持只读和存活。详见 [metadata 复用](design.md#51-签名与复用规则)。
- func/varlen 请求 LSE 返回二项，kvcache 返回四项，未 split 的两个 accumulator 为空 tensor。PA 的有效零 KV 行一律 `O=0、LSE=-inf`；非 PA 全局空 KV 为 `+inf`。完整 shape 与空值契约见 [design 3.3](design.md#33-共同契约)。

真实/fake 入口及公开 fullgraph compile 已有验收；完整图重放仍在 M16。尚未开放：causal/window、自动或多 split、192/128 prefill、非零 softcap、非默认调度选项和 910C。追加 KV、内置 rotary、量化、backward 等范围排除项见 [design 第 4 节](design.md#4-功能边界)。

## 最近功能闭环证据

**以下是 M07e3c 的历史实测结果，本次前置检查仅复核既有记录，不计为本次实测。** 原始目录为 `build/mla-m07e3c-tnd-paged-public/`，汇总为 `acceptance.json`；保存命令、日志/XML、精度输入/输出、源码快照、实际扩展、固定源库及性能原始数据。

| 检查 | 历史结果 | 目录内依据 |
| --- | --- | --- |
| 构建与 CPU | 2 项通过；重建 38 个对象，仅 `flash_api.o` / `mla_forward.o` 变化 | `cpu.xml`、`production-load.json`、`production-object-delta.json`、`production-final/` |
| TND PA 公开前向 | 2,145 项通过；全部 64 种 P、双 dtype、源/双数学 O/LSE、入口/stride/空值/复用/compile/生命周期 | `public.xml`、`probes/*/tnd-public-accuracy/` |
| 真实 Trap | 46 项通过，逐例独立进程 | `trap.xml`、`kvcache/*/trap-*/` |
| 相关回归 | BSND PA 公开 1,877、TND PA 内部 784、PA metadata 2,840、非分页/独立 TND PA 1,098、普通 FA3/入口 822 项通过 | `bsnd/device/metadata/nonpaged/ordinary.xml`（各一个文件） |
| 原生 CANN 分派 | 双 dtype 各 12 例，非 FD/FD 各 6 例；固定源库、注册与 launch key 已核验 | `native-package.json`、`native-dispatch-proof.json` |
| 公开 E2E | 每种 dtype 的 36 个适用非 FD 调用段均 ≤1.0；FP16 比值 0.348..0.683，BF16 0.353..0.677 | `public-profile-fp16/`、`public-profile-bf16/` |

共 **9,614 项去重 pytest 通过，无失败/跳过**；154 项前置 smoke 属于完整矩阵，不重复计数。E2E 双 dtype 各固定 12 例、R=1/4、预热 5 次、5 轮×20 wall 样本；计入必要转换、分配与完成同步。原生 FD 各 6 例单列，未纳入非 FD 门槛；不能据上表推断 FD 已迁移或全部原生路径均不劣化。验收口径见 [design 第 7 节](design.md#7-验收口径)。

该功能步未改设备数值组件；源 Kernel 组件对照由前一步对象/依赖哈希复核保留。未重跑全部历史 Trap、全部非分页/普通 FA3、NPU/CPU sanitizer 或完整图重放。中止采样、构建失败及未被 pytest 使用的快照差异保留在原目录及 `acceptance.json`，不计为通过。

## 本次清理与无头接续

证据目录为 `build/mla-cleanup-20261012/`。本步只修改忽略规则、流程交接和仓库外脚本；没有修改内核、接口、构建或验收实现，没有重新发射 NPU 或采样性能。

- 删除 **7,269 个文件、1,203 个空目录**，文件逻辑大小约 1.95 GiB，释放文件实际占用约 **1.46 GiB**：两份过期 core 转储、7,040 个空 runtime 日志（含旧备份）、12 个已删除源对应的编译残留、215 个未纳入验收的 pytest 临时文件。每个文件的路径、原因和 SHA256 见 `cleanup-manifest.json`，结果见 `cleanup-result.json`。
- 清理前后 **7,840 个受保护文件**逐项哈希一致，包括共享构建的 6,317 个依赖与 11 个输出、冷热验收原始证据以及 causal 草稿。验收源码闭包 793 个文件与上轮冷测一致，见 `readiness.json`。保留在用 `.so`、有效增量构建对象、共享数学缓存、历史数值输入/输出、性能原始数据和固定源快照；剩余 `build/` 约 88.3 GiB，主要是这些证据。`kernel_meta/` 新增忽略规则，避免空 runtime 日志污染工作树。
- 无头脚本仍为 **`/home/zhangqijia/repo/mla_codex_loop.py`**，按用户要求放在仓库外，脚本及其测试均未纳入 Git。旧版、新版及差异保存在本步证据目录的 `loop-before.py`、`loop-after.py`、`loop-change.patch`，外部文件哈希见 `external-update.json`。
- 脚本取消固定五轮文档审计，默认每轮按当前 status/plan 完成一个提交；仍用明确会话 ID 恢复中断。新增 `--check` 启动自检、`--review-docs` 单次文档核验、`--retire-pending 原因` 归档已被替代的未完成轮次。恢复基线偏离时在调用模型前报错；归档不计成功、不删除源码或会话日志。
- 原待恢复轮次起点为 `a70ed66`，已被后续流程提交替代，现已显式归档在 `build/mla-codex-loop/20261012-015617-development-8punhjk7/retired.json`。开发和文档覆盖计数均保持 45，`pending=null`；原 `mla_causal.h` 草稿和会话日志保留。下一次启动会以当时 HEAD 新建 M08a 开发轮，不再恢复旧计划。
- 外部脚本 **16 项回归通过**，覆盖真实临时 Git 提交、失败/中断恢复、成功回执补计数、错误会话、过期基线、归档、互斥、只读预览与计数迁移，见 `external-loop-tests.log`。本机 CLI `0.162.1` 已登录，实际 `exec/resume` 参数解析、原入口 `--check` 和一轮 `--dry-run` 均通过，见 `headless-check.json`、`cli-resume-help.txt`、`headless-preview.txt`。这些是本地启动与模拟回归检查，**未调用真实模型执行开发**。

## 可继承的流程证据

W02–W04 的记录位于 `build/mla-workflow-tools-20261012/`；共享构建和数学缓存为 `build/mla-validation-artifacts/`。入口及缓存边界仍见 [design 7.4](design.md#74-统一验收入口)。以下均是前一步实测，本步只复核依赖与证据：

- 八组原验收节点及选择条件与 M07e3c 的 9,614 项一致，见 `selection-equivalence.json`；不表示本步执行了这些测试。
- `tools cpu public ordinary` 历史去重通过 3,015 项：46、2、2,145、822；无失败/跳过。冷热公开组分别耗时 258.49 / 124.81 秒，后者 548 次 reference 命中，整组缩短 51.7%。历史恢复继承 3,015 项、当次执行 0 项。见 `cold-summary.json`、`warm-summary.json`、`resume-summary.json` 与 `acceptance.json`。
- 构建核对约 1.1–1.2 秒，实际运行栈检查另约 20 秒。当前扩展 SHA256 仍为 `4e2cf66ad8a507b1c3bad54bf3ab5f440ab928ffbd95041104e1c54470e3782c`。长 KV CPU 参考缓存的逐位一致性与耗时见 `reference-timing/timings.json`；不是内核或整个测试组的加速比。
- 上轮环境为 Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2、910B3 第 6 卡；CATLASS `769cd40a`、固定 CANN 源 `ca18aee6`。环境原始记录仍在该目录；本步没有重新探测设备，不据此推断下一次运行时设备空闲。
- 原 M08 十二步草稿保存在 `preexisting.patch`，原 causal 头文件备份保留；上轮 `preservation.json` 记录的是当时的 3,521 个未跟踪文件。本步按用户要求移除了其中 3,520 个空日志及其重复备份，代码草稿内容未变。

证据和缓存均被 Git 忽略，不随提交分发；迁机须复制实际依赖或重新验收。清理清单与外部脚本测试同样保存在本机忽略目录，不是仓库交付文件。

## 唯一下一步与复现

下一步仅为 [M08a](plan.md#后续功能m08)：复核 CPU causal 草稿，实现独立块范围/分核计划及其差分、边界和 sanitizer 检查。本轮流程提交后停止。不把 CPU 草稿或已编译对象计为 causal 能力已开放。

本步最小检查（不调用模型、不执行 NPU 测试）：

```bash
python3 /home/zhangqijia/repo/mla_codex_loop.py --check
python3 /home/zhangqijia/repo/mla_codex_loop.py --dry-run --max-dev-runs 1
MLA_CODEX_LOOP_PATH=/home/zhangqijia/repo/mla_codex_loop.py \
python3 -m unittest discover -s build/mla-cleanup-20261012 -p 'test_mla_codex_loop.py' -v
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache \
bash ci/code_quality.sh --staged
git diff --check
git diff --cached --check
```

正式启动一轮使用 `python3 /home/zhangqijia/repo/mla_codex_loop.py --max-dev-runs 1`；省略次数参数则持续运行，失败时保留本轮会话并停止。脚本沿用本机模型/登录配置和自动审批，`codex exec/resume` 的接口见 [官方无头模式文档](https://learn.chatgpt.com/docs/non-interactive-mode)。启动检查不保证远端服务或未来设备可用；实际开发仍须满足 AGENTS 的本步验收门槛。

MLA 验收继续使用统一入口，`--resume` 只继承有效成功组。新能力须补相应测试，现有 `public` 仍是无 mask 的 TND PA 组。质量工具的 `/tmp` 路径为本机环境，迁机按 `ci/quality-requirements.txt` 准备。
