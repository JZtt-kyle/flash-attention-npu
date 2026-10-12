# MLA 当前状态

核对日期：2026-10-12；分支：`feat/mla-cann-migration`；接手提交：`e243590`；最近功能闭环：`29ed04f`（M07e3c）。
**本轮完成任务边界复核（纯文档）：M08 调整为七个独立提交，明确阶段验收去重和后续里程碑边界。流程 W01–W05 已闭环；唯一下一步仍为 [M08a](plan.md#后续功能m08)。**

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

## 本次任务拆分复核

证据目录为 `build/mla-plan-review-20261012/`。本步只改 plan/status，不修改实现、测试工具、外部无头脚本或迁移契约，不执行构建/NPU 验收。接手记录与前步闭环核对见 `baseline.json`，审查依据见 `review.json`，文档和质量检查分别见 `docs-check.json`、`quality.log`。

- M08a 的 CPU 算法与分核计划保留为一个提交，补全独立 oracle、边界、ASan/UBSan 及共享 CPU 调度回归条件。原 `mla_causal.h` 只有草稿声明，内容保持不变；没有把它标为已实现。
- 原 M08b 拆为 **b1 数值组件、b2 BSND 完整接入**。共享 `MlaTaskRing` 当前遍历整段 S2，Vector 尚无 causal，mask/空行/任务环语义需要先独立实测；稳定数值组件再同次接通 metadata、生产和公开入口。
- 原 M08c 拆为 **c1 TND varlen、c2 混合 cache**。BSND、TND、混合、PA 的 used 行为分别归属对应接入步，不另排一轮笼统的 used 迁移。M08d/e 保持各自 PA 布局闭环，优先复用已有页 Copy。
- M08e 在冻结实现上运行“本步必跑项与阶段矩阵的并集”；同一当次用例/性能段只执行及计数一次。阶段矩阵仍须实测，不使用历史提交的结果补数，也不混淆组件和公开 E2E。
- 新增对照适配作为相应功能步的前置工作：双数学核心虽已有 mask，现有原生基线、PA reference/profile 仍需接通 causal。缓存要验证 mask 语义键及失效行为；不能因已有无 mask 驱动就跳过新能力验收。
- M09–M17 明确为能力里程碑，临近实施时再拆出可闭环提交。特别是 split、prefill 新数值表示、softcap/调度和图执行，不直接把整阶段交给一个无头轮次。已完成流程的长命令移出当前待办区。

本步结论是按风险适当拆开两块过大的工作，同时合并稳定布局的接入与验证；没有恢复逐层多轮验收。当前顺序为 **a → b1 → b2 → c1 → c2 → d → e**，具体产物、影响和完成条件只在 plan 维护。

## 可继承的流程证据

以下流程实现没有变化，本步继承其有效记录，未重跑这些测试。W02–W04 的记录位于 `build/mla-workflow-tools-20261012/`；共享构建和数学缓存为 `build/mla-validation-artifacts/`。入口及缓存边界仍见 [design 7.4](design.md#74-统一验收入口)。以下均为历史实测：

- 八组原验收节点及选择条件与 M07e3c 的 9,614 项一致，见 `selection-equivalence.json`；不表示本步执行了这些测试。
- `tools cpu public ordinary` 历史去重通过 3,015 项：46、2、2,145、822；无失败/跳过。冷热公开组分别耗时 258.49 / 124.81 秒，后者 548 次 reference 命中，整组缩短 51.7%。历史恢复继承 3,015 项、当次执行 0 项。见 `cold-summary.json`、`warm-summary.json`、`resume-summary.json` 与 `acceptance.json`。
- 构建核对约 1.1–1.2 秒，实际运行栈检查另约 20 秒。上轮实际加载的扩展 SHA256 为 `4e2cf66ad8a507b1c3bad54bf3ab5f440ab928ffbd95041104e1c54470e3782c`。长 KV CPU 参考缓存的逐位一致性与耗时见 `reference-timing/timings.json`；不是内核或整个测试组的加速比。
- 该轮记录中的环境为 Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2、910B3 第 6 卡；CATLASS `769cd40a`、固定 CANN 源 `ca18aee6`。环境原始记录仍在该目录；本步没有重新探测设备，不据此推断下一次运行时设备空闲。
- 原 M08 十二步草稿保存在 `preexisting.patch`，causal 头文件备份仍在。W05 已清理 3,520 个空 runtime 日志及重复备份，代码草稿保留；旧 `preservation.json` 是清理前的记录。

W05 的记录位于 `build/mla-cleanup-20261012/`：删除 7,269 个文件和 1,203 个空目录，释放文件实际占用约 1.46 GiB，7,840 个受保护文件清理前后哈希一致；有效构建、缓存及验收证据保留。外部无头脚本 16 项回归、本机 CLI `0.162.1` 自检及预览当时通过，未实际调用模型开发；本步仅核对脚本哈希未变。

脚本仍为 `/home/zhangqijia/repo/mla_codex_loop.py`，放在仓库外，不纳入 Git。取消固定五轮文档审计，支持 `--check`、明确 ID 恢复与过期轮次显式归档。旧 `a70ed66` 开发轮已归档为 `build/mla-codex-loop/20261012-015617-development-8punhjk7/retired.json`，不计成功；当时计数保持 45、pending 清空。下次启动读取当时 HEAD 和当前 status/plan，实际状态以 `--check` 为准。

证据和缓存均被 Git 忽略，不随提交分发；迁机须复制实际依赖或重新验收。清理清单、外部脚本测试和本次文档检查也保存在本机忽略目录。

## 唯一下一步与复现

下一步仅为 [M08a](plan.md#当前步-m08acpu-算法闭环)：复核 CPU causal 草稿，实现块范围/分核计划及差分、边界和 sanitizer 检查。按 plan 的当前步执行即可，无需重新进行任务拆分或流程审批；本次文档提交后停止，没有开始 M08 实现。

本次文档复核的最小命令：

```bash
python3 build/mla-plan-review-20261012/check_docs.py
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache \
bash ci/code_quality.sh --staged
git diff --check
git diff --cached --check
```

正式启动一轮仍使用 `python3 /home/zhangqijia/repo/mla_codex_loop.py --max-dev-runs 1`，启动前可运行 `--check`；省略次数则持续运行，失败时保留本轮并停止。当前 M08a 是待实现的 CPU 检查，不能用统一入口的旧 `cpu` 或无 mask `public` 组代替。后续新能力的组与性能对照随对应功能步补齐，按 plan 验收。
