# MLA 当前状态

核对日期：2026-10-12；分支：`feat/mla-cann-migration`；代码基线与最近功能闭环：`29ed04f`（M07e3c）。
**M07 已完成，下一步仍为 [M08：512+64 右下 causal](plan.md#下一步m08)。本次仅核验和整理文档、docstring 与注释，没有新增功能或重跑 NPU 测试。**

本页保存当前能力、最近闭环证据和环境差异；接口与架构见 [design](design.md)，实施顺序见 [plan](plan.md)。本次文档提交可用 `git log -1 -- docs/mla/status.md` 定位。

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

**以下是 M07e3c 的历史实测结果，本次只复核证据。** 原始目录为 `build/mla-m07e3c-tnd-paged-public/`，汇总为 `acceptance.json`；保存命令、日志/XML、精度输入/输出、源码快照、实际扩展、固定源库及性能原始数据。

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

## 本次文档核验与环境差异

本次证据目录：`build/mla-docs-audit-20261012/`。

- 编辑前执行原 `check_acceptance.py` 成功，复核上述测试记录、实际扩展、38 个对象、274 个仓库源及 6,200 个依赖和性能数据；输出保存在 `prerequisite.log` / `prerequisite-acceptance.json`。
- 对照 Python/fake、C++ 分派、v1–v6 定义、PA 计划/消费者及对应测试，修正 TND PA 接入状态、Q-used 范围和旧阶段注释；整理三份文档职责。固定官方 FA3 内容的本机快照保存在 `fa3-reference/`，版本与来源见 design。
- 本次核对的 263 个本地链接/锚点、29 张表格、4 个公开接口的参数名/默认值、v1–v6 字数及历史证据计数均通过；Python 去 docstring 后 AST 相同，C++ 去整行注释后内容相同。结果保存在 `docs-check.json`；`git diff --check` 与适用质量钩子通过，输出在 `quality.log`。

历史实测环境为 Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2；8 张 910B3，性能/精度使用物理卡 6，相关回归/Trap 使用卡 7。**本次不探测设备占用、不加载或重建扩展**；这份历史环境不代表下一次实机运行时的可用资源。

保留扩展 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 为 `6734d89388dfd4c6368bac932d3b41471860f3219198253a3e3dd2b003a4bbc9`。本次 docstring/注释使源码字节哈希变化，但可执行代码未变；后续开发仍须按正常依赖规则重建并核验实际加载路径。CATLASS 固定版本干净；只读 CANN 固定源原有 67 项差异、本仓原有未跟踪 `kernel_meta/` 均保留。

两份证据目录均被 Git 忽略，**不随提交分发**；迁机须复制它们及 manifest 引用的历史构建、工具链和固定源依赖。缺少证据时不能仅凭本页数字宣布复核通过。

## 唯一下一步与复现

下一步仅为 **[M08](plan.md#下一步m08)**：先拆分 512+64 右下 causal 的独立验收步骤，再开始实现。本次提交成功后停止。

本次最小复核（仓库根目录，非 login shell，不发射 NPU）：

```bash
python3 build/mla-docs-audit-20261012/check_docs.py
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache \
bash ci/code_quality.sh --staged
git diff --check
git diff --cached --check
```

质量命令中的 `/tmp` 路径是本机已有工具环境；其他机器按 `ci/quality-requirements.txt` 准备依赖。原 M07e3c 的 `check_acceptance.py` 比较整份源码/文档快照，**仅适用于 `29ed04f` 对应工作树**；本次改文档后应使用上面的复核命令，不能把旧快照不匹配误报为数值回归。重新实机验收时，将原目录的 `run_checks.py` / `run_device_checks.py` 复制到新证据目录，按各模式 `*-command.json` 准备环境，先核对源码、构建、设备和固定源库，不覆盖历史数据。
