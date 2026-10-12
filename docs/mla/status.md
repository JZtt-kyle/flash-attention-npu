# MLA 当前状态

核对日期：2026-10-12；分支：`feat/mla-cann-migration`；接手提交：`c947d96`；最近功能闭环：`29ed04f`（M07e3c）。
**本轮完成 W02–W04：统一验收入口、构建/reference/证据分别复用、按组恢复和 M08 计划重排；流程 W01–W04 已闭环。M08 尚未开始，唯一下一步为 [M08a](plan.md#后续功能m08)。**

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

## 本次流程工具与证据

本次证据目录：`build/mla-workflow-tools-20261012/`；共享构建及数学缓存：`build/mla-validation-artifacts/`。入口为 [tests/mla_validation.py](../../tests/mla_validation.py)，参数和边界见 [design 7.4](design.md#74-统一验收入口)。

- 八个原验收组的 9,614 个节点及全部选择条件与 M07e3c 完全一致，见 `selection-equivalence.json`。这是完整清单对照，不代表本轮执行了全部历史用例。独立页映射及数学 reference 的原函数体经 AST 对照保持一致。
- 46 项执行器/缓存检查已通过：退出码、失败/跳过/空集合、加载身份、证据损坏、输入/布局/源码/构建依赖变化、恢复和互斥。构建依赖只取本次实际链接对象，已覆盖历史 `.d` 残留导致的误判。
- 新入口的 `tools cpu public ordinary` 共 **3,015 项去重通过，无失败/跳过**：工具 46、构建基础 2、公开组 2,145、普通 FA3/入口 822。见 `cold-summary.json`；公开组又以有效缓存完整实测 2,145 项，548 次 reference 全部命中，见 `warm-summary.json`。后者是重复验证，不重复计入去重总数。
- 公开组首次缓存运行 258.49 秒，命中运行 124.81 秒，整组缩短 **51.7%**；NPU 路径、节点集合和精度判定不变。有效构建核对约 1.1–1.2 秒，另有约 20 秒的实际运行栈检查。真实 `--resume` 将四组全部标为 `inherited`，继承 3,015 项、本次执行 0 项，见 `resume-summary.json`；汇总见 `acceptance.json`。
- 初版收集/构建期间的源码变化，以及扫描历史 `.d` 导致的依赖误判，保留在 `prepare.log` 和旧 attempt 中，不计通过。依赖收集已绑定实际链接对象；最终构建核对 6,317 个依赖，实际扩展 SHA256 为 `4e2cf66ad8a507b1c3bad54bf3ab5f440ab928ffbd95041104e1c54470e3782c`。
- 原有 3,521 个未跟踪文件内容保持不变，见 `preservation.json`。本步仅提交明确列出的流程/测试文件；文档链接、表格、命令语法和差异核对见 `docs-check.json`，暂存质量检查日志为 `quality.log`。
- M08 由原十二个分层子步重排为 CPU 算法、BSND 非 PA、TND/used、BSND PA、TND PA 及阶段验收五个阶段。各阶段明确必跑项和可继承边界，精度、普通 FA3 回归及性能矩阵不变；原草稿存为 `preexisting.patch`，原 `mla_causal.h` 与 `kernel_meta/` 内容保留并备份在 `original-untracked/`。

本机长 KV CPU 参考缓存测量（Q allocation 7/0/3，KV 8193/17/1025，H=128，P=112，双数学 O/LSE）：

| dtype | 禁用缓存 | 首次写入 | 命中读取 |
| --- | --- | --- | --- |
| FP16 | 182.471 秒 | 197.827 秒 | 0.069 秒 |
| BF16 | 215.990 秒 | 216.477 秒 | 0.070 秒 |

四个张量在关闭、首次计算和命中读取之间逐位一致。原始数据见 `reference-timing/timings.json`，命令快照为 `reference_timing.py`；这是本机一次 CPU 参考预检样本，不是内核性能或整个测试组的加速比。性能采样仍用现有 `tests.mla_paged_profile` 实测双方设备调用与 E2E。

本次环境：Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2、910B3 第 6 卡；CATLASS `769cd40a`、固定 CANN 源 `ca18aee6`。接手状态见 `environment.json` 与 `npu-before.log`；本次实际加载身份由执行器另存到每个 attempt。当前所有改动属于流程、测试工具和 reference 复用，没有修改生产数值核或公共接口。

证据和缓存均被 Git 忽略，不随提交分发；迁机须复制其实际依赖或重新验收。旧 M07 功能和性能结果仍按上表标为历史，本次未重新采样完整 E2E 性能矩阵。

## 唯一下一步与复现

下一步仅为 [M08a](plan.md#后续功能m08)：复核 CPU causal 草稿并实现独立块范围/分核计划及其差分、边界和 sanitizer 检查。本轮流程提交后停止。不把 CPU 草稿或已编译对象计为 causal 能力已开放。

```bash
python3 -m tests.mla_validation --output build/mla-workflow-tools-20261012/run \
  --suite tools cpu public ordinary --device 6 --resume
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache \
bash ci/code_quality.sh --staged
git diff --check
git diff --cached --check
```

`--resume` 继承有效成功组，失败/中断组重新执行；需要当前实测时去掉该参数。`--dry-run` 只预览命令，`--collect-only` 只核对节点，不作为测试通过。新能力测试须另加组，现有 `public` 仍是无 mask 的 TND PA 组。质量工具的 `/tmp` 路径为本机环境，迁机按 `ci/quality-requirements.txt` 准备。
