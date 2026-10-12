# MLA 当前状态

**增量验收提速已完成；M08b1 保持闭环。** 分支 `feat/mla-cann-migration`。数学参考缓存、固定日常回归清单及按组依赖恢复已接通并验收；唯一下一步是 [plan 的 M08b2](plan.md)。精度仍按 [design 第 7 节](design.md#7-验收口径) 的固定源逐元素一致优先级执行。本次提交后停止，不启动 M08b2。

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

## 本次验收提速与证据

证据根目录为 `build/mla-validation-speedup/`，汇总为 `validation-summary.json`。最终验收 **868 次执行通过（542 个不同节点），0 项失败、0 项跳过、0 项必跑未完成**：工具 145 项、原独立数学契约 71 项，两遍设备各 326 项。CPU 成功组恢复另核对 216 项，标为 inherited，不重复计入执行数。旧完整矩阵没有删项，本次未重跑无关的完整精度/性能矩阵。

| 检查 | 节点数 | 第一遍秒数 | 缓存命中运行 |
| --- | ---: | ---: | --- |
| `regression-dense` | 84 | 560.0 | 321.8 秒 |
| `regression-nonpaged` | 86 | 749.1 | 771.8 秒 |
| `regression-paged` | 120 | 83.1 | 41.8 秒 |
| `causal-check` | 36 | 149.9 | 123.2 秒 |

第一遍四组合计 **25.7 分钟**，缓存命中运行 **21.0 分钟**，均含独立 probe 构建；生产准备和运行栈检查每遍另约 22 秒。三个共享回归组从原 **1,675 项、110.1 分钟**精简为 **290 项、缓存命中运行 18.9 分钟**；覆盖范围不同，这表示已确认的日常选择与缓存的共同收益。第二遍 **504 次参考读取全部命中、0 次未命中**，所有设备用例重新执行。16K KV 的 BF16 用例从 67.34 秒降到 0.32 秒，见 `long-reference-cost.json`。非分页组仍耗约 12.9 分钟，主要受独立 probe 重新编译影响，本次未实现这些构建器的跨运行缓存。

本次交付：

- `mla_reference_pair` 对实际逻辑输入复用双数学结果；dense、非分页、causal 已接通，PA 保留原始描述外层缓存。每次仍执行独立映射、固定源与迁移设备调用和比较。统一入口默认启用内容缓存，关闭/刷新/损坏重算和身份失效由 CPU 工具检查。
- 新固定清单的每行记录覆盖理由；双 dtype、M/N 尾块、16K KV、任务环、有效长度/三布局 used、stride、页边界、空值和生命周期在对应组覆盖。清单及范围见 [design 7.4](design.md#74-统一验收入口)，不代替新增功能、公开协议/Trap/E2E 或阶段完整矩阵。
- 成功组恢复保留完整构建与执行身份，Python 依赖按所选测试的导入闭包核对；无关测试模块和无头脚本不再使数值组失效。原生源码/驱动仍保守覆盖，未知动态导入退回完整闭包。现有 `setup.py` AST 构建器单独识别并保留其源码依赖。新格式不会强行继承旧身份，失败组仍须重跑，性能始终重新采样。

集合证据：`selection-audit.json` 对照 M08b1 的原节点与本次实际 collect-only，证明新清单是对应完整组的固定子集；`full-selection-audit.json` 确认 **17 个原有完整组的选择条件保持不变**，tools 仅增补缓存和依赖检查。`mla_reference`、mask 和两种输入映射的原公式 AST 保持不变。CPU 的独立标量对照、缓存冷算/命中相等、参数与源码失效、错误传播及真实依赖闭包检查均通过。

执行目录分别为 `cpu/`、`device-cold/`、`device-warm/`，每个 attempt 保存节点、XML、实际加载、源码身份及 golden 统计；`frozen-sources.json` 记录冻结实现。汇总复核两遍完整节点一致、全部组回执哈希、生产依赖和每遍 28 份 probe 构建记录；最终两遍期间实现保持冻结。两项源精度限制在当前 causal 清单中再次实测，冷算记录见 `source-precision-limits.json`。先前 PA 双层缓存的工具断言失败已修复为单层缓存并重跑通过；依赖检查误把已知 setup AST 构建器视为未知代码时，主动中断了首次设备尝试，修复并补测试后才冻结重跑。失败/中断 attempt 保留，不计入通过结果。质量工具使用已安装的 `/tmp/mla-m01-precommit` 和 `/tmp/mla-m01-precommit-cache`；初次新建 Go 环境的下载错误另存 `quality-install-failure.log`。

### 既有算法闭环与限制

M08b1 的完整算法闭环在提交 `a21cb45`，原始证据为 `build/mla-m08b1-closeout/validation-complete.json`（2,928 项）及 `acceptance.json`。这些数字是历史记录，不计入本次执行。该步 CPU/sanitizer、完整数值和组件性能的依赖与适用边界从原记录复核；仅有本次快速组通过不证明历史完整矩阵已重跑。

保留两项 FP16 源精度限制：`B=2,Sq=129,Hq=3,Sk=1025`、KV `[17,1025]`、最大值变化 amplitude=4/8 的 O 数学误差 0.015625，迁移与固定源逐元素相同。仍按 `source_precision_limit` 单列，原数学阈值不变，不能记成数学通过。旧记录见 `build/mla-m08b1-closeout/source-precision-limits.json`。M08b1 的 PA 组件 Kernel 相对固定源最大约慢 15.5%，不等于公开 E2E；首次 BSND causal 公开 E2E 仍须由 M08b2 当次完成。

本次数值实现、CANN/CATLASS、构建参数及生产产物未变：Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2、910B3 第 6 卡；CATLASS `769cd40a`、固定 CANN 源 `ca18aee6`。实际扩展仍为本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `f3d8373c94ac3aaf32aa54949d4ed7126adbeb3d960c677ab27abbde6baffd36`。准备命令复核已有构建依赖及产物后复用；环境见 `runtime-initial.json` 和各 attempt 的运行栈。

## 无头接续与唯一下一步

唯一下一步为 **M08b2：BSND 非 PA causal 的 metadata、生产和公开接入**。未变化的数值组件先复核继承；新入口、协议、实际参数传递及适用公开 R=1/4 E2E 当次实测。共享数值或地址改变时选相关 `regression-*` 并补清单未覆盖的边界，完整组合在 M08e 阶段验收，不默认重跑全部历史组。

`scripts/mla_codex_loop.py` 的开发提示已同步上述选择及缓存规则。兼容入口 `--check` 为 ready，pending 为空，成功计数仍为 46；`--dry-run --max-dev-runs 1` 包含新提示并新建会话。检查记录为 `headless-check.json` / `headless-preview.log`，没有调用模型或改写成功计数。旧失败轮仍保持 M08b1 的归档状态。

提交后直接接续：

```bash
python3 /home/zhangqijia/repo/mla_codex_loop.py
```

本次固定清单复现（先 dry-run/collect-only；对比缓存效果时换输出目录，不加 resume）：

```bash
python3 -m tests.mla_validation --output build/mla-validation-speedup/device-warm --suite regression-dense regression-nonpaged regression-paged causal-check --device 6
```

日常按实际影响选择其中的组；`causal-check` 可单独自检，`causal-smoke` 包含昂贵成本边界，不作为每次编辑后的默认检查。`--resume` 只恢复有效成功组，不能跳过失败组内节点；模型会话恢复也不等于测试进程恢复。新输入或参考公式/缓存身份变化仍须冷算。

本次只提交 13 个文件；提交前检查完整差异、链接/表格，执行 `git diff --check` 和 `ci/code_quality.sh --staged`，最终记录见 `acceptance.json` / `quality-final.log`。当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位；证据与缓存位于忽略目录，迁机须复制依赖和产物或重新验证。
