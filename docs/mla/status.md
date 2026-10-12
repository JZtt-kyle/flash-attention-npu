# MLA 当前状态

**M08b1 已完成。** 分支 `feat/mla-cann-migration`。独立 BSND causal 组件完成固定源/双数学对照、共享回归和逐例组件性能；公共 causal 接入为 [plan 的唯一下一步 M08b2](plan.md)。精度冲突按用户确认的源逐元素一致优先级验收，条件见 [design 第 7 节](design.md#7-验收口径)。本次提交后停止，不启动 M08b2。

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

## 最近一次闭环证据

证据根目录 **`build/mla-m08b1-closeout/`**；汇总为 `validation-complete.json`，最终质量及无头接续记录汇入 `acceptance.json`。本次 **2,928 项通过，0 项失败、0 项跳过、0 项必跑未运行**；历史 CPU/sanitizer 证据另外继承 136 次执行。代表组 20 项是完整 causal 的子集，不重复计数。

| 检查 | 本次通过 | 耗时（秒） | 证据位置 |
| --- | ---: | ---: | --- |
| 工具 | 99 | 10.79 | 独立 tools attempt / `tools` |
| 独立 causal | 324 | 1221.89 | 首次完整 attempt / `causal-attention` |
| 共享 dense | 547 | 2903.72 | 最终设备 attempt / `shared-dense` |
| 共享非分页 | 776 | 1224.15 | 最终设备 attempt / `shared-nonpaged` |
| 共享分页 | 352 | 2480.62 | 最终设备 attempt / `shared-paged` |
| 普通 FA3 | 822 | 27.97 | 最终设备 attempt / `ordinary` |
| causal 组件性能 | 2 | 251.56 | 最终设备 attempt / `causal-profile` |
| 共享组件性能 | 6 | 369.89 | 最终设备 attempt / `shared-profile` |

- 独立 tools attempt：`build/mla-m08b1-closeout-tools/attempt-1791789418929188940/`，原 69 项保留，新增源优先比较 20 项及无头循环 10 项。
- 首次完整 attempt：`build/mla-m08b1-closeout/attempt-1791791056385446200/`；最终设备 attempt：`build/mla-m08b1-closeout/attempt-1791792938821447230/`。各组保存命令、完整节点、XML、日志及实际加载身份；两次运行日志分别为 `validation.log` 和 `remaining-validation.log`。
- 代表与成本边界：`build/mla-m08b1-closeout-smoke/attempt-1791789569820075770/`，20 项通过，506.73 秒，包含双 dtype 最大 reference 和原最大值变化输入。
- 集合复核：`selection-audit.json` / `remaining-selection-audit.json`。没有删减输入或性能形状；仅将 dense 的 12 个边界节点改名，输入和顺序保留。

首次共享 dense 在 173 项通过后遇到旧空容量拒绝断言；M04c4 已支持空 batch/Q/KV，本次按既有契约修正为检查单核、零 tile/端点/workspace，零 heads 和越界仍拒绝。完整 547 项重跑通过。修改仅在该测试函数，causal 使用的共享定义及其余源码不变；`retained-causal.json` 核对全部 871 个原始证据文件，保留本次已实测的 324 项结果，没有改写统一入口的恢复身份。初始工具收集的同名 `scripts` 包冲突已改为明确文件加载；沙箱内设备收集失败后在设备可见环境完整重收集，失败日志均保留。

### 精度结果

完整 causal 记录包含 **646 个 O/LSE 数学比较通过、2 个源精度限制**；条目数与 pytest 节点数分别统计。两个限制均为 FP16、`B=2,Sq=129,Hq=3,Sk=1025`、有效 KV `[17,1025]`、最大值变化下 amplitude=4/8 的 O：最大数学误差均为 0.015625，迁移与固定源逐元素相同。配置、原阈值、双数学误差、输入/输出 `.pt` 路径及哈希见 `source-precision-limits.json`；逐次记录在 `accuracy.jsonl`。

这些结果单列为 `source_precision_limit`，不记为数学通过。`tests/mla_source_compare.py` 保持原数学阈值，拒绝源不一致、NaN、shape/inf 契约错误；FA3 空行 LSE 适配先独立检查再归一化。历史数学失败及原失败的 `build/mla-m08b1/acceptance.json` 保持原样，未改成通过。

### 组件性能

固定矩阵共 **158 个逐例对照**，原始样本、调度、Kernel/raw CANN CSV 和逐例汇总均保留在两个 profile 组中；汇总索引为 `component-performance.json`。双方同调度，预热 5 次、五轮 wall 20 次/kernel 10 次，固定相同 CPU 亲和性；`performance-isolation.json` 核对全部 16 份设备前后快照，第 6 卡健康且无其他并发进程。

下表为迁移/固定源的当次中位数耗时比。Kernel 最大比值约 **1.155**，逐例差异如实保留；预分配 probe 调用耗时是组件诊断，公开 R=1/4 E2E 仍须在 M08b2 接入时按 design 实测，不能据此宣称公开不劣化。

| 组件与 dtype | 对照数 | Kernel 比值范围 | Kernel 比值 >1 的例数 | probe 调用比值范围 |
| --- | ---: | ---: | ---: | ---: |
| causal FP16 | 29 | 0.892–1.062 | 5 | 0.973–1.014 |
| causal BF16 | 29 | 0.894–1.055 | 5 | 0.969–1.020 |
| dense FP16 | 26 | 0.907–1.048 | 7 | 0.970–1.030 |
| dense BF16 | 26 | 0.740–1.000 | 1 | 0.928–1.008 |
| BSND PA FP16 | 12 | 1.009–1.155 | 12 | 1.002–1.034 |
| BSND PA BF16 | 12 | 1.014–1.145 | 12 | 1.005–1.023 |
| TND PA FP16 | 12 | 1.008–1.148 | 12 | 1.008–1.031 |
| TND PA BF16 | 12 | 1.004–1.142 | 12 | 1.003–1.027 |

### 依赖和运行环境

本次代码已固定：最初 `frozen-sources.json`，边界测试修正后 `final-frozen-sources.json`（809 个文件）；快照在 `source-snapshot/`。最终汇总逐文件核对执行证据及 36 份实际独立 probe 构建记录，12.37 秒。工具组故意损坏身份的模拟 fixture 不作为真实 probe 证据，相关失败/失效行为由工具测试验证。

`inheritance.json` 修正后复核通过，3.36 秒：6,318 个生产依赖、11 个产物和 230 个实际加载库匹配；复用的 `.so` 与本次源码构建身份一致。M08a 两组各 68 项、共 136 次 CPU/ASan/UBSan 执行按源码、工具链及每组 98 个证据文件继承，原目录 `build/mla-m08a/attempt-1791780656927232530/`。这些历史执行不计入本次 2,928 项。

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2、910B3 第 6 卡；CATLASS `769cd40a`、固定 CANN 源 `ca18aee6`。实际扩展为本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `f3d8373c94ac3aaf32aa54949d4ed7126adbeb3d960c677ab27abbde6baffd36`。运行栈见 `runtime-initial.json` 及最终 attempt 的 runtime 记录。

## 无头接续与唯一下一步

循环实现位于 `scripts/mla_codex_loop.py`，原 `/home/zhangqijia/repo/mla_codex_loop.py` 已保留为兼容入口；旧脚本备份在证据根目录。正常结束无提交会输出最终说明并记为 blocked，原因处理后用 `--retry-blocked` 明确恢复；网络/进程中断后普通重启恢复同一会话。工具测试使用隔离 Git 仓库和模拟 CLI，不调用模型。

旧失败轮 `build/mla-codex-loop/20261012-045849-development-ycpv3j1x/` 已按“本交互收尾替代”归档，记录为其中的 `retired.json`；14 个原有日志/结果文件全部保持原样。pending 已清空，成功计数仍为 46，不把交互收尾计为无头成功。实际兼容入口的 `--check` 返回 ready，CLI 0.162.1 已登录；`--dry-run --max-dev-runs 1` 确认下一轮新建会话，不恢复旧失败会话。证据为 `headless-check-after.json`、`headless-preview.log`、`headless-preserved-files.json`，检查均在本次提交前执行，实际启动时重新读取当前 HEAD。

本次提交后可直接运行：

```bash
python3 /home/zhangqijia/repo/mla_codex_loop.py
```

唯一下一步为 **M08b2：BSND 非 PA causal 的 metadata、生产和公开接入**；具体边界见 plan。本步独立组件和组件性能不能代替下一步真实调用、协议校验及公开 E2E。

完整复现使用版本管理中的统一入口，先核对 dry-run/collect-only，再运行：

```bash
python3 -m tests.mla_validation --output build/mla-m08b1-closeout --suite tools causal-attention shared-dense shared-nonpaged shared-paged ordinary causal-profile shared-profile --device 6
```

中断恢复先检查已有进程、锁及依赖；只对未完成或失效部分运行，`--resume` 仅继承有效成功组，性能重新采样。本次 causal 在无关 dense 测试修正后的保留依据已单列，不能靠改写旧 identity 使恢复命中。

本次只暂存 22 个本步文件；提交前核对差异、链接与表格，执行 `ci/code_quality.sh --staged` 和 `git diff --check`。质量记录见 `quality-before-handoff.log` / `quality-final.log` 及 `acceptance.json`。闭环提交可用 `git log -1 -- docs/mla/status.md` 定位；原始日志/产物在 Git 忽略的 build 中，迁机须复制真实依赖和证据或重新验证。
