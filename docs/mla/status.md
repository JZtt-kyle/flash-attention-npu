# MLA 当前状态

核对日期：2026-10-12；分支：`feat/mla-cann-migration`；接手基线：`cec845b`。
本轮完成 **M07e3b：TND PA 内部生产前向**，8,936 项去重 pytest 通过，双 dtype 固定源 Kernel 性能对照完成。**公开 TND PA 前向仍明确拒绝**；唯一下一步为 [M07e3c：TND 只读 PA 公开 kvcache](plan.md#下一步m07e3c)。

本页记录当前能力、最近闭环证据和唯一下一步；契约见 [design](design.md)，迭代边界见 [plan](plan.md)。本轮闭环提交用 `git log -1 -- docs/mla/status.md` 定位。

## 当前可调用能力

共同边界：**910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、无梯度、无 mask、`num_splits=1`**。Q/K 末维为 64，qv/V 为 512，四路同 dtype、同 NPU 设备。单样本逻辑 KV 容量上界 16384；TND 总 KV 可超过 16384。完整边界见 [design 6.2–6.4](design.md#62-调度字段与边界)。

| 层次 | 已接通能力 | 记录与边界 |
| --- | --- | --- |
| 公开前向 | `flash_attn_func`，BSND Q/KV | 整容量，内部 v1 |
| 公开前向 | `flash_attn_varlen_func`，TND Q/KV | 两轴分配前缀；无 used 时内部 v3，任一 used 存在时内部 v5 |
| 公开只读 cache | `flash_attn_with_kvcache`，BSND Q + BSND KV | 无长度且无记录走 v1，否则 v2；支持显式 v2 复用 |
| 公开只读 cache | `flash_attn_with_kvcache`，TND Q + BSND KV | Q 使用完整分配段，生成或显式复用 v4 |
| **公开只读 PA cache** | `flash_attn_with_kvcache(page_table=...)`，BSND Q + ND PA KV | **内部生成或显式复用 v6；None/整数/tensor KV 长度；完整 Q 分配段** |
| 公开 metadata | `get_scheduler_metadata` | 非 PA：BSND v2、TND v4；指定 `page_size`：BSND/TND v6 `int32[408+5B]` |
| 内部前向 | 非 PA 三布局 Q-used；BSND PA Q-used | 非 PA v5、PA v6；Q-used 每次生成记录，不开放显式复用 |
| **内部 TND PA 前向** | `_mla_paged_tnd_forward`，完整 Q 分配/内部 Q-used | 双 dtype 生产实例；无 Q-used 时生成或复用 v6，有 Q-used 时逐次生成；公开入口未开放 |
| 静态计划 | BSND/TND PA host 计划 | 两种计划均由相应生产前向使用；静态成功不代表动态长度合法 |

调用时须注意：

- kvcache/metadata 的 `num_splits` 默认 0，MLA 须显式传 **1**；生成器的 `qkv_dtype` 默认 BF16，须匹配输入。生成器只接受长度 tensor，None/整数便利输入属于公开 kvcache。
- PA 页大小 P 为 **16..1024 的 16 倍数**；逻辑容量 **C=页表列数×P，在 0..16384**。None 表示 C，整数须在 0..C，按 Q 的逻辑 B 物化；物理页数不决定 B 或 C。生成器的 `max_seqlen_k` 须等于 C。
- v6 不绑定 cache 地址、stride、页表或 scale。每次前向都在设备核对当次长度及全部有效页；未用列允许 poison。PA 与非 PA 协议互斥。四路末维连续、外层非负独立 stride、offset、广播和重叠按完整 storage 校验。
- 请求 LSE 时 func/varlen 返回二项，kvcache 返回四项；未 split 的两个 accumulator 为同设备空 tensor。PA 所有有效零 KV 行均为 `O=0、LSE=-inf`，含 C=0/空物理池；非 PA 全局空 KV 的 `+inf` 分支保持不变。
- TND 专用内部前向要求同设备连续 `int32[B+1]` 分配前缀及声明 max Q；KV 长度只接 None/tensor。max Q 可宽于实际分配；Tq=0 且 B/maxQ/C 非零时，按公开 v6 的全核策略校验，workspace 为零。
- BSND PA 的专用/通用 fake、公开 fullgraph compile 与真实路径校验一致；编译后的真实调用仍检查 storage。完整 graph replay 留在 M16。公开没有 Q-used 参数；内部通用 `fwd` 可传 Q-used，但不能显式复用其记录。

尚未开放：公开 TND PA 前向、causal/window、自动或多 split、192/128 prefill、非零 softcap 和非默认调度选项；910C 未开放或实机验收。追加 KV、内置 rotary、量化、backward 等排除项见 [design 第 4 节](design.md#4-功能边界)。

## 本轮闭环与证据

`PagedTndForward` 复用 `BuildPagedTndPlan`、共同 PA host 校验/提交流程及已验收的 `MlaPagedTnd`；新增 FP16/BF16 C1V2 生产实例和专用内部算子。Q 始终按分配前缀定位，Q-used 只限制有效范围；前缀、长度、页表及全部 storage 保持异步生命周期。无 Q-used 时生成或显式复用 v6，有 Q-used 时逐次生成并拒绝显式复用。设备消费者、协议、Kernel / Block / Tile 数值算法和公开入口未改。

实机复现并修复 Tq=0、声明 max Q>0 时的 v6 复用发射核数不匹配：生产适配器采用公开生成器按 B/maxQ/C 确定的全核校验策略，保留静态计划的零 workspace。复现记录为 `empty-reuse-before.json` / `.log`；修复前构建和 smoke 保留在 `before-empty-query-fix/`，不计本轮闭环。最终新增 16 项空 Q/宽松 max 用例，连同 12 项非空宽松 max 专项全部通过，并包含在最终完整矩阵内。

证据位于忽略目录 `build/mla-m07e3b-tnd-paged-production/`，下表路径相对此目录。保存复现脚本、命令/日志/XML、输入/输出、固定源快照及实际库；本机证据不随 Git 分发，迁机须复制此目录及 manifest 引用的依赖。

| 核验 | 结果与依据 |
| --- | --- |
| 前置闭环 | 编辑前核对 M07e3a 的 4,174 项历史 pytest、928 组 sanitizer、439 组 BSND 差分及实际扩展/对象/依赖通过，不计本轮测试；`prerequisite.json` |
| CPU | **1,305 项通过**：BSND/TND 静态计划、全部 P 双 dtype 固定源预算/v6 消费、布局/storage/空值/字段/平台边界及实例生成/依赖检查；`cpu.xml`、`cpu-command.json`、`cpu/` |
| 910B TND PA | **784 项通过**：全部 64 种 P、双 dtype 源/双数学 O/LSE、分配/used、独立 stride/宽地址、空值/宽松 max、记录复用、动态前缀/长度及生命周期；每 dtype 保存 344 组结果，其中 328 组含源对照；`device.xml`、`probes/*/tnd-production-accuracy/` |
| 真实 Trap | **40 项通过**：双 dtype 的过期/非法前缀、used、KV 长度、有效页、空 Q/空池/零 KV、协议与记录损坏；每个独立进程均确认真实 Trap；`trap.xml`、`production/*/trap-*/` |
| BSND PA | **713 项通过**：共享 host 前向全页大小源/双数学、stride/storage/空值/复用/生命周期；每 dtype 保存 328 组，其中 313 组源对照；`bsnd.xml`、`probes/*/production-accuracy/` |
| 其他回归 | **1,098 项非分页/相关独立 TND PA、2,840 项 PA metadata、1,334 项公开 BSND PA、822 项普通 FA3/入口检查通过**；公开 TND PA 在 Python/通用 dispatcher/raw C++/fake 继续拒绝；对应模式 XML/日志 |
| 生产 Kernel 性能 | 双 dtype 各 12 个固定代表用例、相同调度/核数的固定源 PA/non-FD 对照；P=16/112/128/256/528/1024，短/长 KV、零 Q/KV、Q-used 和 PA02；原始样本及逐例结果见 `profile-fp16/`、`profile-bf16/` |
| 构建与加载 | **38 个链接对象、274 个仓库源、6,200 个依赖核验**；新增两个 TND 实例，旧对象仅 `mla_forward.o` 变化，其余 35 个哈希一致；`production-load.json`、`production-object-delta.json`、`production-final/` |
| 独立对照构建 | 双 dtype probe 重新构建，各 55 源/953 依赖与固定 CANN blob 校验，v6 producer 22 源核对；`probe-build-command.json`、`probe-audit.json`、`probes/`、`producer/` |
| 提交质量 | 三份文档链接/锚点/命令/表格、差异及 staged 质量钩子通过，仅暂存本步 15 个文件；`docs-check.json`、`quality-staged-result.json`、`source-snapshots.json`、`acceptance.json` |

最终 pytest 无失败/跳过；修复前 56 项 smoke 和最终 28 项专项不重复计数。CPU 之后只修改了未被 CPU 套件读取的 host 前向、新生产测试和 Trap 驱动；原快照差异在 `acceptance.json` 单列，CPU 实际依赖及最终设备运行源码均核验。

生产 Kernel/同调度源 Kernel 比值：FP16 **0.995..1.306**（11/12 大于 1），BF16 **1.018..1.173**（12/12 大于 1）；原生 FD 各 6/6 例单列。迁移计时含 v6/前缀/当次有效页校验，源 Q-used 压紧和 stride 物化在计时外；AICPU 与内部 wall 另报，不能据此宣称公开 E2E 不劣化。profiler 原始 `op_summary` 与 text Kernel CSV 的双 dtype 各 1,800 条设备任务逐条核对通过；两组 framework API 数据库导出均因 `Failed to find node launch str id` 失败，不用于这些计时，也不标为通过，见 `profile-export-audit.json`。

## 验收范围与保留限制

本步只接通专用 TND PA 内部生产前向，未开放其 fake/compile 或公开 kvcache。公开长度便利输入、公开双 dtype 精度和 R=1/4 E2E 留在 M07e3c。未重跑全部历史 Trap、非分页 MLA/普通 FA3 全套、完整图重放、NPU sanitizer 或 CPU sanitizer；本轮 CPU 静态算法及既有设备数值算法未改。910C 未开放或实机验收。

## 环境与构建

Python 3.11.15，torch 2.7.1+cpu、torch_npu 2.7.1，CANN 9.0.0-beta.2；8 张 910B3，使用物理卡 6、进程内设备 0，设备任务串行。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f` 干净；只读 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0` 原有 67 项差异及本仓未跟踪 `kernel_meta/` 保留。环境、占用与前后状态见 `environment.json`、`*-npu-before.log`、`*-status-before.txt` / `*-status-final.txt`。沙箱内驱动查询失败，沙箱外检查与实机验证通过；首次失败见 `npu-sandbox-failure.log`。所有命令使用非 login shell。

最终实际扩展为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `f9984021e81d33f10622465601a107d6d02f8ff89b36ecdc91f23d8136e95097`。构建脚本 `build_production.py`，实际加载/对象/依赖审计为 `audit_production.py`；最终快照位于 `production-final/`。

## 唯一下一步与复现

下一步仅为 **[M07e3c](plan.md#下一步m07e3c)**：接入 TND 只读 PA 公开 kvcache，补齐 None/整数/tensor 长度、通用入口与 fake/compile、公开精度和 R=1/4 E2E。本轮只提交一次，成功后停止。

最小闭环复核（仓库根目录，非 login shell，不发射 NPU）：

```bash
python3 build/mla-m07e3b-tnd-paged-production/check_acceptance.py
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache \
bash ci/code_quality.sh --staged
git diff --check
git diff --cached --check
```

`run_checks.py` 的闭环模式为 `cpu`、`device`、`trap`、`bsnd`、`nonpaged`、`metadata`、`public`、`ordinary` 和 `profile-fp16/bf16`；命令与环境保存于各模式 `*-command.json`。`run_device_checks.py` 核对扩展后串行执行给定模式，失败即停；`audit_profile_exports.py` 核对原始性能样本。重跑先复制脚本到新证据目录，不覆盖本轮数据；NPU 在沙箱外执行，CPU 禁用 torch 自动加载 NPU。
