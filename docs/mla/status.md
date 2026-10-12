# MLA 当前状态

核对日期：2026-10-12；分支：`feat/mla-cann-migration`；接手基线：`3c7dc0f`。
本轮完成 **M07e3c：TND 只读 PA 公开 kvcache**，**M07 已完成**。9,614 项去重 pytest 通过；双 dtype、R=1/4 公开 E2E 的已迁移非 FD 用例逐例通过 ≤1.0 门槛。唯一下一步为 [M08：512+64 右下 causal](plan.md#下一步m08)，进入前先拆为独立验收的小步。

本页只记录当前能力、最近闭环证据和唯一下一步；契约见 [design](design.md)，迭代顺序见 [plan](plan.md)。本轮闭环提交用 `git log -1 -- docs/mla/status.md` 定位。

## 当前可调用能力

共同边界：**910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、无梯度、无 mask、`num_splits=1`**。Q/K 末维为 64，qv/V 为 512，四路同 dtype、同 NPU 设备。单样本逻辑 KV 容量上界 16384；TND 总 KV 可超过 16384。完整边界见 [design 6.2–6.4](design.md#62-调度字段与边界)。

| 层次 | 已接通能力 | 记录与边界 |
| --- | --- | --- |
| 公开前向 | `flash_attn_func`，BSND Q/KV | 整容量，内部 v1 |
| 公开前向 | `flash_attn_varlen_func`，TND Q/KV | 两轴分配前缀；无 used 时 v3，任一 used 存在时 v5 |
| 公开只读 cache | `flash_attn_with_kvcache`，BSND Q + BSND KV | 无长度且无记录走 v1，否则 v2；可显式复用 v2 |
| 公开只读 cache | `flash_attn_with_kvcache`，TND Q + BSND KV | Q 完整分配段，生成或显式复用 v4 |
| **公开只读 PA cache** | `flash_attn_with_kvcache(page_table=...)`，BSND/TND Q + ND PA KV | **None/整数/tensor KV 长度，内部生成或显式复用 v6；Q 完整分配段** |
| 公开 metadata | `get_scheduler_metadata` | 非 PA：BSND v2、TND v4；指定 `page_size`：BSND/TND v6 `int32[408+5B]` |
| 内部 Q-used 前向 | 非 PA 三布局，BSND/TND PA | 非 PA v5、PA v6；每次生成记录，拒绝 Q-used 显式复用 |
| fake/compile | BSND/TND PA 专用/通用 fake 与公开 fullgraph compile | 真实执行重新核对完整 storage；完整图重放留在 M16 |

调用时须注意：

- kvcache/metadata 的 `num_splits` 默认 0，MLA 须显式传 **1**；生成器的 `qkv_dtype` 默认 BF16，须匹配输入。生成器只接受长度 tensor，None/整数便利输入属于公开 kvcache。
- PA 页大小 P 为 **16..1024 的 16 倍数**；逻辑容量 **C=页表列数×P，在 0..16384**。None 表示 C，整数须在 0..C，按逻辑 B 物化；TND 的 B 来自 Q 前缀元素数减一，不是 Tq 或物理页数。生成器的 `max_seqlen_k` 须等于 C。
- TND PA 需要同设备连续 `int32[B+1]` 的 `cu_seqlens_q` 和 `max_seqlen_q`；Q 始终按完整分配前缀定位。max Q 可宽于观测值，Tq=0 且 B/maxQ/C 非零时仍使用公开 v6 的全核校验策略、零 workspace。
- v6 不绑定 cache 地址、stride、页表或 scale；每次前向都在设备核对当次前缀、长度及全部有效页，未用列允许 poison。长度/前缀更新后须重生成记录。PA 与非 PA 协议互斥。
- 四路末维连续，外层独立非负 stride、offset、广播和重叠按完整 storage 校验。前缀、长度、页表和记录在异步执行期间须保持只读和存活。
- 请求 LSE 时 func/varlen 返回二项，kvcache 返回四项；未 split 的两个 accumulator 为同设备空 tensor。TND O/LSE 为 `[Tq,Hq,512]` / `[Hq,Tq]`。PA 所有有效零 KV 行均为 `O=0、LSE=-inf`，含 C=0/空物理池；非 PA 全局空 KV 的 `+inf` 分支保持不变。
- 公开没有 Q-used 参数；通用内部 `fwd` 和专用 PA 算子可传 Q-used，但不能显式复用其记录。

尚未开放：causal/window、自动或多 split、192/128 prefill、非零 softcap 和非默认调度选项；910C 未开放或实机验收。追加 KV、内置 rotary、量化、backward 等排除项见 [design 第 4 节](design.md#4-功能边界)。

## 本轮闭环与证据

公开 TND PA kvcache 直接进入已有 `PagedTndForward`；整数便利长度先验证几何和范围再按逻辑 B 物化。通用 Python/raw C++ 分派接同一前向，专用/通用 fake 补齐 TND 静态几何、前缀和完整 storage 校验。没有改变设备消费者、metadata 协议、Kernel / Block / Tile 数值算法，也没有调用 CANN attention 作为生产回退。

证据位于忽略目录 `build/mla-m07e3c-tnd-paged-public/`，下表路径相对此目录。保存复现脚本、命令/日志/XML、精度输入/输出、源码快照、实际扩展与固定源对照库。证据不随 Git 分发；迁机须复制此目录及 manifest 引用的历史构建、工具链和固定源依赖。

| 核验 | 结果与依据 |
| --- | --- |
| 前置闭环 | 编辑前复核 M07e3b 的 8,936 项历史 pytest、38 个对象/274 个仓库源/6,200 个依赖及扩展哈希通过，不计本轮测试；`prerequisite.json`、`prerequisite-acceptance.json` |
| 构建与 CPU | **2 项通过**：实例生成与设备依赖陈旧检查；重建 38 个对象的扩展，仅 `flash_api.o`、`mla_forward.o` 变化，其余 36 个哈希一致；`cpu.xml`、`production-load.json`、`production-object-delta.json`、`production-final/` |
| TND PA 公开前向 | **2,145 项通过**：全部 64 种 P、双 dtype 固定源/双数学 O/LSE、None/整数/tensor、各真实/fake 入口、compile、独立 stride、宽地址、完整 storage、空值/宽松 max、复用及生命周期；`public.xml`、`probes/*/tnd-public-accuracy/` |
| 真实 Trap | **46 项通过**：双 dtype 的非法/过期前缀与 KV 长度、全部有效页、空 Q/空池、协议/记录损坏；逐例独立进程，确认真实 Trap；`trap.xml`、`kvcache/*/trap-*/` |
| 相关回归 | **1,877 项 BSND PA 公开、784 项 TND PA 内部、2,840 项 PA metadata、1,098 项非分页/独立 TND PA、822 项普通 FA3/入口检查通过**；各模式 XML/日志 |
| 对照组件 | 数值组件及依赖未改；复用并逐文件核验上一轮双 dtype probe 与 v6 producer，复制到本步目录保存新精度结果；`probe-reuse-audit.json`、`probes/`、`producer/` |
| 原生 CANN | 固定源独立构建 TND PA 非 FD/FD 双 dtype 实例，保留 host 库哈希及实际注册/launch key；双 dtype 各 12 例，其中 6 非 FD、6 FD；`native-package.json`、`native-dispatch-proof.json`、`native-*.log` |
| 公开 E2E | 双 dtype 固定 12 例、P=16/112/128/256/528/1024，短/长 KV、单/多 Q、零长度及 PA02；R=1/4、预热 5、5 轮×20 wall 样本、相同 CPU 亲和性；`public-profile-fp16/`、`public-profile-bf16/` |
| 共享性能脚本 | BSND 路径双 dtype 各 1 例固定源/双数学精度预检通过，不含计时；`adapter-smoke-command.json`、`adapter-smoke-fp16/`、`adapter-smoke-bf16/` |
| 提交质量 | 三份文档链接/锚点/命令/表格、差异与 staged 质量钩子通过；`docs-check.json`、`quality-staged-result.json`、`source-snapshots.json`、`acceptance.json` |

最终 pytest 无失败/跳过；154 项前置 smoke 是完整矩阵子集，不重复计数。CPU 之后补充公开 TND 用例；部分 pytest 启动后，仅调整它们不导入的性能驱动，使同一输入的双数学 reference 计算一次并分别校验双方。未被 pytest 使用的快照差异在 `acceptance.json` 单列，前向/校验代码均为最终版本。初次 FP16 性能运行在 CPU 预检阶段主动中止，未产生计时样本，归档在 `before-reference-reuse/`；最终双 dtype 性能与原生分派均重新运行。

公开已迁移非 FD 用例逐例 E2E/固定源 CANN 比值：FP16 **0.348..0.683**、BF16 **0.353..0.677**，包括 R=1/4 的复用、生成加前向和内部生成；每种 dtype 36 个调用段均 ≤1.0。原生 FD 每种 dtype 各 6 例保留原始性能并单列，不计入当前非 FD 门槛。源 TND Q 使用完整分配段，累计 Q 尾值来自测试输入；KV 长度保持非累计。每次必要的 Q/cache/页表物化、输出分配和完成同步计入源 E2E。metadata 生成、设备 Kernel 与转换耗时分别报告；profiler 原始 `op_summary` 与 text Kernel CSV 逐任务核对，framework API 数据库导出未报告错误，详情见 `acceptance.json`。

## 验收范围与保留限制

本步只完成 M07e3c 公开接入及相关回归。完整图重放留在 M16；未重跑全部历史 Trap、全部非分页/普通 FA3、NPU sanitizer 或 CPU sanitizer。设备数值组件和静态计划算法未改，上一轮同调度源 Kernel 对照通过对象/依赖哈希复核保留，不重复标为本轮实测。910C 未开放或实机验收。

## 环境与构建

Python 3.11.15，torch 2.7.1+cpu、torch_npu 2.7.1，CANN 9.0.0-beta.2；8 张 910B3，公开 TND 精度及性能使用物理卡 6；相关回归和 Trap 使用空闲物理卡 7、CPU 136–143，与性能驱动 CPU 128–135 隔离；各卡进程内设备为 0、卡内任务串行。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f` 干净；只读 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0` 原有 67 项差异及本仓未跟踪 `kernel_meta/` 保留。环境及前后状态见 `environment.json`、`regression-environment/environment.json`、`*-npu-before.log`、`*-status-before.txt` / `*-status-final.txt`。

首次环境探测命令覆盖 PYTHONPATH 导致缺少 `tbe`，保留继承环境后通过。CANN opc 在沙箱中因 `psutil.net_if_addrs` 权限失败，沙箱外重新构建通过，失败日志保留为 `native-*-sandbox-failure.log`；首次原生采样在包构建完成前被前置检查阻止，完成打包后才运行。实机校验及原生 opc 构建在沙箱外执行，CPU 禁用 torch 自动加载 NPU；全部命令使用非 login shell。

最终实际扩展为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `6734d89388dfd4c6368bac932d3b41471860f3219198253a3e3dd2b003a4bbc9`。构建/加载审计脚本为 `build_production.py`、`audit_production.py`；实际扩展、对象和依赖快照在 `production-final/`。

## 唯一下一步与复现

下一步仅为 **[M08](plan.md#下一步m08)**：先拆分 512+64 右下 causal 的独立验收步骤，再按已确认范围推进。本轮只提交一次，成功后停止。

最小闭环复核（仓库根目录，非 login shell，不发射 NPU）：

```bash
python3 build/mla-m07e3c-tnd-paged-public/check_acceptance.py
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache \
bash ci/code_quality.sh --staged
git diff --check
git diff --cached --check
```

`run_checks.py` 的闭环模式为 `cpu`、`public`、`trap`、`bsnd`、`device`、`metadata`、`nonpaged`、`ordinary`、`native-fp16/bf16`、`public-profile-fp16/bf16` 和 `adapter-smoke`，命令与环境保存于各模式 `*-command.json`。`run_device_checks.py` 串行执行给定模式，失败即停；`native_audit.py` 核对原生分派。重跑先复制脚本到新证据目录，不覆盖本轮数据，并先核对源码、扩展和对象哈希。
