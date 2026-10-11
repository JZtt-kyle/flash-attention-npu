# MLA 当前状态

核对日期：2026-10-11；分支：`feat/mla-cann-migration`；接手基线：`f14aad6`。
本轮完成 **M07e3a：独立 TND PA 静态发射计划**，4,174 项去重 pytest 通过；完整 928 组静态输入通过 ASan/UBSan 和泄漏检查，439 组 BSND 计划与基线逐字段一致。TND PA 前向仍明确拒绝。唯一下一步为 [M07e3b：TND PA 内部生产前向](plan.md#下一步m07e3b)。

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
| 独立静态计划 | BSND/TND PA host 计划 | TND 的 `BuildPagedTndPlan` 已验收；仍无 TND PA 生产 launcher |
| 独立 TND PA attention | TND Q 分配/used、独立 stride | 尚无 TND PA 生产前向；独立 probe 不代表公开能力 |

调用时须注意：

- kvcache/metadata 的 `num_splits` 默认 0，MLA 须显式传 **1**；生成器的 `qkv_dtype` 默认 BF16，须匹配输入。生成器只接受长度 tensor，None/整数便利输入属于公开 kvcache。
- PA 页大小 P 为 **16..1024 的 16 倍数**；逻辑容量 **C=页表列数×P，在 0..16384**。None 表示 C，整数须在 0..C，按 Q 的逻辑 B 物化；物理页数不决定 B 或 C。生成器的 `max_seqlen_k` 须等于 C。
- v6 不绑定 cache 地址、stride、页表或 scale。每次前向都在设备核对当次长度及全部有效页；未用列允许 poison。PA 与非 PA 协议互斥。四路末维连续、外层非负独立 stride、offset、广播和重叠按完整 storage 校验。
- 请求 LSE 时 func/varlen 返回二项，kvcache 返回四项；未 split 的两个 accumulator 为同设备空 tensor。PA 所有有效零 KV 行均为 `O=0、LSE=-inf`，含 C=0/空物理池；非 PA 全局空 KV 的 `+inf` 分支保持不变。
- BSND PA 的专用/通用 fake、公开 fullgraph compile 与真实路径校验一致；编译后的真实调用仍检查 storage。完整 graph replay 留在 M16。公开没有 Q-used 参数；内部通用 `fwd` 可传 Q-used，但不能显式复用其记录。

尚未开放：TND PA 生产前向、causal/window、自动或多 split、192/128 prefill、非零 softcap 和非默认调度选项；910C 未开放或实机验收。追加 KV、内置 rotary、量化、backward 等排除项见 [design 第 4 节](design.md#4-功能边界)。

## 本轮闭环与证据

`BuildPagedTndPlan` 复用完整 TND/PA storage 校验和共享静态资源准备，分离逻辑 B、实际 Tq 与声明 max Q；按实际 Tq 校验总地址、按 `min(Tq,maxQ)` 预算 M，并保留原 max Q 的 v6 context。仅接收静态描述，不读取前缀、used、KV 长度或页表值。BSND/TND 共用 `MlaPagedPlan`，保留原 BSND 类型别名；Kernel/Block/Tile、消费者协议和公开入口未改。

证据位于忽略目录 `build/mla-m07e3a-tnd-paged-plan/`，下表路径相对此目录。包含复现脚本、日志、XML、输入/输出、固定源快照和实际库；本机证据不随 Git 分发，迁机须复制本目录及 manifest 引用的历史固定源产物。

| 核验 | 结果与依据 |
| --- | --- |
| 前置闭环 | 编辑前核对 M07e2c 的 8,280 项历史 pytest、扩展/对象/依赖与固定源性能证据通过，不计本轮测试；`prerequisite.json` |
| CPU | **1,305 项通过**：BSND/TND 计划、全部 P×双 dtype 的固定源预算/v6 消费、Q 分配/used/宽松 max、空值/storage/宽字段/平台边界、PA/TND 布局及相关构建检查；`cpu.xml`、`cpu-command.json`、`cpu/` |
| Sanitizer | **928 组完整静态输入**（BSND 439、TND 489）按布局批量执行，开启 ASan/UBSan、失败即停和泄漏检查，所有字段与 CPU 输出一致、stderr 为空；`sanitizer-batch.json`、`sanitize-batch/` |
| BSND 基线差分 | 从 `f14aad6` 提取原计划和驱动，439 组输入的全部计划字段、workspace、核数、布局及拒绝状态一致；`bsnd-differential.json`、`baseline-bsnd/` |
| 910B BSND PA | **713 项通过**：全部页大小、双 dtype 源/双数学 O/LSE、stride/大地址/空值/记录复用/生命周期；每 dtype 保存 328 组结果，其中 313 组含源对照；`device.xml`、`probes/*/production-accuracy/` |
| 公开入口与普通 FA3 | **1,334 项公开 PA + 822 项普通 FA3/入口回归通过**，含真实/fake/compile、长度更新、storage、生命周期及未接通 TND PA 的明确拒绝；`public.xml`、`ordinary.xml` |
| 构建与加载 | 36 个链接对象、271 个仓库源、6,197 个依赖核验；仅 `mla_paged_plan.o` / `mla_forward.o` 变化，34 个对象含全部设备 Kernel 保持原哈希；`production-load.json`、`production-object-delta.json`、`production-final/` |
| 独立探针复用 | 双 dtype 各 55 源/953 依赖及 v6 producer 22 源核对；唯一变化是实际 include 依赖未消费的计划头，保留原 manifest 与快照差异，复制库并隔离新输出；`probe-reuse-audit.json` |
| 提交质量 | 三份文档链接/锚点、命令/表格、差异与 staged 质量钩子通过，只暂存本步文件；`docs-check.json`、`quality-staged-result.json`、`source-snapshots.json`、`acceptance.json` |

上述 pytest 无失败/跳过；初轮 184 项 smoke 不重复计数。Sanitizer 首轮在沙箱内因 LeakSanitizer 的 ptrace 限制退出，日志保留在 `failed-sanitizer-sandbox/`；沙箱外逐项运行因每进程退出约 4.5 秒，在完整批量矩阵通过后主动停止，40 项部分运行不计闭环测试。该中止记录为 `sanitizer-unbatched-stop.json`，最终批量复现脚本为 `check_sanitizer_batch.py`。

## 验收范围与保留限制

本步只接入独立 TND 静态计划；虽随共享 host 单元构建，仍无 TND PA 生产 launcher 或公开前向。动态前缀/used、记录和有效页由既有 v6 消费者逐次检查，静态成功不代表任意动态长度合法。

未修改设备数值对象，本轮未重测 Kernel 性能或公开 E2E；M07e2c 的生产 Kernel 比值和 72 个适用公开 R=1/4 调用段为历史结果，见其提交和 `build/mla-m07e2c-bsnd-paged-public/acceptance.json`。未重跑完整 TND PA attention、全部历史 Trap、PA metadata/非分页 MLA 全套、普通 FA3 全套或 NPU sanitizer；本轮回归针对共享 host 计划及现有 BSND 生产路径。TND PA 生产精度/性能留在 M07e3b，公开精度与 E2E 留在 M07e3c，910C 未实机验收。

## 环境与构建

Python 3.11.15，torch 2.7.1+cpu、torch_npu 2.7.1，CANN 9.0.0-beta.2；8 张 910B3，使用物理卡 6、进程内设备 0，设备任务串行。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f` 干净；只读 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0` 原有 67 项差异及本仓未跟踪 `kernel_meta/` 保留。环境与前后状态见 `environment.json`、`*-status-before.txt` / `*-status-final.txt`。沙箱内首次驱动查询失败，沙箱外检查通过。

最终扩展实际路径为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `e03589054dc67611025bb965f8cc81048178b55c256650fe5d63d483a3e9d1fe`。构建为 `build_production.py`，实际加载/对象/依赖审计为 `audit_production.py`，证据保存于 `production-final/`；双 dtype 精度使用核对后复制的固定源 probe 库，不覆盖历史结果。

## 唯一下一步与复现

下一步仅为 **[M07e3b](plan.md#下一步m07e3b)**：复用 TND 静态计划和既有 attention，接双 dtype 生产实例与内部前向，完成相应 910B 精度、代表性能和回归；公开 TND PA kvcache 留在 M07e3c。本轮只提交一次，成功后停止。

最小闭环复核（仓库根目录，非 login shell，不发射 NPU）：

```bash
python3 build/mla-m07e3a-tnd-paged-plan/check_acceptance.py
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache \
bash ci/code_quality.sh --staged
git diff --check
git diff --cached --check
```

`run_checks.py` 的闭环模式为 `cpu`、`device`、`public`、`ordinary`；命令与环境保存于各模式 `*-command.json`。`compare_bsnd.py` 比较历史 BSND 计划，`check_sanitizer_batch.py` 从 CPU 全部捕获输入构建独立插桩驱动并验收；`run_device_checks.py` 审计扩展后串行运行三个 NPU 模式。重跑先复制脚本到新证据目录，不覆盖本轮数据；NPU 与 sanitizer 在沙箱外执行，CPU 禁用 torch 自动加载 NPU。
