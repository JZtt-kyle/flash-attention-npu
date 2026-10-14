# MLA 当前状态

当前分支 `feat/mla-cann-migration`，本步起点 `905d343`。**M10c2a BSND 自动 split 任务与归约计划已闭环。** 本步是独立 CPU 算法组件，公开自动 split 尚未接通；既有非 PA BSND 显式 1/2..256 和五布局 mask 能力保留。当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。

## 本步产物与边界

新增 `mla/mla_auto_split_kv_plan.h/.cpp`，将固定 CANN 原生变长 AIC 分区映射为完整 M 直接输出、跨核 FD 部分输出及零成本初始化任务，保留原生 Vector 的 8 行归约所有权。只为 FD 片分配紧凑持久槽，静态上界为 2×AIC 个 M 槽；scratch 仍按物理核复用。计划只持有 O(B+AIC) 的长度/前缀/分区，64 位 M 编号不截断，最终 O/LSE/accumulator 地址保留 BSND 分配间隙，mask 使用实际长度。

静态 accumulator 描述以 AIC 数作为容量上界，区别于各 M 的实际片数；Direct 结果属于第 0 片，未使用片及空行需要后续生产调用者初始化。单核无 accumulator。这个 CPU 描述不开放公开自动返回契约，不能作为 metadata 或设备记录直接使用，设备同步/数值与公开入口由下一步验收。协议见 [design 6.2.13](design.md#6213-bsnd-自动-split-的独立任务与归约计划)。

新增独占 `tests/mla/unit/auto_split_kv_plan/`、CPU 驱动/固定源构建与工具测试。共享数值核、setup、metadata、生产分派、公开接口、CANN 源仓和 CATLASS 均未修改。尚无新公开可调用行为，不注册伪自动 E2E；累计 134 项小输入清单与上步逐项相同，见 `/tmp/mla-m10c2a/selection.json`。没有把旧无 mask 测试当成新算法验收，新组件的源/独立几何覆盖含 causal 与 window。

## 接手、环境与证据

接手工作树干净，宿主进程核对仅有当前无头循环，无遗留 MLA 构建/测试。上步 E2E 的 13 个原始文件、904 个源码及生产 6,349 个依赖/12 个产物逐项完整，见 `/tmp/mla-m10c2a/takeover.json`。本步新增源码/统一入口改变保守依赖身份，累计 E2E 实跑、继承 0；不累计历史单元或性能。

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。设备为第 6 张 910B3。统一构建完成后实际加载仓库 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA-256 `dd89923c6fc0badb44b4c3ede8dc6d66adfe3a418a0bf761b138b36ab14f1458`；运行时与 E2E 的 loaded 记录一致。构建耗时 486.1 秒，运行时核对 20.1 秒，不用旧 `.so` 代替当前构建身份。

本步证据根 `/tmp/mla-m10c2a/`；构建/reference 沿用 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`，`build/` 仅有 `mla-codex-loop/`。原始证据清理后不可继续继承。

## 最近闭环验收

所有组均先 dry-run/collect-only。CPU 和 sanitizer 为同一完整 38 项；18 个具名源/几何组合覆盖原生 FD/非 FD、双 CV 比、单核/最大核、三种 mask、变长分区、空值/尾部，其余检查非法参数、宽整数、溢出、失败原子性和生命周期。未按固定数量填充或截断。

| 本步集合 | 本次通过 | 耗时 |
| --- | --- | --- |
| `auto-split-kv-plan-tools` | 13/13，选择、失败/中断传播、恢复策略、固定源身份和 CPU 退出检查 | 8.9 秒 |
| `auto-split-kv-plan-check` 开发代表 | 4/4，三种 mask 原生 FD 与最大逻辑容量；后续完整集合包含这些节点 | 24.5 秒 |
| `--unit auto_split_kv_plan` | 38/38，源字段/workspace/FA3 shape、独立 tile/归约/持久地址及边界 | 24.5 秒 |
| `auto-split-kv-plan-sanitize` | 同一 38/38，宿主 ASan/UBSan、LeakSanitizer 及退出检查通过 | 86.5 秒 |
| 累计 `e2e` | 134/134，本次 910B 实跑，继承 0 | 28.8 秒 |

工具最终为 `cpu-final/attempt-1791993347934148170/auto-split-kv-plan-tools/`；CPU 为 `cpu-corrected/attempt-1791993417145285030/unit-auto_split_kv_plan/`；sanitizer 为 `sanitize-final/attempt-1791993558345829360/`；设备为 `e2e-final/attempt-1791993448558562700/`。CPU 最大逻辑容量验证 5,368,709,120 个 M 块，仅存少量描述字段，无巨量 tensor 分配。

最终 `final-evidence-check.json` 复核工具/CPU/sanitizer/E2E 的全部原始文件及对应源码、生产 6,349 个依赖/12 个产物、运行时实际加载库，共核对 7,308 个唯一文件，错误 0。普通 CPU 和 sanitizer 各 48 个原始文件完整，驱动输入/输出、固定 Git blob、编译命令、二进制哈希与退出结果均保留。

数学 reference 使用内容缓存：本次 E2E hit 187、miss 94，94 项均成功写入，无 read/write error 或强制 refresh。CPU 代表与模块验收不计算 attention reference，本步设备累计 E2E 只执行一次；不将缓存 miss 写成命中或继承，待测设备始终实际执行。

复现先加 `--dry-run`、`--collect-only`，再执行；输出使用新目录保留当前证据。E2E 需可访问设备的宿主环境，sanitizer 需可正常运行 LeakSanitizer 的环境：

```bash
python3 -m tests.mla --suite auto-split-kv-plan-tools --output /tmp/mla-m10c2a/tools-repro
python3 -m tests.mla --unit auto_split_kv_plan --output /tmp/mla-m10c2a/cpu-repro
python3 -m tests.mla --suite auto-split-kv-plan-sanitize --output /tmp/mla-m10c2a/sanitize-repro
python3 -m tests.mla --suite e2e --device 6 --output /tmp/mla-m10c2a/e2e-repro
```

## 失败处理、未运行与质量

开发代表首次编译发现测试驱动重命名遗漏，已修复并通过代表及完整集合。完整 CPU 首次在单核 shape 的测试预期失败：固定源 oracle 返回两个零 rank `[0,0]` 表示无 accumulator，修正预期后 38 项通过；未修改生产数值或放宽门槛。

沙箱内 E2E collect 无法查询 NPU，改在宿主收集 134 项并完成实跑。沙箱 sanitizer 在进程退出时触发 LeakSanitizer 的 ptrace 环境限制，该次 `cpu-corrected/` sanitizer 记录不算通过；宿主 `sanitize-final/` 保留完整检测重跑同一集合并通过。未关闭泄漏检测，也未将 37 项断言通过冒充 sanitizer 闭环。早期失败和最终证据分目录保留。

最终必要集合全部通过，无未解决失败。公开自动设备数值/Trap/生命周期、平均公开性能、普通 FA3 独立组、历史单元/性能、full-matrix 和 910C 未运行，不列为通过或继承；本步不改生产调用和数值，相关公开回归由累计 E2E 完成。三份文档按新增 CPU 协议与提交边界核对；链接、仓库 pre-commit 和 `git diff --check` 通过，证据见 `doc-links-final.json`、`quality-final.log`。

## 唯一下一步

[M10c2b BSND 自动 split 生产接入](plan.md#唯一下一步-m10c2bbsnd-自动-split-生产接入)。先核对本步进程和完整证据、实际依赖，再明确自动请求返回形状与静态上界的适配，将 metadata、设备变长 partial/8 行 merge、公开 real/fake/compile 和该能力 E2E/性能整体接通。本轮只提交 M10c2a 一次，提交后停止，不开始下一步。
