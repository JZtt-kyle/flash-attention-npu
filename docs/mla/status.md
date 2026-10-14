# MLA 当前状态

当前分支 `feat/mla-cann-migration`，本步起点 `84f36a1`。**M10c1b BSND 显式 split 生产接入已闭环。** 非 PA BSND 512+64、FP16/BF16 的显式 `num_splits=2..256` 已开放；保留既有单 split 和五布局 mask 能力。当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。

## 本步产物与边界

新增 v17 BSND split metadata 的 CPU/AICPU 生成及设备独立校验，记录原分配/有效长度、窗口、S 与 64 位紧凑 M 前缀；复用 M10c1a 计划，持久 partial 槽不随物理核复用。partial 多任务/四路 stride 与 FD merge 在同一 stream 分两次发射，内部 Q-used 保留原始 mask 几何与分配输出地址。生产 host、FP16/BF16 实例、公开 func/只读 cache、真实 FP32 accumulator 与 fake/compile 已接通。无 mask、causal、有限/单侧 window 均通过新路径验收。自动 split、TND/混合和 PA split 继续拒绝。协议见 [design 6.2.12](design.md#6212-bsnd-显式-split-生产适配)。

模块单元归 `tests/mla/unit/split_kv_bsnd/`，公开专项归 `e2e/split_kv_bsnd/`。累计小输入 E2E 原 119 项全部保留，新增该能力 15 项，共 134 项；清单核对在 `/tmp/mla-m10c1b/selection-final.json`。性能预先固定 8 个输入、双 dtype 共 16 配置，保留既有 R=1/4、5 次预热、5 轮 20 wall/10 profiler 样本及平均门槛；partial/merge 原始时间分别保存，输入身份包含 S。CANN 源仓与 CATLASS 未修改。

## 接手、环境与证据

接手工作树干净，无遗留 MLA 构建/测试。上步 E2E 的 13 个原始文件、857 个源码及生产 6,333 个依赖/12 个产物逐项核对完整，见 `/tmp/mla-m10c1b/takeover.json`。本步共享 Kernel、metadata、公开入口及构建改变，累计 E2E 实跑，继承 0；不继承历史单元/性能。

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。实机使用第 6 张 910B3。最新统一构建的实际加载路径为仓库 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA-256 `296058799934c9e54dd1f42989c47e009fedae912b5037107e52e8008131900b`，对应 `acceptance-final/attempt-1791990623804824840/runtime.json`。已有库不替代该构建/加载证据。

本步证据根 `/tmp/mla-m10c1b/`；构建与内容数学缓存沿用 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`。`build/` 仅有 `mla-codex-loop/`。原始证据清理后不可继续继承。

## 最近闭环验收

所有组已先 dry-run/collect-only。CPU 与 sanitizer 为同一 79 项；最终设备集合为 64 项新行为专项（含 10 项源差分）、28 项 Trap、134 项累计 E2E、516 项普通 FA3 与一个双 dtype 公开性能组，不使用截断。

| 本步集合 | 本次通过 | 耗时 |
| --- | --- | --- |
| `split-bsnd-tools` | 最终 15/15，含源身份/失败传播、两阶段 profiler 解析及正式证据目录 | 7.3 秒 |
| `split-bsnd-cpu` | 79/79，源片段/序列化/独立消费/空值/宽整数/失败原子性 | 36.4 秒 |
| `split-bsnd-sanitize` | 同一 79/79，ASan/UBSan 及退出/泄漏检查通过 | 113.2 秒 |
| `split-bsnd-check` 开发代表 | 10/10；之后修复 fake 分派，本组节点已纳入完整专项重跑 | 49.7 秒 |
| `split-bsnd` 完整专项 | 64/64；20 项 O/LSE 均逐元素源一致且双数学通过，真实 accumulator 检查通过 | 157.7 秒 |
| `split-bsnd-trap` | 28/28，全部设备 event 错误与实际加载身份匹配 | 377.7 秒 |
| 累计 `e2e` | 134/134，本次实跑，继承 0 | 26.7 秒 |
| `ordinary` | 516/516 | 32.8 秒 |
| `split-bsnd-profile` | 双 dtype 全部 16 配置实测；4 个适用配置平均比值 0.732195 ≤1，通过 | 368.6 秒 |

CPU 最终记录为 `cpu-expanded/attempt-1791989880158511480/`，工具最终为 `tools-final/attempt-1791990500883327780/`，设备最终为 `acceptance-final/attempt-1791990623804824840/`。全部完成、无残留本步测试进程。原始命令、集合、JUnit、数学指标及输入/输出证据保存在各组目录，固定源探针构建/加载证据在 `split-bsnd/probes/`。

最终证据复核见 `final-evidence-check.json`：各设备组源码及原始文件、生产 6,349 个依赖/12 个产物、实际加载和原生对照身份均匹配，错误 0。CPU 普通/sanitizer 各 47 个原始文件完整，重新提取的 34 个本地编译输入未变；其后的 Autograd 注册及设备证据目录修改不影响 CPU 生成/消费，未继承历史单元。公开性能从原始 wall 样本重新汇总，与保存报告一致。

数学 reference 继续按内容缓存，相同输入不重复计算，待测实现仍实际执行。最终新行为专项缓存 hit 29/miss 28，累计 E2E hit 281、无重算，普通 FA3 hit 1/miss 30，性能 hit 26/miss 26；新增 miss 均成功写入。开发代表与后续完整专项复用相同输入的 reference。193..256 超出固定源 FD 表示域，只列扩展双数学，不声称源差分；本步源可表示域的 20 项 O/LSE 比较均源逐元素一致且双数学通过，无需启用源精度限制例外。

复现仍先加 `--dry-run`、`--collect-only` 核对，再执行（输出使用新目录以保留本次证据）：

```bash
python3 -m tests.mla --suite split-bsnd-tools --output /tmp/mla-m10c1b/tools-repro
python3 -m tests.mla --suite split-bsnd-cpu split-bsnd-sanitize --output /tmp/mla-m10c1b/cpu-repro
python3 -m tests.mla --suite split-bsnd split-bsnd-trap e2e ordinary split-bsnd-profile --device 6 --output /tmp/mla-m10c1b/acceptance-repro
```

## 公开性能与适用边界

最终 `split-bsnd-profile/public-profile/average.json` 保留 16 个固定配置的全部结果。依照既有双方执行路径已迁移的口径，4 个适用配置先在配置内等权平均入口/R，再取算术平均，结果 **0.732195**，整体退化配置 0、分段退化 2。各段均值：

| 入口/R | FA3/CANN 耗时比 |
| --- | --- |
| 复用 metadata，R=1 | 0.724420 |
| 生成加前向，R=1 | 0.975979 |
| 生成加前向，R=4 | 0.635481 |
| 复用 metadata，R=4 | 0.572092 |
| 隐式生成，R=1 | 0.793576 |
| 隐式生成，R=4 | 0.691623 |

其余 12 配置的原生自动 FD 变长调度尚未迁移，按 [design 6.2.12](design.md#6212-bsnd-显式-split-生产适配) 单列；诊断平均比值 1.455413，退化配置 6、分段退化 50，明细见 `native-fd-diagnostics.json`。本步显式均匀分片已开放，不将 FD Block 闭环等同于原生自动分派完成。原始 wall、profiler、两阶段 kernel 时间及全部单例保留，不按耗时筛选输入；上述诊断不算适用性能通过。性能采样设备检查保存在本步根目录和组内。

## 失败处理、未运行与质量

开发编译修正了 AICore 头文件的 host 可见性和误用 Vector Min 的标量调用；CPU 驱动修正全局空容量的零 workspace，FA3 空尾片 oracle 按约定规范化端点。首次工具测试的 monkeypatch 隔离问题已修复并通过最终工具集合。

第一次完整设备专项在 24 项通过后因 FakeTensor 进入真实 C++ 内核失败，记录在 `acceptance/`。已将新算子的 Autograd 注册改为与既有路径相同的 fallthrough，当前修复版本的公开 fake/compile、生命周期、长度/scale/内容复用和静态溢出用例已通过。`acceptance-fake-fixed/` 的编译成功，但准备记录因期间补齐固定源探针正式证据目录而被统一工具判为 `Source changed during preparation`；该记录不算通过。最终验收重新启动，核验并复用修复构建，准备 1.2 秒、运行时核对 27.2 秒。

最终必要集合全部通过，无未解决失败；full-matrix、历史单元/性能和 910C 未运行，不列为通过或继承。三份文档按本步新增公开契约核对，相关链接检查见 `doc-links-final.json`；本步文件的仓库 pre-commit 质量检查见 `quality-final.log`，`git diff --check` 通过。构建和大日志均在外部工作目录，未修改 CANN/CATLASS，未暂存其他工作。

## 唯一下一步

[M10c2 BSND 自动 split](plan.md#唯一下一步-m10c2bsnd-自动-split)。下次先核对本步进程、完整原始证据及当前依赖，复核原生变长调度与持久布局差异，再明确独立验收边界和必跑集合。本轮只提交 M10c1b 一次，提交后停止，不开始下一步。
