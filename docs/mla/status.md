# MLA 当前状态

<!-- mla-loop-next: M20 -->

**M19 已闭环；唯一下一开发步为 M20：TND Q + ND PA INT8 公开前向。** BSND 完整调用链、必要正确性、累计回归及 eager 性能通过。用户于 2026-10-19 明确允许本次暂缓 graph 性能优化；graph 平均比 **1.1969448036 > 1** 仍记失败，按本次授权接受该限制，不改写为性能达标。本次例外不自动扩展到 M20/M21，范围见 [design 3.4](design.md#34-int8-压缩-mla-扩展)，下一步见 [plan M20](plan.md#m20tnd-公开-int8)。本次只完成 M19 提交，提交后停止。

分支 `feat/mla-cann-migration`，本步基线 `3ed72dc`，M18 开发闭环为 `619787b`；当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。收尾接手的是 M19 未提交实现，保留其生产代码和性能工具；新增测试包声明，修复累计收集的同名模块冲突，并修正旧 INT8 compile 拒绝测试对 Dynamo 异常包装的断言。

## 当前能力与验收

BSND INT8 的 host/AICPU/Kernel dispatch、v48 metadata、三个 keyword-only scale 和 real/fake/compile/图重放已接通，复用 M18 数值核并导出真实 FP32 partial。TND、非 PA、有限窗口/softcap、显式多 split 和非默认有效调度选项仍明确拒绝。公开缓存为 ND PA、页大小 128，INT8 内容配 BF16 RoPE/输出、FP32 scale；生产路径不转调 CANN。

证据根 `/tmp/mla-m19/`，工作根 `/tmp/flash-attention-npu-mla-46928a793876`。下表区分本次收尾实际执行与同一 M19 已完成的实测；继承历史 E2E **0**，不继承历史单元或性能。本次收尾实际通过完整专项与累计回归 **1,504 项**，另有修复验证用的 16 项 INT8 日常小组。第一次失败累计组的部分通过不计入最终通过数。

| 检查 | 结果与耗时 | 证据（相对证据根） |
| --- | --- | --- |
| 同一步单元 `--unit int8_bsnd` | 38 通过，10.8 s；固定 native 调度、边界/溢出、ASan/UBSan、性能工具成功/失败夹具；收尾复核原证据 | `tools-parallel/summary.json`、`closure-handoff-check.json` |
| 同一步生产构建与代表组 | 原完整构建 587.3 s、代表 5 项通过 21.2 s；收尾构建内容校验复用 1.4 s、runtime 21.0 s | `parallel-check/summary.json`、`closure-regression-final/summary.json` |
| dry-run / collect | 完整收集 109 + 889 + 504 + 2 项；修正后的 INT8 小组 16 项；正式命令和用例清单一致，无截断 | `closure-collect-fixed/summary.json`、`closure-int8-collect/summary.json`、`closure-selection-check.json` |
| `detail-int8_bsnd` | 109 通过，266.2 s；46 real/fake 拒绝、12 Trap、28 source/math 与 partial 对照，及长度/stride/空值/compile/图/stream | `closure-regression/summary.json`、`closure-detail-recheck.json` |
| INT8 修复验证 `--e2e int8 int8_bsnd` | 16 通过，22.0 s；明确核对缺 scale 的 eager 原因和 compile 包装，及已开放 BSND 行为 | `closure-int8-check/summary.json` |
| 完整累计 `e2e` | 889 通过，213.5 s，reference 缓存命中 1,235 次；实现全部实际执行 | `closure-regression-final/summary.json` |
| `ordinary` / `graph-ordinary` | 504 / 2 通过，27.0 / 21.0 s | `closure-regression-final/summary.json` |
| 同一步性能预检 | 通过，129.6 s；4 配置覆盖 decode FD、causal/stride、快速 FD、最大公开分配，400 graph Kernel 样本 | `preflight-parallel/summary.json` |
| 同一步正式公开性能 | 完整 24 配置，648.2 s；eager 平均比 **0.7502044961，通过**；graph 平均比 **1.1969448036，失败，本次授权暂缓优化** | `performance-parallel/summary.json`、`closure-performance-check.json` |

最新专项证据为 `closure-regression/attempt-1792369849844459450/detail-int8_bsnd/`。28 组 source 对照 O/非空 LSE 均逐元素通过；独立数学 O **4 通过/24 源限制**、LSE **1 通过/27 源限制**，与修正前统计一致。双 reference、误差、native/source/actual、输入和 v48 记录均保留；uniform 日常数学用例不混入该统计。源整体缩放 RoPE 的限制见 design，不称为全部数学通过。后续只修改旧拒绝测试断言，统一驱动确认本专项源码闭包、外部依赖与原始证据仍匹配，未重复执行该专项。

最终累计证据为 `closure-regression-final/attempt-1792370703133276430/`。`closure-final-evidence-check.json` 使用现有 `validate_result`、`reusable`、`matching_files` 与源码依赖闭包检查，核对实际用例、summary、加载扩展、构建 manifest 和原始通过证据。数学 reference 可复用，同步执行待测实现。

## graph 性能限制与环境

正式性能原始数据在 `performance-parallel/attempt-1792365862152597060/int8-bsnd-profile/public-profile/`：43,200 条 wall 样本，eager/graph 各 2,400 个 Kernel 样本；5 次预热、5×20 wall、5×10 Kernel，24 配置、三入口 R=1/4 全部保留。eager 配置退化 0/24、分段退化 1/144；graph 配置退化 23/24、分段退化 117/144。graph R=4 的 reuse/generate/implicit 分段平均比为 1.3405206890 / 1.4069310678 / 1.3553210139。收尾从完整原始样本重算，与原报告逐字段一致；这是同一步原采样复核，不计新的性能执行。性能工具仍按原门槛报告 graph 失败，未来显式重跑该组仍可能失败。

实测环境为 Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2、910B3 物理设备 7（20 AIC/40 AIV），CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f` 未修改。910C 未实机验收。实际扩展为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `0b2742c3ff2a4b1b4173d19a5c32380a2b23b2e56faf9b9c111e1493153610b8`；本步构建 manifest、收尾 runtime 与测试加载身份一致，见 `closure-runtime.json` 和最终 summary。收尾没有重建生产扩展。

固定 CANN 源为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，INT8 native `/tmp/mla-m18/native` 的配方、12 个 key 与包内容由既有验证器核对；非量化 native 为 `/tmp/mla-m13b/native`。构建、数学缓存及组件 DSO 留在工作根，不放入 `build/`。使用 `/tmp/mla-m16-schedule/run.py` 中已冻结的 CANN 环境；默认 shell 库路径会导致 `npu-smi` 符号冲突，沙箱内收集也不能访问设备，正式验证在宿主环境完成。

已修复的收尾失败保留原记录：`closure-collect-host/` 的同名 `test_public.py` 导入冲突通过新目录包声明解决；`closure-regression/` 中第一次累计 E2E 为 879 通过、1 个 compile 文案断言失败，未运行剩余组。修正后先通过 16 项小组，再完整重跑全部累计组，未降低正确性要求。早期 graph Trap 诊断、准备阶段源码身份失配及 profiler 存储问题的原日志仍在证据根；正式性能的 238 个归档文件已逐文件校验，见 `performance-parallel.log`。

## 复现与唯一下一步

```bash
export MLA_WORK_DIR=/tmp/flash-attention-npu-mla-46928a793876
export MLA_INT8_NATIVE_ROOT=/tmp/mla-m18/native
export MLA_INT8_SOURCE_LOG=/tmp/mla-m18/source-final/attempt-1792357915791404830/int8-source-check/output.log
python3 /tmp/mla-m16-schedule/run.py python3 -m tests.mla \
  --suite detail-int8_bsnd --device 7 --output /tmp/mla-m19/next-public
python3 /tmp/mla-m16-schedule/run.py python3 -m tests.mla \
  --suite e2e ordinary graph-ordinary --device 7 --output /tmp/mla-m19/next-regression
```

性能复现仍使用 `int8-bsnd-profile-check` / `int8-bsnd-profile`，通过 `unshare -Ur -m python3 /tmp/mla-m16-schedule/run_private_storage.py <新输出目录> <上述环境包装器及验收命令>` 使用私有 tmpfs；新输出目录须尚不存在，正式采样保持设备隔离。该工具不会自动豁免 graph 失败。

质量入口 `PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-precommit-cache ci/code_quality.sh --staged` 已通过全部适用 hooks，未改写源码；仅暂存本步 25 个文件。差异、14 个本次新增/修改链接、复现 Bash 语法及唯一 M20 标记检查通过。证据见 `closure-quality.log`、`closure-docs-check.json`；本次验收进程已退出，见 `closure-process-check.json`。

唯一下一步是 M20，按 plan 同一提交完成 TND Q 的完整公开接入与必要验收；本次不启动它。M19 graph 性能仍是已知限制，本次授权只解除本次提交阻塞，后续范围与默认门槛不变。无头脚本继续读取唯一 `M20` 标记，不修改脚本或伪造其历史运行状态。
