# MLA 当前状态

<!-- mla-loop-next: none -->

**M21 已完成，已授权的 M18–M21 INT8 范围全部收尾，无下一开发步。** 本轮基线 `a284d12`，分支 `feat/mla-cann-migration`。交付可运行的 BSND/TND INT8 示例、中英文接口说明及 [组合覆盖映射](design.md#78-int8-收尾映射)，并修复 BSND balanced 调度的尾空 KV 输出初始化。阶段正确性、eager 性能及质量检查通过；graph 性能按 [design 3.4](design.md#34-int8-压缩-mla-扩展) 的既有用户授权暂缓，实测失败保留。完成本次提交后停止，不扩张范围。

## 当前能力与本轮修复

公开范围仍为 512+64 共享 KV、BF16 RoPE/输出、FP32 三 scale、BSND/TND Q、ND PA 128 页；v48/v49、mask/split 及拒绝项按 design 3.4。示例直接使用公开 API，覆盖 metadata 复用、scale 原地更新、图内重生成记录及改变 KV 长度。日常 E2E 增加两布局示例和 decode/causal 尾空段代表，旧清单无删减。

新 BSND 示例首先暴露 KV 从 `[129,0,65]` 变为 `[65,0,0]` 后尾样本保留旧输出。源 balanced 分核可能提前结束，原公开补初始化只覆盖 TND；现两布局均在数值流水与 FD 归并后由最后使用核首 AIV 清理未访问的空 KV 尾段。保留数值终点、flush、调度协议和量化算法，非 balanced 路径不变；六个具名用例覆盖短/长 KV、单/多 Q、自动/单片与连续多个空尾段。未修改 CATLASS 或 CANN 源。

## 本次闭环证据

工作根为 `/tmp/flash-attention-npu-mla-46928a793876`，本轮证据实际存放于其 `mla-m21/` 子目录。`/tmp/mla-m21` 是兼容别名，保留 summary、manifest 与原始证据中的既有绝对路径；以下路径相对此证据根。构建/reference 均在工作根，`build/` 仅保留无头日志与恢复状态。

统一 prepare 已重建生产扩展；实际加载 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `1e5ee5b0c7d5aafbd8e0613e917f1040e6bc4fe5f2a6de7a7244d88fa9c6a015`。收尾使用现有驱动的结果、依赖、构建和原始证据校验函数，见 `closure-correctness.json`、`closure-build.json`。**本次历史 E2E 继承为 0；不继承历史单元或性能。**

| 检查 | 本次结果与耗时 | 证据 |
| --- | --- | --- |
| 接手基线 | M20 原 1,667 项正确性及构建/native 身份有效，仅作修复前基线 | `handoff-evidence.json`、`handoff-runtime.json` |
| 工具/平台 | 15 / 41 通过，5.6 / 6.9 s；平台组只检查 CANN 910B/C 配置 | `final-tools/summary.json` |
| 集合与命令 | dry-run/collect 完成；BSND 专项新增 6、累计 E2E 新增 4，无截断或删减 | `final-dry-run.json`、`collect/summary.json`、`fix-collect/summary.json`、`selection-diff.json` |
| 用户示例 | FP16/BF16/INT8 BSND/TND 共 4 通过，21.8 s；两布局 INT8 CLI 运行通过 | `fix-check/summary.json`、`example-cli.log` |
| 两布局代表 | BSND 11 / TND 28 当次通过，24.0 / 82.1 s；旧 probe 身份处理见下文 | `fix-check/summary.json` |
| 固定源代表 | BSND 23 / TND 15 通过，27.4 / 25.4 s | `fix-check/summary.json` |
| 性能预检 | 两布局各 4 配置通过，145.2 / 132.4 s；231 文件归档校验 | `profile-check/summary.json`、`profile-check.log` |
| 完整专项 | BSND 115 / TND 151 通过，273.2 / 506.6 s | `acceptance/summary.json` |
| 累计 E2E | 905 通过，247.0 s | 同上 |
| 普通 FA3 / 普通图 | 504 / 2 通过，28.5 / 21.7 s | 同上 |
| 构建与质量 | 生产重建、实际加载核对、相关文档链接、diff 和仓库 hooks 通过 | `fix-build.log`、`fix-quality.log`、`quality.log`、`documentation-check.json` |

正式正确性共 **1,677 项通过，零失败、零跳过**。代表检查新增 4 个 reference 缓存、命中 97 次；完整正确性命中 1,407 次且无新增计算，待测实现每次实际执行。

早期 BSND 11 / TND 28 代表组执行后，固定源组重建了 INT8 `migrated.so` 及其 manifest，其旧依赖身份已过期，不能据此继承通过。收尾已核对这些节点全部包含于重建后完整专项，且完整专项的当前依赖和原始证据有效。首轮旧扩展示例失败仍保留在 `check/summary.json`（2 通过 / 1 失败 / TND 未运行）；修复后四项示例及完整回归均通过。

独立数学判定与固定源一致性分开记录，汇总为 `source-math.json`：

| 完整专项源记录 | 固定源逐元素一致 | O 数学通过 / 源限制 | LSE 数学通过 / 源限制 |
| --- | --- | --- | --- |
| BSND 28 | 全部 | 4 / 24 | 1 / 27 |
| TND 20 | 全部 | 2 / 18 | 0 / 20 |

固定 CANN 的 Q/K scale 作用于内容与 RoPE 合并后的分数，且保留动态概率量化，独立反量化数学存在既有精度限制。按 design 第 7 节执行源优先级，不修改阈值、不声称双数学全部通过。

## 阶段公开性能

固定 BSND 24 / TND 15 配置全部实测，三入口 R=1/4，5 次预热、5×20 wall、5×10 kernel 与双方计时边界不变。复用现有汇总器从原始样本重算并核对报告；39 配置按配置等权平均，无排除项。`performance-summary.json` 保留汇总、所有单例/分段及原始证据哈希。

| 布局 | 配置 | eager 均值 | eager 退化配置 / 分段 | graph 均值 | graph 退化配置 / 分段 | 组耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| BSND | 24 | 0.722964，通过 | 0 / 1 | 1.247368，失败、暂缓 | 24 / 130 | 709.3 s |
| TND | 15 | 0.683287，通过 | 0 / 0 | 1.116739，失败、暂缓 | 15 / 76 | 454.3 s |
| 合计 | 39 | **0.707703，通过** | 0 / 1 | **1.197126，失败、暂缓** | 39 / 206 | 1,163.5 s |

两组默认工具均退出 1，唯一失败为 graph 平均门槛；未改工具或把失败记为通过。用户于 2026-10-19 明确“图模式性能暂不做要求”，当前 INT8 graph 性能按 design 3.4 暂缓；图正确性和 eager 已通过。原始证据为 `bsnd-performance/summary.json`、`tnd-performance/summary.json` 及各组 `public-profile/`。默认失败组不生成成功证据哈希，因此汇总单独使用现有 `files_identity` 保存并校验其原始文件身份。

共 70,200 条 wall 样本，eager/graph 各 3,900 条 kernel 样本。源 NZ pack 准备成本单列于 `preflight.json`：BSND 合计 0.1712 s（单例 0.0006–0.1340 s），TND 合计 0.2377 s（0.0008–0.2045 s）；双方调用内转换成本、capture 成本及设备隔离记录均保留。性能进程固定 CPU 0、可见设备 7；私有 tmpfs 包装器分别归档并校验 238 / 221 文件，见两组外层日志。

## 环境与复现

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2；910B3 物理设备 7（20 AIC/40 AIV），**910C 未实测**。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`，固定 CANN 源 `ca18aee610729ca5270a44e2d84c751bc687fbf0`。INT8 native `/tmp/mla-m18/native` 的配方、12 keys 与包内容有效，非量化 native `/tmp/mla-m13b/native`。设备验证在宿主运行，环境包装器 `/tmp/mla-m16-schedule/run.py`；质量使用已有 `/tmp/mla-m01-precommit` 包及仓库入口。

```bash
export MLA_WORK_DIR=/tmp/flash-attention-npu-mla-46928a793876
export MLA_INT8_NATIVE_ROOT=/tmp/mla-m18/native
export MLA_INT8_SOURCE_LOG=/tmp/mla-m18/source-final/attempt-1792357915791404830/int8-source-check/output.log
python3 /tmp/mla-m16-schedule/run.py python3 -m tests.mla \
  --suite detail-int8_bsnd detail-int8_tnd e2e ordinary graph-ordinary \
  --device 7 --output "$MLA_WORK_DIR/mla-m21-correctness-recheck"
```

性能复现分别选择 `int8-bsnd-profile` / `int8-tnd-profile`，输出使用工作根下的新目录；命令前加 `unshare -Ur -m python3 /tmp/mla-m16-schedule/run_private_storage.py <新输出目录>`，后接环境包装器与统一驱动，`--output` 使用同一目录。包装器逐文件校验归档；默认 graph 失败须与 eager 判定分开记录。

[计划 M21](plan.md#m21int8-收尾) 的交付条件已完成，唯一下一步为 `none`。没有新的已授权迁移任务。
