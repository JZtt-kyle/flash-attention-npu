# MLA 当前状态

<!-- mla-loop-next: M19 -->

**M18 已完成；唯一下一开发步为 M19：BSND Q + ND PA INT8 完整公开前向。** 分支 `feat/mla-cann-migration`，最近开发闭环为 `619787b`。本轮仅修正文档，不推进功能；公开 INT8 的 real/fake/compile 仍明确拒绝，三个 `mla_*_scale` 公开参数尚未接入。M18–M21 已获授权，范围见 [design 3.4](design.md#34-int8-压缩-mla-扩展)，后续步骤见 [plan](plan.md)。

## 当前能力与限制

非量化两种表示、五布局、mask/split/softcap、GQA 调度选项及图执行已在 M17 收尾。M18 交付 INT8 内容 / BF16 RoPE 与输出 / FP32 scale 的独立 Kernel/Block/Tile、ND PA 读取及直接/普通与快速 FD；原始 tiling 仅是测试夹具，不是生产 metadata。TND 组件 LSE 仍为源 `[Tq,Hq]`，公开 `[Hq,Tq]` 适配属于 M20。源边界、缩放语义和必要适配统一见 design 3.4。

- M18 的 56 组组件 O 及非空 LSE 与固定源逐元素一致，空行和真实 partial 检查通过；独立 FP32/FP64 数学的 O 为 **21 通过、35 源限制**，LSE 为 **10 通过、46 源限制**。源先合并内容与 RoPE 点积再整体乘 Q/K scale，与独立反量化公式不同，不能把超限写成数学通过。
- 24 配置组件性能的目标/源平均 Kernel 比为 **1.1450877174**（23 项退化），预分配 probe wall 比为 **1.0181407505**（21 项退化）。这是 O-only、同原生分区的源 NZ / 目标 ND 单 Kernel 对照；采样协议通过不表示平均不劣化。INT8 公开 eager/graph 尚未实现或验收，M19/M20 仍分别要求平均 ≤1。
- M17 的非量化图平均比 **1.484216023946693** 未达 ≤1，用户仅接受该历史收尾限制；失败仍保留于 `/tmp/mla-m17/performance/summary.json`，完整记录用 `git show ebdf3eb:docs/mla/status.md` 查询，不豁免 INT8。

## 最近开发闭环证据

证据根 `/tmp/mla-m18/`。下表是 **M18 当时实测**，不是本轮重跑结果；当时无 E2E 继承，最终各测试组无失败/跳过。原始节点、selection、依赖/构建 manifest、输入/输出/reference 和性能样本由统一驱动保存。

| 统一入口组 | M18 结果 | 耗时 | summary 路径（相对证据根） |
| --- | --- | --- | --- |
| `int8-tools` | 45 通过 | 28.6 s | `tools-final/summary.json` |
| `int8-source-check` | 36 通过：22 组源几何、13 项拒绝及 RoPE 缩放辨别 | 29.6 s | `source-final/summary.json` |
| `--unit int8` | 131 通过：数值、CPU 边界、固定源、工具和独立 Trap | 141.3 s | `unit-empty/summary.json` |
| `int8-profile-check` | 1 项预检通过，覆盖 24 个固定配置 | 22.5 s | `profile-preflight/summary.json` |
| `int8-component-profile` | 1 项采样协议检查通过，24 配置性能差距见上 | 71.6 s | `performance/summary.json` |
| `e2e` | 879 通过，含 6 项 INT8 公开拒绝 | 219.4 s | `regression/summary.json` |
| `ordinary` / `graph-ordinary` | 504 / 2 通过 | 34.9 / 21.9 s | `regression/summary.json` |

生产重建 588.0 s、运行环境核对 20.8 s 后执行回归。组件数学统计及 `rope-scale-order.json` 位于 `unit-empty/attempt-1792358703638268390/unit-int8/`；性能的全部配置、direct/FD 分段、样本和 trace 位于 `performance/attempt-1792358984629355250/int8-component-profile/profile/`。早期诊断失败及修复从 `git show 619787b:docs/mla/status.md` 查询，原日志保留，不能当作通过证据。`/tmp` 证据仅在仍可访问且身份有效时使用；缺失后不得继续继承。

## 证据身份与本轮复核

M18 记录的环境为 Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1，CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，910B3 物理设备 7（20 AIC/40 AIV）；910C 未实机验收。本轮未探测设备或导入扩展，环境值不作为新的运行证明。固定 CANN 为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，只读 checkout HEAD 为 `2060a3db064c3352fec6262faed8a2decaf3a233`，native/probe 必须取固定 Git 对象；CATLASS 为 `769cd40a8716b28650b6bebb08db4834eea4462f`。

- M18 生产回归记录实际加载 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `2adfe6e4b8308dccf1bd5794b4175dd9c1a3a41111ce55fe8b2bd1e40017999e`；本轮静态文件哈希仍匹配。`regression/attempt-1792359105965091850/` 的 `runtime.json`、`build-manifest.json` 和各组 `loaded.json` 绑定该产物。此前 INT8 独立组记录的包扩展仍是 M17 的 `14edf068…`，其数值执行对象是独立 probe/native，不能拿它们证明 M18 生产重建。
- 工作根 `/tmp/flash-attention-npu-mla-46928a793876`，生产构建/数学缓存在 `mla-validation-artifacts/`，独立 INT8 DSO 在 `int8-probes/`。INT8 native 为 `/tmp/mla-m18/native`，本轮按现有 `native_vendor()` 核对配方、12 个 key 和包内容通过；非量化包仍为 `/tmp/mla-m13b/native`。
- M18 验收后仅修改 `mla_int8_params.h`、`tile/mla_int8_ops.hpp`、`block/mla_int8_merge.hpp` 的来源注释。`notice-only-correction.json` 与当前文件及 `#pragma once` 后代码正文哈希匹配；生产 manifest 的源码身份和 probe 的依赖身份因这三文件失配，已保存的产物哈希仍匹配。下一开发步由驱动决定重建/重跑，不手改 manifest 或假称缓存有效。

本轮用已有 `validate_result`、`reusable`、`matching_files` 静态复核八个最终组的结果与原始证据，均完整；这只确认历史记录，**本轮设备实测 0、E2E 继承 0**，未重建扩展。核对固定 FA3 本地源码（`flash_api.cpp` 固定哈希匹配）、CANN Git 对象、当前入口/构建/采样代码，修正非量化与 INT8 范围、构建清单、历史引用和复现环境，压缩已完成计划。复核记录在 `/tmp/mla-doc-review/{evidence,identities}.json`。文档一致性、508 个本地链接/锚点、45 个表格、4 个 Bash 代码块语法及 9 组选项 dry-run 检查通过（约 1.0/1.7 s，`docs-check.json` / `commands-check.json`）；未收集或执行设备用例。`git diff --check` 和 staged 质量入口的适用钩子通过（约 0.9 s，`quality.log`）。默认 Python 未装 pre-commit，复用已有工具：`PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-precommit-cache ci/code_quality.sh --staged`。

## 复现与唯一下一步

以下是 M18 最小复现顺序（在 Bash 子 shell 中任一步失败即停止），供后续确需重跑时使用，本轮未执行。先核对包装器 `/tmp/mla-m16-schedule/run.py`、`environment.json` 和私有 tmpfs 包装器 `run_private_storage.py` 仍适用，并选择新的复现目录；它固定非量化 native 路径，INT8 路径须独立导出。每个选择先 dry-run/collect；只有源组成功后才从新 summary 设置源日志，不能引用失败 attempt 或旧日志。

```bash
(
set -e
export MLA_WORK_DIR=/tmp/flash-attention-npu-mla-46928a793876
export MLA_INT8_NATIVE_ROOT=/tmp/mla-m18/native
MLA_RECHECK_DIR=/tmp/mla-m18-recheck
python3 /tmp/mla-m16-schedule/run.py python3 -m tests.mla \
  --suite int8-tools int8-source-check --device 7 --output "$MLA_RECHECK_DIR/source"
MLA_INT8_SOURCE_LOG="$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["groups"]["int8-source-check"]["directory"] + "/output.log")' \
  "$MLA_RECHECK_DIR/source/summary.json")"
export MLA_INT8_SOURCE_LOG
python3 /tmp/mla-m16-schedule/run.py python3 -m tests.mla \
  --unit int8 --device 7 --output "$MLA_RECHECK_DIR/unit"
python3 /tmp/mla-m16-schedule/run.py python3 -m tests.mla \
  --suite int8-profile-check --device 7 --output "$MLA_RECHECK_DIR/preflight"
# 性能须在空闲设备上独立采样；包装器要求 performance 目录尚不存在。
unshare -Ur -m python3 /tmp/mla-m16-schedule/run_private_storage.py \
  "$MLA_RECHECK_DIR/performance" \
  python3 /tmp/mla-m16-schedule/run.py python3 -m tests.mla \
  --suite int8-component-profile --device 7 --output "$MLA_RECHECK_DIR/performance"
python3 /tmp/mla-m16-schedule/run.py python3 -m tests.mla \
  --suite e2e ordinary graph-ordinary --device 7 --output "$MLA_RECHECK_DIR/regression"
)
```

唯一下一开发步仍是 [M19](plan.md#m19bsnd-公开-int8)：同一提交完成 BSND 的 host/AICPU/Kernel、metadata、公开 real/fake/compile、真实图及必要公开性能；按开放范围调整 INT8 拒绝清单。本次文档轮提交后停止，不启动 M19，也不改写无头脚本的 pending 或成功次数。当前文档提交可用 `git log -1 -- docs/mla/status.md` 定位，最近开发闭环须查询上述 `619787b`。无头用户入口仍为 `/home/zhangqijia/repo/mla_codex_loop.py`。
