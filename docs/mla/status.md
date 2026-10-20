# MLA 当前状态

<!-- mla-loop-next: none -->

**R9 已完成：MLA 使用普通调用路径，移除额外的图捕获拒绝。** 基线 `a96f49b`，分支 `feat/mla-cann-migration`。用户于 2026-10-20 明确“不用特殊处理就行，不要加额外的”，本轮按该要求修正 R8 对图处理的解释。

## 改动与身份

- 生产代码仅删除 `mla_forward.cpp` 的 9 行：`CheckEagerExecution`、`DensePlatform` / `CausalPlatform` 两处调用及 `NPUGraphsUtils.h` 依赖。前向和 metadata 沿用普通提交路径，不查询捕获状态或增加图调用拒绝。
- R8 已删除的 metadata 捕获切流、INT8 冷捕获 mask 特例和 MLA 图专项测试保持删除。本轮数值组件、发射、动态校验、多流、Python real/fake 与测试文件均未修改；普通 FA3 既有图测试继续回归。同步修正 design、plan 与开发约定，不将取消拒绝记为 MLA 图验收通过。
- 证据根 `W=/tmp/flash-attention-npu-mla-46928a793876/mla-plain-capture-20261020`。实际加载主树 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 **`d372ce62ee861296b1d8c679ef8e319487e42204012aa74d44df8a2cd31507ac`**；源码/测试哈希为 `final-identity.json`，加载记录为 `candidate-environment.log`。基线扩展另存 `before-extension.so`（SHA256 `1d28ae6eb482f1fe12ba80d8908aa6d1ccd1f663bb34e845259cc34f32dea812`），生产差异见 `production.patch`。

环境：Ascend910B3 物理设备 **3** / 逻辑 0，性能 CPU 48–51；Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。固定 CANN 源为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，非量化包 `/tmp/mla-m13b/native`、INT8 包 `/tmp/mla-m18/native` 的 2,027/2,025 个文件逐项验 SHA；CANN/CATLASS 未修改。构建、缓存和日志均位于仓库外。

## 本次验证

| 检查 | 结果与证据 |
| --- | --- |
| 构建与加载 | **通过 / 151.87 s**，复用原对象目录，仅重编 `mla_forward.cpp` 并重新链接；`build-cached.log`、`build-sources.log`、`candidate-environment.log` |
| 收集 | **1,480 MLA / 合计 1,620**，与 R8 节点清单逐项一致；`collect.log`、`acceptance-collect.log`、`collection-check.log` |
| 代表/成本边界 | **106 passed / 39.36 s**，覆盖五布局、双 dtype、长 KV、INT8、fake/compile 与多流；`representative.xml` |
| 完整公开回归 | **1,480 MLA + 140 普通 FA3 = 1,620 passed / 141.72 s**，零失败/错误/跳过；`acceptance.log` / `acceptance.xml`；普通 FA3 含 129 metadata 签名、6 metadata、3 compile、2 图节点 |
| 性能预检 | 非量化 10 配置、INT8 BSND/TND 各 4 配置的数值与最长分配通过；正常/缺失/重复/非有限统计夹具及 native 包哈希通过；各 `*-preflight.log` |
| 性能结果校验 | 非量化两侧各 **56,000** 条样本、INT8 **35,100** 条样本的完整性、有限值、输入/扩展身份、统计重算与门槛通过；`performance-validation.log`、`e2e-summary-validation.log` |
| 仓库质量 | 本步五个文件的既有 pre-commit、149 个本地文档链接、最终源码/扩展身份与 `git diff --check` 通过；`quality.log`、`final-checks.log` |

数值使用现有独立 reference 与 `assert_fa_close`，阈值未放宽。公共 golden 缓存仅复用未变的 reference，当前扩展均实际执行；106 项代表不重复累计到完整 1,620 项，R8 的通过数和性能不累计继承。MLA 图专项未运行，仓库中未新增或恢复图测试。

准备调整：裸 shell 的 `npu-smi` 因驱动库路径冲突失败，固定环境启动器下正常且设备 3 空闲。最初新 `build-lib` 目录绕过对象缓存，285.66 s 后主动停止；改用原对象目录完成上述构建。中断构建不计通过，保留 `build.log`、`build-restart.log`；最终无未解决失败。

## 当前普通调用性能

沿用 R8 固定输入与普通调用计时边界：5 次预热、5×20 wall、适用 implicit/generated/reuse 及 R=1/4；先对配置内入口/R 等权，再对配置等权。采样及统计代码仅更换证据目录。下表 **FA3 / 固定 CANN** 均值门槛 ≤1，各阶段分开验收。

| 集合 | 配置数 | 当前 / CANN | 当前 / R7 | 对 CANN 退化配置/分段 |
| --- | ---: | ---: | ---: | ---: |
| 原调度 | 20 | **0.749028，通过** | 1.019583 | 0 / 6 |
| 原 eager 矩阵 | 40 | **0.783272，通过** | 1.014362 | 0 / 21 |
| 默认压缩路径 | 10 | **0.698634，通过** | 0.693762 | 0 / 2 |
| INT8 BSND | 24 | **0.597725，通过** | 本次未测 | 0 / 0 |
| INT8 TND | 15 | **0.607951，通过** | 本次未测 | 0 / 0 |

非量化同时保留原 R7（`36e8609`）对照：扩展 SHA256 `92ca4037ef8a6e08632090b12bb675ac62ca3b924a332d6897d4d32006bea233`。原调度/eager 组分别比 R7 慢 **1.96% / 1.44%**，14/20 与 28/40 个配置变慢；默认压缩路径比 R7 快 **30.62%**。这些差值包含 R8 优化，不能归因于本步删除捕获检查。完整逐配置/分段、R7/CANN、原生对照波动及输入哈希在 `e2e-nonquant/comparison.json` 和原始样本中。非量化采样总耗时 422.85 s。

INT8 39 配置合计均值 **0.601658**，配置/分段均无对 CANN 退化；BSND **21,600**、TND **13,500** 条样本在 `fixed-e2e-int8-{bsnd,tnd}/`，合计见 `e2e-int8-summary.json`。两组总耗时分别为 56.05/47.66 s。

## 复现与交接

每项完整命令保存在 `W/*-command.json`，环境启动器为 `/tmp/mla-m16-schedule/run.py`。`python3 "$W/run.py" acceptance` 运行完整 pytest，性能任务为 `e2e-nonquant`、`int8-bsnd`、`int8-tnd`；`python3 "$W/verify_performance.py"` 重算并核对所有性能结果。构建任务为 `build-cached`，复用的对象目录在上一轮工作根，命令已记录。

R8 的优化与此前图性能失败见 `git show a96f49b:docs/mla/status.md`；旧捕获拒绝诊断不再属于当前契约，历史图性能结论不改写。

唯一下一步：**none**。本轮已授权改动与验证完成。当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。
