# MLA 当前状态

<!-- mla-loop-next: none -->

**R8 已完成：普通调用优化通过验收，MLA 图模式按用户要求移除。** 基线 `36e8609`，分支 `feat/mla-cann-migration`。用户于 2026-10-20 先确认端到端优化方案，随后明确“先删除图模式相关的实现和测试”。当前支持范围为普通调用、多 stream、fake/compile；MLA NPU 捕获明确拒绝，普通 FA3 图功能保留。本轮按最终范围完成一次提交，不将历史图性能失败记为通过。

## 最终改动与身份

- 五布局自动 split 的 init/partial/merge 合为一次发射，每核保留一次完整动态校验及必要屏障；分片归属用有序端点二分查询，归并只遍历所属任务。复用原数值组件、分区与舍入。
- 私有 real/fake 链传递是否请求 accumulator，O-only 跳过无用 accumulator 的分配/初始化/写回，INT8 O-only 跳过 LSE。请求完整四项输出时契约不变。BSND/TND 共用已完成初始化的不可变 mask，保留普通跨流依赖和生命周期。
- 删除 metadata 捕获切流、INT8 冷捕获 mask 特例及 MLA 图捕获/重放测试。`DensePlatform` / `CausalPlatform` 在真实前向和 metadata 准备边界抛出 `MLA: NPU graph capture is not supported`；保留的捕获查询仅用于拒绝。原混合 stream/graph 生命周期用例改为普通多流，fake/compile 不再追加图重放。
- 证据根 `W=/tmp/flash-attention-npu-mla-46928a793876/mla-no-graph-20261020`；实际加载主树 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 **`1d28ae6eb482f1fe12ba80d8908aa6d1ccd1f663bb34e845259cc34f32dea812`**。源码/测试逐文件 SHA 在 `final-identity.json`，实际加载见 `candidate-environment.log`。删除前差异、文件和扩展另存 `before.diff`、`before/`、`before-extension.so`。

环境：Ascend910B3 物理设备 **3** / 逻辑 0，性能 CPU 48–51；Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。固定 CANN 源为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，非量化包 `/tmp/mla-m13b/native`，INT8 包 `/tmp/mla-m18/native`，性能工具导入前逐文件核对 native 包哈希。R7 对照扩展 SHA256 `92ca4037ef8a6e08632090b12bb675ac62ca3b924a332d6897d4d32006bea233`，Python 接口 SHA256 `389f35aebf6d31f143476f7eeefff49e2ba3db2aec621a19c7d780f2a6e65245`。CANN/CATLASS 未修改。

## 本次验证

| 检查 | 结果与证据 |
| --- | --- |
| 构建与加载 | 最终构建成功 / 129.91 s，实际扩展加载通过；`build-fixed.log`、`candidate-environment.log` |
| 收集差分 | **1,480 MLA**；只删除 99 个图专用节点，24 个生命周期节点改为普通多流，无其他遗漏；`collect.log`、`test-node-changes.json`、`check_collection.py` |
| 代表/成本边界 | **106 passed / 72.53 s**，含五布局、双 dtype、最长 KV、INT8、fake/compile 与多流；`representative.xml` |
| 完整公开回归 | **1,480 MLA + 140 普通 FA3 = 1,620 passed / 403.88 s**，零失败/错误/跳过；`acceptance.log` / `acceptance.xml`；普通 FA3 含 metadata、compile 与 2 个图节点 |
| 捕获拒绝 | 一次性公开入口诊断 **44 项通过 / 30.78 s**，覆盖 func/varlen/cache/metadata、两种表示、单片/自动及两种 INT8 布局；拒绝后普通调用数值检查通过；`capture-refusal.log`，仓库内不保留图测试 |
| 性能工具预检 | 非量化 10 配置、INT8 两布局各 4 配置的数值/最长分配检查通过；正常/缺失/重复/非有限统计夹具通过，见各 `*-preflight.log` 与 `stats-preflight.log` |
| 仓库质量 | 14 个本步文件的既有 pre-commit、最终链接/测试映射、源码/扩展身份及 `git diff --check` 通过；`quality-final.log`、`final-checks.log` |

数值使用独立 reference 与仓库 `assert_fa_close`，阈值未放宽。所有改名/修改用例及新扩展重新实跑，不累计此前 1,719 项或重复代表的通过数。未改的分片查询算法可参考前一候选的 2,800,000 次等价检查，不能代替本次公开 E2E。

首次全量编译成功，但加载因当前 torch_npu 未导出 `IsContextInitialized` 而失败；已改用仓库原有 `currentStreamCaptureStatusMayInitCtx` 并重建。初始失败记录在 `initial-candidate-environment.log`，不计作通过。最终全部验证均绑定上述新扩展。

## 当前普通调用性能

固定输入、5 次预热、5×20 wall、适用 implicit/generated/reuse 及 R=1/4 不变；先对配置内入口/R 等权，再对配置等权。下表 **FA3 / 固定 CANN** 耗时比门槛 ≤1，不混合阶段隐藏退化。

| 集合 | 配置数 | R7 / CANN | 当前 / CANN | 当前 / R7 | 对 CANN 退化配置/分段 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原调度 | 20 | 0.756947 | **0.771881，通过** | 1.016219 | 0 / 8 |
| 原 eager 矩阵 | 40 | 0.793327 | **0.802949，通过** | 1.016538 | 0 / 29 |
| 默认压缩路径 | 10 | 1.077746 | **0.741942，通过** | **0.685154** | 0 / 2 |
| INT8 BSND | 24 | 本次未测 | **0.592033，通过** | 本次未测 | 0 / 0 |
| INT8 TND | 15 | 本次未测 | **0.606122，通过** | 本次未测 | 0 / 0 |

默认 10 配置相对同轮 R7 对照的等权耗时平均减少 **31.48%**；原调度/eager 组分别慢 **1.62% / 1.65%**，其中 14/20 与 26/40 配置比 R7 慢，如实保留。非量化 70 配置各自的对 CANN 均值均 ≤1，但仍有上表所列分段退化。非量化两侧各 **56,000** 条样本、输入哈希、逐配置/分段和分组结果在 `e2e-nonquant/`，重新计算校验见 `e2e-summary-validation.log`。

INT8 39 配置合计均值 **0.597452**，配置/分段均无对 CANN 退化；BSND **21,600**、TND **13,500** 条样本在 `fixed-e2e-int8-{bsnd,tnd}/`，合计 `e2e-int8-summary.json`。配置清单逐项核对与删除前固定矩阵相同，只停止图采样；现有统计函数继续校验样本完整性与有限值，结果见 `performance-validation.log`。性能任务总耗时非量化 386.55 s、INT8 BSND/TND 61.93/54.39 s。

## 历史与复现

删除前候选 SHA256 `8af0dcad7d27c04304616de133e3d7bbf5e16413ce0ac68378ad8ba1bd8cc142` 的 1,719 项回归及图性能数据在 `/tmp/flash-attention-npu-mla-46928a793876/mla-e2e-opt-20261020/`：非量化图均值 **1.917826**、INT8 图均值 **1.283331** 均失败。旧前后对照也有局部变慢，A/A 不能证明它们全部是噪声。以上只适用于旧扩展，当前删除图功能不改变这些历史结论。

每项完整命令保存在 `W/*-command.json`，使用 `/tmp/mla-m16-schedule/run.py` 环境。`python3 "$W/run.py" acceptance` 在主工作树运行完整 pytest；性能任务为 `e2e-nonquant`、`int8-bsnd`、`int8-tnd`。构建、参考缓存和证据均在仓库外；不恢复已退役的 MLA 测试目录或验收框架。

唯一下一步：**none**。本轮已授权范围完成，MLA 图支持已移除，不自动恢复或继续图优化。当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。
