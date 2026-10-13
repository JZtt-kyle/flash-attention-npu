# MLA 当前状态

**M08b2 已完成，唯一下一步是 M08c1：非 PA TND causal varlen。** 本轮提交后立即停止，不开始下一步。当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。

## 当前可调用能力

BSND 非 PA causal 已接通 func、只读 cache、metadata 生成/复用及内部 Q-used，范围为 FP16/BF16、512+64、Hkv=1、split=1、右下 causal 和合法外层 stride。v7 使用 `version=kind=7`、`410+8B` 个 int32；公开 cache/metadata 不新增 Q-used 参数。TND、混合、PA causal、window 等组合仍明确拒绝。真实/fake、fullgraph compile、多线程/stream 生命周期已验收；完整图重放不在本步能力内。

v7 直接复用公共长度校验、快照和几何后执行 causal 分核，复用准备好的 SplitContext。host 按线程/设备缓存不可变平台资源及最多 16 个整容量分核候选；AICPU 每次读取当次长度、核对全部身份，只有整容量匹配才复用，否则重算。不缓存设备长度或 metadata tensor；数值核、mask、CATLASS 与 CANN 源仓未改。发射与协议见 [design 5.9](design.md#59-v7bsnd-causal-记录)。

用户已调整验收规则并落实到工具及 AGENTS：**正确性每个测试组默认最多 100 个固定 pytest 节点；公开性能最多 100 个固定输入配置，整体平均耗时比 ≤1，单例退化仅记录。** 配置内各入口/R 先等权平均，再对适用配置等权取算术平均。所选正确性用例仍须全部通过；FD 排除规则、数学阈值、R=1/4、5 次 warmup、5 轮×20 次 wall 和完成同步不变。`--full-matrix` 仅在明确要求全量时使用。

## 本轮证据

证据根 `build/mla-m08b2/recovery-20261013/`。生产冻结身份为 `candidate-linked-frozen-sources.json`，最终工具身份为 `policy-frozen-sources.json`。规则调整后仅修改 Python 选择/汇总及其测试，未改生产源码、数值测试主体、实际扩展和采样协议，继承审计见 `policy-inheritance.json`。旧规则下已经完成的检查保留，不为降低用例数重新执行。

| 检查 | 本轮结果、耗时和证据 |
| --- | --- |
| causal / v7 / shared CPU | 68 / 172 / 1,313 项通过，21.3 / 55.2 / 240.5 秒；`candidate-cpu/attempt-1791867196410184970` |
| causal / v7 / shared ASan+UBSan | 68 / 172 / 1,313 项通过，57.8 / 164.2 / 642.9 秒；`candidate-sanitize/attempt-1791867198434399600`，退出 LeakSanitizer 保留 |
| CPU 等价及安全审计 | 272 条 v7 全记录、192 条调度、1,395 份既有输入/输出文件相同；412,409 个 sanitizer 请求的退出码、stderr、输入输出行数正常；`candidate-v7-equivalence.json`、`candidate-output-equivalence.json`、`candidate-sanitizer-audit.json` |
| 构建与代表 | `candidate-linked-check/attempt-1791868200297881700`：重建 468.3 秒、runtime 20.3 秒、56 项代表 106.5 秒通过；原生预检 234.9 秒完成。CPU 驱动不调用本次 probe 补链分支，继承依据见 `candidate-linked-cpu-inheritance.json` |
| 完整公开性能 | `final-candidate/attempt-1791869205082412290/causal-bsnd-profile`：4 个原测试节点通过，1,096.0 秒；96 个配置中 56 个适用、40 个原生 FD 单列。新平均门槛通过，均值 **0.586111**，退化配置/分段均为 0，见 `candidate-average-performance.json`。原始逐例、各入口/R、分派和 profiler 均保留；此前旧门槛 404/404 段也通过 |
| causal 数值 / 组件性能 | 同一 `final-candidate`：324 项 / 2 项通过，828.3 / 248.1 秒。324 项 O/LSE 均与固定源逐元素一致；FP16 O 有 2 项源精度限制，其余数学通过，详见下文 |
| causal 公开 / Trap / 旧入口 | 254 / 24 / 444 项通过，213.7 / 340.0 / 23.1 秒；同一 `final-candidate` 的对应组 |
| ordinary / dense / nonpaged / paged 回归 | 816 / 84 / 86 / 120 项通过，27.0 / 309.9 / 732.5 / 40.3 秒；同一 `final-candidate`。10 个完整设备组共 2,158 项通过 |
| 限量与平均工具 | `policy-tools/attempt-1791874632432207500`：限量后的 100 项通过，15.1 秒；覆盖平均允许单例退化、整体失败传播、数量/确定性/分层、collect 与实跑一致、完整模式、证据失效及已有工具回归。策略专项 14 项首次通过见 `policy-unit.log`；随后新增 CLI 失败传播用例已纳入上述工具组 |
| 选择与质量 | 候选和限量清单见 `policy-collection/`、`policy-device-collection/` 及 `policy-selection-audit.json`；源代码与链接检查、提交前质量结果见 `quality-final.log`。规则调整前的全部清单与失败均保留 |

原阻塞形状 `(B,Sq,Hq,Sk)=(10,3,86,8193)` 为真实原生 template 5 非 FD。最终 FP16/BF16 的生成加 R=4 比值为 0.996985/0.996935，内部生成 R=4 为 0.991956/0.990192，func R=4 为 0.990137/0.991751；见 `candidate-full-gates.json`。平均规则由用户明确调整，不把旧失败改写为通过，也不重采样挑选结果。

## 精度限制、未运行与历史失败

- 当次数值证据 `candidate-source-precision.json`：FP16 O 为 160 项数学通过、2 项源精度限制，LSE 162 项数学通过；BF16 O/LSE 各 162 项数学通过。FP16 amplitude=4/8 的 O 误差为 0.015625，迁移与固定源逐元素一致，符合源精度优先规则；与 b1 已记录的限制相同。
- 内部 Q-used 同调度组件性能已实测；全零 Q-used 的极小 kernel 比值 FP16/BF16 为 2.4471/2.2033，含消费者校验开销。组件诊断不是公开 E2E 门槛，不能称所有组件 kernel 均不劣化。
- 用户调整验收后，旧 `shared-profile` 在 688.8 秒时有序中断，**不计通过**；尚未执行的额外无 mask varlen used/stride E2E 扩展撤销。共享数值回归已通过，未变组件的历史证据仍可按依赖复核；不再为本步扩展历史全矩阵。
- 原无头目录 `build/mla-codex-loop/20261012-115818-development-sq7w7u42` 的运行正常结束但因公开性能失败没有提交，属于验收阻塞，非 Git 锁或审批阻塞。此前完整候选 401/404 段通过（`prepared-full-gates.json`）、probe 缺少 `PrepareUsedSchedule` 符号、SAN 中断、沙箱枚举失败等均保留，未计为最终通过。probe 修复后已重建并完成代表/完整设备检查。
- 新限量收集首次在沙箱导入设备模块时无法枚举芯片，保留在 `policy-collection/`；设备组改用主机可见环境收集，不执行或重复设备测试。

## 环境与复现

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2；第 6 卡 910B3、20 AIC/40 AIV。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`；只读 CANN 源 `ca18aee610729ca5270a44e2d84c751bc687fbf0`。

实际扩展为本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `29dc6612d28e629df5399041e277f5218edb7a23525e7e13df6c9f9d11e775a0`；最终工具修改后再次核对未变。数学参考使用 `build/mla-validation-artifacts/golden`；最长 reference/最大内存策略未变，继承有效成本证据。

日常命令为 `python3 -m tests.mla_validation --output build/mla-<step> --suite <受影响组> --max-cases 100 --device 6`，先 `--dry-run` 和 `--collect-only`。本步平均重算：`python3 -m tests.mla_public_performance <final-candidate/attempt-1791869205082412290/causal-bsnd-profile/public-profile/fp16/profile.json> <同路径/bf16/profile.json> --cases-per-dtype 48 --output <average.json>`。不要 source CANN set_env 或覆盖原 PYTHONPATH；设备导入/执行需要主机设备可见性。

下一轮按 [plan](plan.md#唯一下一步-m08c1非-pa-tnd-causal-varlen) 准备 M08c1 的 TND 行映射、分配段/used、地址间隙和公共边界验证；本轮不开始实现。
