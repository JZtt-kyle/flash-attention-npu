# MLA 迭代计划

本页只记录实施顺序、产物和完成条件。**非量化 M17、INT8 独立组件 M18 和 BSND 公开接入 M19 已完成，唯一下一开发步为 M20。** 用户于 2026-10-18 批准 M18–M21 的 INT8 扩展，无需各步重复确认；范围以 [design 3.4](design.md#34-int8-压缩-mla-扩展) 为准，当前能力、限制和证据以 [status](status.md) 为准。M19 的 graph 性能暂缓按 2026-10-19 本次明确授权处理，不沿用 M17 例外或改变后续默认门槛。

## INT8 后续步骤

只迁移 512+64 共享压缩 KV 的 INT8 前向，RoPE/输出为 BF16，公开缓存为 ND PA、页大小 128，query 为 BSND/TND。固定源旧 IFA 的 INT8 算法依赖 NZ PA，迁移时改为读取 ND 并在片上完成矩阵所需布局；不交付外部 NZ 格式、每次全 cache 转换或转调 CANN 的生产路径。无 mask/右下 causal、直接/原生 FD 沿用 M18 固定源实测边界；不把非量化的所有布局、窗口、softcap 或显式 split 自动复制成 INT8 需求。

| 步骤 | 单次提交的产物 | 完成条件 |
| --- | --- | --- |
| M18（已完成） | INT8 独立数值组件、ND 读取适配及固定源对照 | 固定源和双数学精度、INT32/舍入/缩放/FD/地址边界、适用组件性能实测；公开入口继续拒绝 INT8 |
| M19（已完成） | BSND Q + ND PA INT8 完整公开前向 | host/AICPU/Kernel/metadata/real/fake/compile/图同轮接通，必要正确性、回归及 eager 公开性能通过；graph 性能按本次授权暂缓，保留实测失败 |
| M20 | TND Q + ND PA INT8 完整公开前向 | 沿用 M18 数值组件和 M19 接入，完整 allocation/长度/scale 地址、公开图及性能验收 |
| M21 | INT8 组合覆盖与用户示例收尾 | 两布局及拒绝边界、累计 E2E、性能矩阵、示例与文档核对完成；唯一下一步改为 `none` |

### M18：INT8 数值与 ND 适配（已完成）

交付 INT8/BF16 数值组件、ND PA 读取、固定源与双数学对照、地址/舍入/空值/Trap、组件性能及共享改动的生产重建和累计回归。源码对应关系见 design 3.4，最近证据与限制见 status；原始 tiling 仅为测试夹具，不是生产 metadata。

单列组件步的独立风险是 INT32 累加、逐 token/head Q 反量化、动态 P 量化与舍入、量化 FD 及源 NZ 到目标 ND 读取；既有 FP16/BF16 组件不覆盖这些风险。后续布局、mask、split 和协议接入不再各拆一轮。执行历史、诊断和完整命令从 `git show 619787b:docs/mla/status.md` 查询。

### M19：BSND 公开 INT8

交付完整 BSND Q + ND PA INT8 前向：`flash_attn_npu_3/flash_attn_npu_interface.py` 的 kvcache/metadata 分派、三个 keyword-only scale 和 fake 校验，910 目录 `mla_forward.cpp` 的 host 注册及生命周期，`mla_int8_schedule.h`、`mla_int8_metadata.aicpu`、`mla_int8_launch.h`、`mla_dispatch_int8.cpp` 的调度与发射，以及 `block/mla_int8_vector.hpp` 的 int32 长度和真实 partial 输出适配。`setup.py` 注册生产编译单元。复用 M18 数值核、固定源旧 IFA 调度和 ND 页校验；v48 区分混合 dtype 与量化 FD 分区，scale 值及地址仍属当次调用，协议原因与安全边界见 design 3.4。

必跑集合为 `--unit int8_bsnd`（固定源调度差分、边界/溢出、ASan/UBSan、性能工具成功/失败夹具）、`detail-int8_bsnd`（含完整 source/boundaries、双数学、真实 partial、长度/stride/空值、metadata/生命周期/Trap、real/fake/compile 及固定/动态图重放），再运行累计 `--suite e2e ordinary graph-ordinary`。新增日常清单归属 `--e2e int8_bsnd`；旧 `e2e/int8` 仅保留缺失 scale 与未接通 TND 的拒绝，非 PA、有限窗口/softcap、显式多 split 等边界继续拒绝。共享 Python、host、发射与构建身份改变，M18 E2E 只作基线核对，受影响累计回归实际运行，不继承历史单元或性能。

公开性能固定 24 配置（三入口 R=1/4，与 M20 合计不超过 100），先 `int8-bsnd-profile-check`，再 `int8-bsnd-profile`，eager/graph 按配置等权均值 ≤1 分别判定。v48 消费校验跨完整向量 grid 分摊记录字；native 在调用内处理原 device Q/Q-scale/table 的必要物化，仅源 NZ pack 位于准备阶段。工具、代表及成本边界检查完成后固定实现，再执行完整采样，保留全部单例、分段和失败。用户于 2026-10-19 明确允许本次暂缓 graph 性能优化：eager、必要正确性（含图重放）、累计回归和质量检查通过后可闭环，graph 实测失败与完整原始证据保留，默认门槛和工具不变，不自动扩展该例外至 M20/M21。

收尾补齐 `tests/mla/e2e/int8_bsnd/__init__.py`，修复累计收集时与既有 `test_public.py` 的导入冲突；完整 `detail-int8_bsnd` 已重新实测。累计回归还发现旧 `e2e/int8/test_unconnected.py` 的 compile 拒绝断言依赖原始异常文案：先断言同一 eager 调用因缺少 scale 而拒绝，再检查 Dynamo 的 `Unsupported/Observed exception` 包装及 RuntimeError 类型，避免把正确拒绝误判为失败。该修正只影响测试断言，不改生产、用例选择或性能工具；先 dry-run/collect 和实跑 `--e2e int8 int8_bsnd`，通过后完整重跑 `--suite e2e ordinary graph-ordinary --device 7 --output /tmp/mla-m19/closure-regression-final`（环境包装器见 status），不以失败组的部分通过代替累计验收。已通过的完整专项按依赖与原始证据复核，源码/构建未变时不重复；本步单元与性能同样保留原实测结果，不计作新运行或历史继承。必要验收、文档和质量检查完成后一次提交，下一步 M20，提交后立即停止。

### M20：TND 公开 INT8

同一提交完成 TND Q 的 host/AICPU/metadata/Kernel/real/fake 接入、`tests/mla/e2e/int8_tnd/`、日常清单及性能；保留原 allocation 地址，Q scale 使用 `[Tq,Hq]`，源组件 LSE 的 `[Tq,Hq]` 须适配公开 `[Hq,Tq]`；不能把分配前缀、实际长度和压紧后的源索引混用。复用 M18 数值组件与 M19 PA/量化参数协议，不因布局另拆数值或协议轮。

拟新增并在注册后按 M19 顺序运行 `int8-tnd-check/source/boundaries/profile-check/profile`、`detail-int8_tnd`、`--e2e int8_tnd`。覆盖非均匀/零长度、分配间隙 poison、scale 行/head stride、ND 页尾/乱序/越界、metadata 复用及 Trap、真实/fake/compile/动态图、跨层与 stream 生命周期。TND 公共生成器沿用 FA3 可表达的完整 Q 分配段，不凭空添加 Q-used 参数。BSND INT8、相关非量化和普通 FA3 E2E 按依赖继承或重跑，新数值/地址及公开 eager/graph 性能本步实测。通过后一次提交，下一步 M21。

### M21：INT8 收尾

按 M18 冻结的源有效组合与 M19/M20 开放边界核对两布局覆盖；补独立覆盖缺口，不自动穷举所有轴。实测 INT8 两布局专项、累计日常 E2E、相关普通 FA3 与图回归，复核实际加载、来源/版权和 910B/C 配置兼容；910C 未实测如实标记。沿用 M19/M20 固定输入重跑阶段 eager/graph 性能，最多 100 配置且平均门槛不变，所有单例、分段、布局转换成本与失败保留。

添加可运行 INT8 BSND/TND 示例及对应 E2E，中英文 README 和 design 的公开接口/拒绝项/验收映射同步；status 保存唯一当前结果和证据路径。必要检查通过才单次提交；将 status 的 `mla-loop-next` 标记设为 `none`，无头脚本正常结束。性能失败或环境受限时保留同一待办和失败结果，不能为了自动运行而提交或跳过。

## 已完成步骤

这里只保留交付边界。各布局步均包含完整调用链和必要验收；非量化拒绝项见 design 第 2 节，INT8 边界见 3.4，旧步骤的命令、数量、耗时、失败修正及证据从对应提交的 status 查询。

| 步骤 | 已交付产物 |
| --- | --- |
| M00–M03 | 固定迁移交集/FA3 契约、独立双 reference 与固定源执行器、入口识别/校验/未实现拦截 |
| M04–M06 | 512+64 dense 数值与非 PA BSND/TND/混合生产前向，v1–v5、allocation/used、完整 storage/stride 与生命周期 |
| M07 | ND PA 地址/Copy/调度、v6、BSND/TND PA 公开 metadata/cache 及阶段收尾 |
| M08–M09 | 五布局右下 causal 与 window，v7–v16、公开接入及各阶段收尾 |
| M10 | partial/FD 归并与真实 accumulator；五布局显式/原生自动 split，v17–v26、公开入口及组合/性能阶段收尾 |
| M11–M14 | 192/128 独立 K/V、GQA 数值组件及五布局完整公开前向，v27–v31；三 mask、全部 split、真实 accumulator、allocation/used、stride、页安全与组合收尾 |
| M15a–f | softcap 独立数值组件及两种表示、五布局完整前向，v32–v46；三 mask、全部 split、allocation/used、分页安全与公开性能 |
| M15g1–g2 | packed/unpacked GQA 坐标与资源组件、pack_gqa/sm_margin 公开接入，v47；修复共享 softcap 比较尾部 |
| M16–M17 | 固定/动态图重放、发射优化、910B/C 配置兼容、双 dtype 用户示例及非量化阶段收尾；图性能失败按当次明确授权接受，默认门槛未变 |
| 工具整理、R1、R2 | 统一验收与缓存/证据管理、模块归属与共享 helper、共用前向分派、文档轮和无头交接；原 R3 的内核可读性核对合入 R2 |

非量化收尾为 `ebdf3eb`，M18 开发闭环为 `619787b`。组合映射见 [512+64](design.md#75-51264-组合覆盖)、[192/128](design.md#76-192128-组合覆盖) 和 [非量化收尾](design.md#77-非量化收尾映射)。历史查询：

```bash
git log --oneline -- docs/mla/status.md
git show ebdf3eb:docs/mla/status.md
git show 619787b:docs/mla/status.md
```

## 迭代与收尾规则

遵循 [AGENTS.md](../../AGENTS.md#一次工作只完成一个提交) 和 [design 第 7 节](design.md#7-验收口径)。实现前在当前步补齐实际影响路径、必跑组/命令及选择依据、拟继承证据与依赖、完成条件；验证受限或失败时保留交接，不提交为已闭环。每成功完成五轮开发插入一次专门文档修正轮；文档轮不计开发轮数，手动任务按当次用户要求停止。无头状态与计数由脚本维护，不在文档中复制历史计数。
