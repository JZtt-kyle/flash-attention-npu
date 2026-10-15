# MLA 迭代计划

本页只记录实施顺序、产物和完成条件。**M10 的五种 512+64 布局及阶段收尾、M11a 的 192/128 数值组件已完成，公开 prefill 尚未接入。** 当前能力与最近证据见 [status](status.md)，契约与验收口径见 [design](design.md)。本轮仅修正文档，不推进下述开发步骤。

## 唯一下一步 M11b

一次接通 192/128 BSND 的完整公开行为，范围按 [design 3.2](design.md#32-展开-mla-prefill192--128)，复用 [M11a 数值组件](design.md#651-192128-通用非量化数值组件) 和既有 mask/split 调度、校验及归并。

M11a 因通用非量化算法的独立 K/V、GQA、192 行 stride 和输出 128 风险单独闭环；这些数值风险现已有组件证据。M11b 不再按 host/AICPU/Kernel、metadata、mask 或 split 开关拆轮。

- 产物：同一提交完成 BSND host、AICPU、Kernel、metadata、公开 real/fake/compile；支持 qv=None、独立 K/V、GQA、完整 storage/stride、有效长度和空输入，输出 O/LSE 及实际 FP32 accumulator。无 mask、causal、window 与自动/1/显式 split 随布局一次接入。
- 接口和安全：沿用已确认的 FA3 形状、返回数量、动态长度校验、metadata 复用/失效及流生命周期契约；新格式必须说明既有格式不能表达的语义或安全边界，不能只因版本号新增而另拆组件步。尚未接通的 TND/混合/PA prefill 仍明确拒绝。
- 影响路径：Python real/fake 共用入口、910 `flash_api.cpp` 与 `mla/` 的校验/前向、调度/metadata/AICPU、生产 launcher 和构建清单；复用 `MlaPrefill` 与 `HeadDim=128` 归并，不能把组件的 KV-head 虚拟 batch 夹具直接当成公开 BSND 头布局。
- 验收：实现前补齐具体文件、测试选择依据、拟继承 E2E 及依赖和统一命令。新增独立 K/V、GQA/192 stride、三 mask、自动/1/显式 split、空值/有效长度、真实 accumulator、metadata 失效/复用及 real/fake/compile/生命周期/Trap 用例；协议变化补所属 CPU/sanitizer。新日常 BSND prefill E2E 注册到 `regression_cases.py`，相关累计 E2E 和普通 FA3 覆盖共享路径；不累计历史单元，组件证据不替代新入口验收。
- 性能：长测前复核工具、廉价夹具和原生通用路径配方/实际分派。现有 `mla_native_dispatch.py`、`mla_public_performance.py` 的公开适用过滤限定 template 5，须为通用 prefill 的实际分派补齐规则与成功/失败测试，不能沿用空适用集合。固定公开输入，保持 R、预热、重复次数与计时边界，完成 metadata 公开 E2E 平均门槛；M11a 的组件耗时及 merge 退化只作定位依据。
- 完成条件：以上完整行为、必要验证和证据身份均闭环，更新本步涉及的 design/plan/status，经质量检查后提交一次并停止；必要验证受限则保留交接，不提交为已闭环。

## 后续里程碑：M11–M17

每行是能力里程碑，不等于一次提交。以下布局步默认各交付一个完整行为，稳定的 mask/split 随布局接入，host/AICPU/Kernel/公开入口在同一轮完成。新数值组件可先独立闭环；进一步拆分须写明新增风险，不能以文件数、层数或开关数机械展开。精度、安全、普通回归、生命周期和适用公开性能均随入口交付。

| 里程碑 | 剩余产物与边界 | 完成条件 |
| --- | --- | --- |
| M11：192/128 BSND | M11a 数值组件已完成；M11b 一次接通 BSND 的 metadata/生产/公开调用 | qv=None、独立 K/V、GQA、192 stride、双 dtype、O/LSE/accumulator；BSND 三 mask 和自动/1/显式 split，新增公开 E2E 及适用平均性能 |
| M12：prefill 非 PA TND/混合 | M12a TND varlen，M12b TND Q + BSND cache；各复用 M11 数值和已有调度，一次接通长度、metadata、mask/split 与公开行为 | allocation/used、非均匀/空长度、stride/GQA、三 mask、自动/1/显式 split、生产精度、生命周期及公开 E2E/性能 |
| M13：prefill PA | M13a BSND PA，M13b TND PA；各一次接通页访问、metadata、三 mask、自动/1/显式 split 和公开行为 | 合法 P 代表边界、独立 K/V 与内容/RoPE stride、长度/空值/复用/生命周期、数值及 PA 公开 E2E/性能 |
| M14：prefill 组合收尾 | 核对 M11–M13 的必迁组合、补明确缺口并完成阶段验收；不再按 mask × split × 布局重复开接入轮 | 前序入口已实测必要组合；新发现的算法缺口按风险单列，已有有效 E2E 按规则继承，阶段性能实际执行 |
| M15：softcap/调度选项 | softcap 数学独立闭环；pack_gqa、sm_margin 按共享调度影响划分 | 有效数学/调度语义、资源校验及适用 E2E，不能忽略参数 |
| M16：图执行 | 固定输入重放与图内长度更新/metadata 重生成分别闭环 | eager/图一致、跨层复用、多 stream 与 fake/真实输出一致 |
| M17：全范围收尾 | 冻结能力，按覆盖矩阵验收并补用户示例；实现问题先修复再验收 | 必迁组合、精度/平均公开 E2E/普通 FA3 回归；910B/C 实现与资源兼容，实机仍用 910B；README 调用与边界 |

## 已完成步骤

这里只保留交付边界，当前拒绝项以 status 和 design 第 2 节为准。每步的命令、数量、耗时、失败修正及原始证据从该提交的 status 查询；不把历史阶段的拒绝条件当作当前限制。

| 步骤 | 已交付产物 |
| --- | --- |
| M00–M03 | 固定迁移交集/FA3 契约、独立双 reference 与固定源执行器、入口识别/校验/未实现拦截 |
| M04–M06 | 512+64 dense 数值与非 PA BSND/TND/混合生产前向，v1–v5、allocation/used、完整 storage/stride 与生命周期 |
| M07 | ND PA 地址/Copy/调度、v6、BSND/TND PA 公开 metadata/cache 及阶段收尾 |
| M08–M09 | 五布局右下 causal 与 window，v7–v16、公开接入及各阶段收尾 |
| M10 | partial/FD 归并与真实 accumulator；五布局显式/原生自动 split，v17–v26、公开入口及组合/性能阶段收尾 |
| M11a | 192/128 通用数值组件，独立 K/V、GQA/stride、三 mask、direct/partial/128 维 merge；公开 prefill 仍拒绝 |
| 工具整理、R1、R2 | 统一验收与缓存/证据管理、模块归属与共享 helper、共用前向分派、文档轮和无头交接；原 R3 的内核可读性核对合入 R2 |

```bash
git log --oneline -- docs/mla/status.md
git show feb189b:docs/mla/status.md  # M10 阶段证据
git show fd7e133:docs/mla/status.md  # M11a 数值组件证据
```

## 迭代与收尾规则

遵循 [AGENTS.md](../../AGENTS.md#一次工作只完成一个提交) 和 [design 第 7 节](design.md#7-验收口径)。实现前在当前步补齐具体影响路径、必跑组/命令及选择依据、拟继承证据与依赖、完成条件；必要验证受限或失败时保留交接，不提交为已闭环。必跑检查通过后进入文档、质量检查和一次提交收尾；仅因新改动、失败或具体未解决风险扩测，提交后停止。

用户最新约定为每成功完成五轮开发插入一次专门文档修正轮；文档轮不计开发轮数，循环随后继续开发。手动文档任务仍按当次用户要求停止，不在本轮推进功能。循环行为由版本管理脚本和工具测试保证，本页不复制其历史实施日志。
