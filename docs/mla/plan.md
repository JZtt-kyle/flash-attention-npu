# MLA 迭代计划

本页只记录实施顺序、产物和完成条件。**M10 的五种 512+64 布局及阶段收尾、M11 的 192/128 BSND、M12a 的紧凑 TND 已完成。** 当前能力与最近证据见 [status](status.md)，契约与验收口径见 [design](design.md)。

## 唯一下一步 M12b

一次接通 192/128 的混合 TND Q + BSND cache 完整公开行为，复用 [M11 数值组件](design.md#651-192128-通用非量化数值组件)、[M12a 紧凑 TND 适配](design.md#512-192128-tnd-prefill-记录) 和已有混合布局的 allocation/used、mask/split 调度。M12a 本轮已闭环；本轮提交后停止，不开始 M12b。

- 产物：同一提交完成混合布局 host、AICPU、Kernel、metadata、公开 real/fake/compile；支持 qv=None、独立 K/V、GQA、TND Q 分配前缀、BSND KV 容量/used、非均匀/空长度及完整 storage/stride，返回 O/LSE 和真实 FP32 accumulator。无 mask、causal、window 与自动/1/显式 split 随布局接入。
- 接口与影响：沿用固定 FA3 的 cache 调用、metadata 参数和返回契约；复用已有调度与 M11 数值核，明确 Q allocation、BSND KV batch/head、输出与归并地址。优先复用现有记录和消费者；若扩展记录，说明既有格式无法表达的语义，不按层、mask、split 或版本号拆轮。PA prefill 仍拒绝。
- 实施前准备：在本节补齐具体改动路径及共享头文件/地址/发射/构建影响，写明必跑组/命令、选择依据、拟继承 E2E 的依赖和完成条件。新增混合布局所属 CPU 固定源差分/边界/sanitizer、双 dtype 设备数值及 real/fake/compile、长度/stride/复用/生命周期/Trap；登记自己的日常 E2E，并同步既有未开放边界测试。
- 验收：统一驱动先 dry-run/collect，核对适用 native 包与廉价夹具，再跑代表及成本边界。实现固定后运行新行为和受影响路径的 E2E、普通 FA3 与公开平均性能；保持固定输入、R、阈值及计时边界，相同数学 reference 使用内容缓存。只继承身份与原始证据完整有效的 E2E，不累计历史单元或性能。
- 完成条件：完整可调用行为、必要验证和证据身份闭环，相关文档及仓库质量检查通过，提交一次后停止；必要验证受限或失败时如实交接，不提交为已闭环。

## 后续里程碑：M12–M17

每行是能力里程碑，不等于一次提交。以下布局步默认各交付一个完整行为，稳定的 mask/split 随布局接入，host/AICPU/Kernel/公开入口在同一轮完成。新数值组件可先独立闭环；进一步拆分须写明新增风险，不能以文件数、层数或开关数机械展开。精度、安全、普通回归、生命周期和适用公开性能均随入口交付。

| 里程碑 | 剩余产物与边界 | 完成条件 |
| --- | --- | --- |
| M12：prefill 非 PA TND/混合 | 剩余 M12b TND Q + BSND cache；复用 M11/M12a 数值与适配，一次接通长度、metadata、mask/split 与公开行为 | allocation/used、非均匀/空长度、stride/GQA、三 mask、自动/1/显式 split、生产精度、生命周期及公开 E2E/性能 |
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
| M11a | 192/128 通用数值组件，独立 K/V、GQA/stride、三 mask、direct/partial/128 维 merge |
| M11b | 完整 BSND prefill host/AICPU/Kernel/v27/real/fake/compile，三 mask、自动/1/显式 split、真实 accumulator、公开 E2E 与性能闭环 |
| M12a | 完整紧凑 TND prefill host/AICPU/Kernel/v28/real/fake/compile，allocation/used、GQA/stride、三 mask、自动/1/显式 split、真实 accumulator、公开 E2E 与性能闭环 |
| 工具整理、R1、R2 | 统一验收与缓存/证据管理、模块归属与共享 helper、共用前向分派、文档轮和无头交接；原 R3 的内核可读性核对合入 R2 |

```bash
git log --oneline -- docs/mla/status.md
git show feb189b:docs/mla/status.md  # M10 阶段证据
git show fd7e133:docs/mla/status.md  # M11a 数值组件证据
```

## 迭代与收尾规则

遵循 [AGENTS.md](../../AGENTS.md#一次工作只完成一个提交) 和 [design 第 7 节](design.md#7-验收口径)。实现前在当前步补齐具体影响路径、必跑组/命令及选择依据、拟继承证据与依赖、完成条件；必要验证受限或失败时保留交接，不提交为已闭环。必跑检查通过后进入文档、质量检查和一次提交收尾；仅因新改动、失败或具体未解决风险扩测，提交后停止。

用户最新约定为每成功完成五轮开发插入一次专门文档修正轮；文档轮不计开发轮数，循环随后继续开发。手动文档任务仍按当次用户要求停止，不在本轮推进功能。循环行为由版本管理脚本和工具测试保证，本页不复制其历史实施日志。
