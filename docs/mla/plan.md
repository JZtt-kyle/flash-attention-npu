# MLA 迭代计划

本页只记录实施顺序、产物和完成条件。**M10 的五种 512+64 布局及阶段收尾、M11 的 192/128 BSND、M12 的紧凑 TND 与混合布局、M13 的 BSND/TND PA 已完成。** 当前能力与最近证据见 [status](status.md)，契约与验收口径见 [design](design.md)。

## 唯一下一步 M14

完成 192/128 prefill 的组合收尾。M11–M13 已分别交付 BSND、紧凑 TND、TND Q + BSND cache、BSND PA、TND PA 的完整公开行为；下一轮核对阶段覆盖、补明确缺口并形成阶段验收，不提前开始 M15。

- 范围与产物：核对双 dtype、五布局、三 mask、自动/1/显式 split、GQA、allocation/used、合法 stride/页大小、空值、metadata 复用及 real/fake/compile/生命周期的已开放组合；以现有日常清单、专项和原始证据建立阶段覆盖说明。新增用例必须对应独立缺口，阶段结束不自动展开全组合，也不重复按布局或开关接入。
- 影响路径：优先复用 `tests/mla/regression_cases.py`、各 `e2e/prefill_*/`、现有公开性能清单/驱动及三份文档；实现前写明实际缺口、修改路径和必跑组。只有发现新增数值算法或共享协议的独立未覆盖风险，才另列组件产物和完成条件；不全面重构已验收生产路径。
- 正确性与继承：从本轮 status 的 summary、用例清单、依赖/构建和原始证据恢复；仅依赖未变、原始证据完整的 E2E 正确性可由 `--resume` 继承。累计小输入以 `python3 -m tests.mla` 为主线，受影响普通回归用 `--suite ordinary`，缺口专项按实际影响选择；不累计历史模块单元。
- 阶段性能：在采样前固定五布局合计不超过 100 个配置（含 dtype）的清单及选择依据，复用 `prefill-bsnd-profile prefill-tnd-profile prefill-mixed-profile prefill-paged-bsnd-profile prefill-paged-tnd-profile`。先核对 native 包、工具、廉价夹具及 dry-run/collect；性能实际执行，从原始入口/R 样本按配置等权汇总，保留全部单例/分段、退化及原生 FD 单列项，门槛与计时协议不变。
- 完成条件：阶段约定的组合和未开放边界已核对，必要正确性、相关回归、阶段性能及证据有效性通过，更新 design/status/plan 并完成质量检查；提交一次后停止。验证受限或失败时如实交接，不提交为已闭环。

## 后续里程碑：M14–M17

每行是能力里程碑，不等于一次提交。以下布局步默认各交付一个完整行为，稳定的 mask/split 随布局接入，host/AICPU/Kernel/公开入口在同一轮完成。新数值组件可先独立闭环；进一步拆分须写明新增风险，不能以文件数、层数或开关数机械展开。精度、安全、普通回归、生命周期和适用公开性能均随入口交付。

| 里程碑 | 剩余产物与边界 | 完成条件 |
| --- | --- | --- |
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
| M12b | 完整 TND Q + BSND cache prefill host/AICPU/Kernel/v29/real/fake/compile，Q allocation、独立 KV stride/used、三 mask、自动/1/显式 split、真实 accumulator、公开 metadata 复用及 E2E/性能闭环；M12 完成 |
| M13a | 完整 BSND Q + ND PA KV prefill host/AICPU/Kernel/v30/real/fake/compile，独立 K/V 与 GQA、页大小/stride/used、三 mask、自动/1/显式 split、真实 accumulator、公开 E2E/性能与隔离 Trap 闭环 |
| M13b | 完整 TND Q + ND PA KV prefill host/AICPU/Kernel/v31/real/fake/compile，Q allocation/used、独立 K/V 与 GQA、页大小/stride/used、三 mask、全部 split、真实 accumulator，公开 E2E/性能及隔离 Trap 闭环；M13 完成 |
| 工具整理、R1、R2 | 统一验收与缓存/证据管理、模块归属与共享 helper、共用前向分派、文档轮和无头交接；原 R3 的内核可读性核对合入 R2 |

```bash
git log --oneline -- docs/mla/status.md
git show feb189b:docs/mla/status.md  # M10 阶段证据
git show fd7e133:docs/mla/status.md  # M11a 数值组件证据
```

## 迭代与收尾规则

遵循 [AGENTS.md](../../AGENTS.md#一次工作只完成一个提交) 和 [design 第 7 节](design.md#7-验收口径)。实现前在当前步补齐具体影响路径、必跑组/命令及选择依据、拟继承证据与依赖、完成条件；必要验证受限或失败时保留交接，不提交为已闭环。必跑检查通过后进入文档、质量检查和一次提交收尾；仅因新改动、失败或具体未解决风险扩测，提交后停止。

用户最新约定为每成功完成五轮开发插入一次专门文档修正轮；文档轮不计开发轮数，循环随后继续开发。手动文档任务仍按当次用户要求停止，不在本轮推进功能。循环行为由版本管理脚本和工具测试保证，本页不复制其历史实施日志。
